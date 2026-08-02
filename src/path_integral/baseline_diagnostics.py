"""Likelihood-tail diagnostics that never alter an ordinary IS estimate."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from src.path_integral.baseline_framework import (
    FrozenBaselineProposal,
    evaluate_baseline_log_q_over_p,
    sample_baseline_proposal,
)


@dataclass(frozen=True)
class BaselineLikelihoodDiagnostics:
    sample_count: int
    normalization_mean: float | None
    normalization_standard_error: float | None
    normalization_z: float | None
    effective_sample_size: float | None
    log_weight_minimum: float
    log_weight_median: float
    log_weight_q99: float
    log_weight_maximum: float
    nonfinite_weight_count: int
    underflow_weight_count: int
    normalization_moments_representable: bool
    component_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.sample_count < 2:
            raise ValueError("likelihood diagnostics require at least two samples")
        if self.nonfinite_weight_count < 0 or self.nonfinite_weight_count > self.sample_count:
            raise ValueError("invalid nonfinite weight count")
        if self.underflow_weight_count < 0 or self.underflow_weight_count > self.sample_count:
            raise ValueError("invalid underflow weight count")
        if sum(self.component_counts) not in {0, self.sample_count}:
            raise ValueError("component counts do not conserve diagnostic samples")
        summaries = (
            self.normalization_mean,
            self.normalization_standard_error,
            self.normalization_z,
            self.effective_sample_size,
        )
        tails = (
            self.log_weight_minimum,
            self.log_weight_median,
            self.log_weight_q99,
            self.log_weight_maximum,
        )
        if any(not math.isfinite(value) for value in tails):
            raise ValueError("log-weight diagnostics must remain finite")
        if self.normalization_moments_representable:
            if any(value is None or not math.isfinite(value) for value in summaries):
                raise ValueError("representable likelihood moments must be finite")
        elif any(value is not None for value in summaries):
            raise ValueError("unrepresentable likelihood moments must serialize as null")


@dataclass(frozen=True)
class CouplingFlowRoundtripDiagnostics:
    maximum_reconstruction_error: float
    maximum_log_jacobian_cancellation_error: float

    def __post_init__(self) -> None:
        values = (
            self.maximum_reconstruction_error,
            self.maximum_log_jacobian_cancellation_error,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("flow roundtrip errors must be finite and nonnegative")


def _component_counts(
    proposal: FrozenBaselineProposal, *, sample_count: int, seed: int
) -> tuple[int, ...]:
    if proposal.family != "gaussian_mixture_shift":
        return ()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    torch.randn((sample_count, proposal.dimension), generator=generator, dtype=torch.float64)
    weights = torch.tensor(proposal.component_weights, dtype=torch.float64)
    labels = torch.multinomial(
        weights,
        num_samples=sample_count,
        replacement=True,
        generator=generator,
    )
    return tuple(int(torch.sum(labels == index)) for index in range(weights.numel()))


def evaluate_baseline_likelihood_diagnostics(
    proposal: FrozenBaselineProposal,
    *,
    sample_count: int,
    seed: int,
) -> BaselineLikelihoodDiagnostics:
    """Estimate ``E_Q[dP/dQ]=1`` and weight tails on a diagnostic-only stream."""

    samples = sample_baseline_proposal(proposal, sample_count=sample_count, seed=seed)
    log_q_over_p = evaluate_baseline_log_q_over_p(samples, proposal)
    log_weight = -log_q_over_p
    maximum_log = math.log(torch.finfo(torch.float64).max)
    minimum_log = math.log(torch.finfo(torch.float64).smallest_normal) - 36.7368005696771
    nonfinite = int((log_weight > maximum_log).sum())
    underflow = int((log_weight < minimum_log).sum())
    counts = _component_counts(proposal, sample_count=sample_count, seed=seed)
    if nonfinite:
        return BaselineLikelihoodDiagnostics(
            sample_count=sample_count,
            normalization_mean=None,
            normalization_standard_error=None,
            normalization_z=None,
            effective_sample_size=None,
            log_weight_minimum=float(torch.amin(log_weight)),
            log_weight_median=float(torch.quantile(log_weight, 0.5)),
            log_weight_q99=float(torch.quantile(log_weight, 0.99)),
            log_weight_maximum=float(torch.amax(log_weight)),
            nonfinite_weight_count=nonfinite,
            underflow_weight_count=underflow,
            normalization_moments_representable=False,
            component_counts=counts,
        )
    scale = float(torch.amax(log_weight))
    scaled_weights = torch.exp(log_weight - scale)
    scaled_mean = float(torch.mean(scaled_weights))
    scaled_standard_error = math.sqrt(
        float(torch.var(scaled_weights, unbiased=True)) / sample_count
    )
    scale_factor = math.exp(scale)
    mean = scale_factor * scaled_mean
    standard_error = scale_factor * scaled_standard_error
    z_score = (
        (mean - 1.0) / standard_error
        if standard_error > 0.0
        else (0.0 if mean == 1.0 else math.copysign(math.inf, mean - 1.0))
    )
    log_sum = float(torch.logsumexp(log_weight, dim=0))
    log_sum_square = float(torch.logsumexp(2.0 * log_weight, dim=0))
    ess = math.exp(2.0 * log_sum - log_sum_square)
    representable = all(math.isfinite(value) for value in (mean, standard_error, z_score, ess))
    return BaselineLikelihoodDiagnostics(
        sample_count=sample_count,
        normalization_mean=mean if representable else None,
        normalization_standard_error=standard_error if representable else None,
        normalization_z=z_score if representable else None,
        effective_sample_size=ess if representable else None,
        log_weight_minimum=float(torch.amin(log_weight)),
        log_weight_median=float(torch.quantile(log_weight, 0.5)),
        log_weight_q99=float(torch.quantile(log_weight, 0.99)),
        log_weight_maximum=float(torch.amax(log_weight)),
        nonfinite_weight_count=0,
        underflow_weight_count=underflow,
        normalization_moments_representable=representable,
        component_counts=counts,
    )


def evaluate_coupling_flow_roundtrip(
    proposal: FrozenBaselineProposal,
    *,
    sample_count: int,
    seed: int,
) -> CouplingFlowRoundtripDiagnostics:
    """Independently apply the analytic inverse and forward flow maps."""

    if proposal.family != "coupling_flow":
        raise ValueError("roundtrip diagnostics require a coupling-flow proposal")
    samples = sample_baseline_proposal(proposal, sample_count=sample_count, seed=seed)
    location = torch.tensor(proposal.location, dtype=torch.float64)
    scale_matrix = torch.tensor(proposal.flow_scale_matrix, dtype=torch.float64)
    scale_bias = torch.tensor(proposal.flow_scale_bias, dtype=torch.float64)
    shift_matrix = torch.tensor(proposal.flow_shift_matrix, dtype=torch.float64)
    shift_bias = torch.tensor(proposal.flow_shift_bias, dtype=torch.float64)
    split = proposal.flow_split
    first = samples[:, :split] - location[:split]
    log_scale = proposal.flow_max_log_scale * torch.tanh(first @ scale_matrix.T + scale_bias)
    shift = first @ shift_matrix.T + shift_bias
    latent_second = (samples[:, split:] - location[split:] - shift) * torch.exp(-log_scale)
    reconstructed_second = torch.exp(log_scale) * latent_second + shift
    reconstructed = torch.cat((first, reconstructed_second), dim=1) + location
    inverse_log_jacobian = -torch.sum(log_scale, dim=1)
    forward_log_jacobian = torch.sum(log_scale, dim=1)
    return CouplingFlowRoundtripDiagnostics(
        maximum_reconstruction_error=float(torch.max(torch.abs(reconstructed - samples))),
        maximum_log_jacobian_cancellation_error=float(
            torch.max(torch.abs(inverse_log_jacobian + forward_log_jacobian))
        ),
    )
