"""Small original-model corrections fitted from an independent IID guide bank.

The event fit targets ``g p``; the risk fit targets ``g**2 p**2 / q0`` for a
*frozen* original proposal ``q0``. The latter is a second-moment contribution
density, not the optimal event proposal. Neither fitting objective guarantees a
population variance improvement. Final estimates require fresh IID draws and
the full proposal density; these helpers do not qualify a reference estimator.

All logged densities are ratios to the same standard Gaussian ``p``. Bank
weights are normalized only for fitting. Their ordinary (not self-normalized)
bank mass is retained in log space alongside concentration diagnostics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from src.path_integral.gaussian_span_marginalization import (
    GaussianMixtureShiftSpec,
    component_log_q_over_p,
)
from src.path_integral.weighted_bank_mixture import fit_weighted_bank_mixture


@dataclass(frozen=True)
class BankWeightDiagnostics:
    """Diagnostics of unnormalized IID-bank weights, not uncertainty estimates."""

    bank_count: int
    log_mass: float
    ess: float
    max_fraction: float


@dataclass(frozen=True)
class ModelCorrectionFit:
    """One fitted identity-covariance mixture and its training diagnostics."""

    proposal: GaussianMixtureShiftSpec
    diagnostics: BankWeightDiagnostics
    cluster_masses: tuple[float, ...]


@dataclass(frozen=True)
class ModelCorrectionFits:
    """Matched-capacity event and frozen-original-risk fits on the same bank."""

    event: ModelCorrectionFit
    risk: ModelCorrectionFit


def _validate_tensor(value: torch.Tensor, name: str) -> None:
    if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
            or value.dtype != torch.float64 or value.requires_grad
            or not bool(torch.isfinite(value).all())):
        raise ValueError(f"{name} must be finite detached CPU float64")


def _validate_spec(spec: GaussianMixtureShiftSpec) -> None:
    if not isinstance(spec, GaussianMixtureShiftSpec):
        raise TypeError("proposal must be a GaussianMixtureShiftSpec")
    _validate_tensor(spec.means, "proposal means")
    _validate_tensor(spec.weights, "proposal weights")
    # Recheck even a frozen dataclass: its tensors can have been mutated.
    GaussianMixtureShiftSpec(spec.means, spec.weights)


def _validate_samples(samples: torch.Tensor, dimension: int) -> None:
    _validate_tensor(samples, "bank samples")
    if samples.ndim != 2 or samples.shape[0] < 1 or samples.shape[1] != dimension:
        raise ValueError("bank samples must have shape (count, proposal dimension)")


def _validate_bank(
    samples: torch.Tensor,
    log_payoff: torch.Tensor,
    log_sampling_density: torch.Tensor,
    proposal: GaussianMixtureShiftSpec,
) -> None:
    _validate_spec(proposal)
    _validate_samples(samples, proposal.dimension)
    for name, value in (("log payoff", log_payoff),
                        ("log sampling density", log_sampling_density)):
        _validate_tensor(value, name)
        if value.shape != (samples.shape[0],):
            raise ValueError(f"{name} must have shape (bank count,)")


def full_mixture_log_q_over_p(
    samples: torch.Tensor, spec: GaussianMixtureShiftSpec,
) -> torch.Tensor:
    """Exact log density ratio of *all* mixture components, not sampled labels."""

    _validate_spec(spec)
    _validate_samples(samples, spec.dimension)
    components = component_log_q_over_p(samples, spec.means)
    result = torch.logsumexp(components + spec.weights.log(), dim=1)
    if not bool(torch.isfinite(components).all()) or not bool(torch.isfinite(result).all()):
        raise FloatingPointError("full mixture log density became nonfinite")
    return result


def bank_weight_diagnostics(log_weights: torch.Tensor) -> BankWeightDiagnostics:
    """Record log ordinary bank mean, weight ESS, and largest mass fraction.

    Finite log weights are required; no nonfinite row is dropped or clipped.
    ESS/max-fraction describe this bank only and do not certify tail coverage.
    """

    _validate_tensor(log_weights, "log bank weights")
    if log_weights.ndim != 1 or log_weights.numel() < 1:
        raise ValueError("log bank weights must be a nonempty vector")
    log_total = torch.logsumexp(log_weights, dim=0)
    normalized = torch.softmax(log_weights, dim=0)
    log_mass = float(log_total) - math.log(log_weights.numel())
    ess = 1.0 / float(normalized.square().sum())
    maximum = float(normalized.max())
    if not all(math.isfinite(value) for value in (log_mass, ess, maximum)):
        raise FloatingPointError("bank weight diagnostics became nonfinite")
    return BankWeightDiagnostics(log_weights.numel(), log_mass, ess, maximum)


def _fit_correction(
    samples: torch.Tensor, log_weights: torch.Tensor, *, clusters: int,
) -> ModelCorrectionFit:
    diagnostics = bank_weight_diagnostics(log_weights)
    fit = fit_weighted_bank_mixture(
        samples, torch.softmax(log_weights, dim=0), clusters=clusters,
        covariance_rank=0, defensive_mass=0.1,
    )
    if any(component.rank != 0 for component in fit.proposal.components):
        raise RuntimeError("model correction must retain identity covariance")
    proposal = GaussianMixtureShiftSpec(
        torch.stack([component.mean for component in fit.proposal.components]),
        fit.proposal.weights.clone(),
    )
    return ModelCorrectionFit(proposal, diagnostics, fit.cluster_masses)


def fit_model_corrections(
    samples: torch.Tensor,
    log_payoff: torch.Tensor,
    log_sampling_density: torch.Tensor,
    frozen_q0: GaussianMixtureShiftSpec,
    *,
    clusters: int = 2,
) -> ModelCorrectionFits:
    """Fit event and frozen-q0 risk corrections from one independent IID ``G`` bank.

    ``log_sampling_density`` means ``log(G/p)`` on the supplied draws ``z~G``.
    The caller is responsible for bank provenance/independence; arrays alone
    cannot establish it. K is limited to 1 or 2, covariance rank is fixed at 0,
    and a designated natural component keeps weight 0.1 in each correction.
    No final or held-out observations may be included in this fitting bank.
    """

    _validate_bank(samples, log_payoff, log_sampling_density, frozen_q0)
    if (isinstance(clusters, bool) or not isinstance(clusters, int)
            or clusters not in (1, 2) or samples.shape[0] < max(2, clusters)):
        raise ValueError("at least two bank draws and one or two clusters required")
    log_q0 = full_mixture_log_q_over_p(samples, frozen_q0)
    event_log_weights = log_payoff - log_sampling_density
    risk_log_weights = 2.0 * log_payoff - log_q0 - log_sampling_density
    # Validate both before fitting either, including overflow of finite inputs.
    bank_weight_diagnostics(event_log_weights)
    bank_weight_diagnostics(risk_log_weights)
    return ModelCorrectionFits(
        event=_fit_correction(samples, event_log_weights, clusters=clusters),
        risk=_fit_correction(samples, risk_log_weights, clusters=clusters),
    )


def fit_model_correction(
    samples: torch.Tensor,
    log_payoff: torch.Tensor,
    log_sampling_density: torch.Tensor,
    frozen_q0: GaussianMixtureShiftSpec,
    *,
    kind: str,
    clusters: int = 2,
) -> ModelCorrectionFit:
    """Fit one arm, allowing the caller to measure event/risk costs separately.

    This is identical to the corresponding arm of ``fit_model_corrections``.
    ``kind`` must be ``"event"`` or ``"risk"``; the frozen-q0 denominator is
    included only in the risk target. Bank independence is a caller contract.
    """

    _validate_bank(samples, log_payoff, log_sampling_density, frozen_q0)
    if (isinstance(clusters, bool) or not isinstance(clusters, int)
            or clusters not in (1, 2) or samples.shape[0] < max(2, clusters)):
        raise ValueError("at least two bank draws and one or two clusters required")
    if kind == "event":
        terms = log_payoff - log_sampling_density
    elif kind == "risk":
        terms = (2.0 * log_payoff - full_mixture_log_q_over_p(samples, frozen_q0)
                 - log_sampling_density)
    else:
        raise ValueError("kind must be 'event' or 'risk'")
    return _fit_correction(samples, terms, clusters=clusters)


def blend_model_correction(
    frozen_q0: GaussianMixtureShiftSpec,
    correction: GaussianMixtureShiftSpec,
    alpha: float,
) -> GaussianMixtureShiftSpec:
    """Return ``(1-alpha) q0 + alpha r`` without zero-weight components.

    In particular alpha=0 returns the original spec itself, preserving an exact
    baseline. For alpha<1, the density inequality gives the worst-case bound
    ``M2(q_alpha) <= M2(q0)/(1-alpha)``; it does not promise improvement.
    """

    _validate_spec(frozen_q0)
    _validate_spec(correction)
    if (isinstance(alpha, bool) or not isinstance(alpha, (int, float))
            or not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be finite and lie in [0, 1]")
    if frozen_q0.dimension != correction.dimension:
        raise ValueError("baseline and correction dimensions differ")
    if alpha == 0.0:
        return frozen_q0
    if alpha == 1.0:
        return correction
    return GaussianMixtureShiftSpec(
        torch.cat((frozen_q0.means, correction.means)),
        torch.cat(((1.0 - alpha) * frozen_q0.weights, alpha * correction.weights)),
    )


def log_second_moment_bank(
    samples: torch.Tensor,
    log_payoff: torch.Tensor,
    log_sampling_density: torch.Tensor,
    proposal: GaussianMixtureShiftSpec,
) -> float:
    """Log ordinary IID-bank mean of ``g**2 / ((G/p) (q/p))``.

    On a fresh common bank, differences can compare proposals without a target
    normalizer. This remains an empirical diagnostic, not a reference or
    coverage certificate. When G=q, its rows are exactly squared ordinary IS
    contributions. No division by the sum of bank weights is performed.
    """

    _validate_bank(samples, log_payoff, log_sampling_density, proposal)
    terms = 2.0 * log_payoff - log_sampling_density - full_mixture_log_q_over_p(
        samples, proposal,
    )
    return bank_weight_diagnostics(terms).log_mass


def shift_spec_parameters(spec: GaussianMixtureShiftSpec) -> dict[str, Any]:
    """JSON-ready exact means/weights for a replayable identity-covariance spec."""

    _validate_spec(spec)
    return {"means": spec.means.tolist(), "weights": spec.weights.tolist()}


def shift_spec_from_parameters(parameters: dict[str, Any]) -> GaussianMixtureShiftSpec:
    """Reconstruct a detached CPU float64 spec, validating every parameter."""

    spec = GaussianMixtureShiftSpec(
        torch.tensor(parameters["means"], dtype=torch.float64),
        torch.tensor(parameters["weights"], dtype=torch.float64),
    )
    _validate_spec(spec)
    return spec
