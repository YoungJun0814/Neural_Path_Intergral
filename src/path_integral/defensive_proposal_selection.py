"""Independent exact-risk selection for frozen defensive IS proposals."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from src.path_integral.finite_rank_gaussian_transport import (
    DefensiveFiniteRankGaussianMixture,
    combine_defensive_gaussian_mixtures,
)


@dataclass(frozen=True)
class DefensiveProposalRiskEstimate:
    candidate_id: str
    second_moment_estimate: float
    standard_error: float
    simultaneous_empirical_bernstein_radius: float
    upper_confidence_bound: float


@dataclass(frozen=True)
class DefensiveProposalSelectionResult:
    proposal: DefensiveFiniteRankGaussianMixture
    selected_index: int
    selected_id: str
    estimates: tuple[DefensiveProposalRiskEstimate, ...]
    validation_samples: int
    validation_defensive_mass: float
    simultaneous_oracle_excess_bound: float
    algorithmic_work_units: float


def _batch_seed(path_seed: int, label_seed: int, role: str, index: int) -> int:
    payload = (
        f"NPI-DEFENSIVE-RISK-SELECTION\0{path_seed}\0{label_seed}"
        f"\0{role}\0{index}"
    ).encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value & ((1 << 63) - 1) or 1


def select_defensive_proposal_by_second_moment(
    proposals: tuple[DefensiveFiniteRankGaussianMixture, ...],
    candidate_ids: tuple[str, ...],
    payoff_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    sample_count: int,
    batch_size: int,
    path_seed: int,
    label_seed: int,
    confidence_delta: float = 0.05,
    payoff_work_per_sample: float = 0.0,
) -> DefensiveProposalSelectionResult:
    """Select the smallest independently estimated ordinary-IS second moment.

    Validation samples come from the exact equal-weight balance mixture ``G``.
    For each frozen candidate ``Q_j`` the identity

    ``M2(Q_j) = E_G[f(X)^2 (dP/dQ_j)(X) (dP/dG)(X)]``

    gives an unbiased risk estimate.  The validation batch is discarded before
    final inference, so selecting a frozen candidate cannot bias the subsequent
    ordinary-IS estimator conditional on the selected proposal.
    """

    if not proposals or len(proposals) != len(candidate_ids):
        raise ValueError("one unique candidate id is required per proposal")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate ids must be unique")
    dimension = proposals[0].dimension
    if any(proposal.dimension != dimension for proposal in proposals):
        raise ValueError("candidate proposals must have a common dimension")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 2
    ):
        raise ValueError("selection sample count must be an integer of at least two")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("selection batch size must be a positive integer")
    if path_seed == label_seed:
        raise ValueError("selection path and label seeds must be distinct")
    if not math.isfinite(confidence_delta) or not 0.0 < confidence_delta < 1.0:
        raise ValueError("selection confidence delta must lie in (0, 1)")
    if not math.isfinite(payoff_work_per_sample) or payoff_work_per_sample < 0.0:
        raise ValueError("payoff work per sample must be finite and nonnegative")

    candidate_count = len(proposals)
    validation = combine_defensive_gaussian_mixtures(
        proposals,
        tuple(1.0 / candidate_count for _ in proposals),
    )
    sums = torch.zeros(candidate_count, dtype=torch.float64)
    square_sums = torch.zeros(candidate_count, dtype=torch.float64)
    completed = 0
    batch_index = 0
    while completed < sample_count:
        count = min(batch_size, sample_count - completed)
        sample = validation.sample(
            count,
            path_seed=_batch_seed(path_seed, label_seed, "path", batch_index),
            label_seed=_batch_seed(path_seed, label_seed, "label", batch_index),
        )
        payoff = payoff_fn(sample.samples)
        if payoff.shape != (count,):
            raise ValueError("selection payoff function returned the wrong shape")
        if (
            payoff.device.type != "cpu"
            or payoff.dtype != torch.float64
            or not torch.isfinite(payoff).all()
            or bool((payoff < 0.0).any())
            or bool((payoff > 1.0).any())
        ):
            raise ValueError("selection payoffs must be finite CPU float64 values in [0, 1]")
        p_over_g = torch.exp(sample.log_p_over_q)
        for index, proposal in enumerate(proposals):
            p_over_q = torch.exp(-proposal.log_q_over_p(sample.samples))
            risk_units = payoff.square() * p_over_q * p_over_g
            if not torch.isfinite(risk_units).all():
                raise FloatingPointError("proposal selection risk became nonfinite")
            sums[index] += torch.sum(risk_units)
            square_sums[index] += torch.sum(risk_units.square())
        completed += count
        batch_index += 1

    means = sums / sample_count
    variances = torch.clamp(
        (square_sums - sums.square() / sample_count) / (sample_count - 1),
        min=0.0,
    )
    log_factor = math.log(2.0 * candidate_count / confidence_delta)
    estimates = []
    radii = []
    for index, (candidate_id, proposal) in enumerate(
        zip(candidate_ids, proposals, strict=True)
    ):
        bound = 1.0 / (
            validation.defensive_mass * proposal.defensive_mass
        )
        radius = math.sqrt(
            2.0 * float(variances[index]) * log_factor / sample_count
        ) + 7.0 * bound * log_factor / (3.0 * (sample_count - 1))
        radii.append(radius)
        estimates.append(
            DefensiveProposalRiskEstimate(
                candidate_id=candidate_id,
                second_moment_estimate=float(means[index]),
                standard_error=math.sqrt(float(variances[index]) / sample_count),
                simultaneous_empirical_bernstein_radius=radius,
                upper_confidence_bound=float(means[index]) + radius,
            )
        )
    selected_index = int(torch.argmin(means))
    validation_density_work = sum(
        component.dimension + component.rank
        for component in validation.components
    )
    candidate_density_work = sum(
        component.dimension + component.rank
        for proposal in proposals
        for component in proposal.components
    )
    work_per_sample = (
        dimension
        + validation_density_work
        + candidate_density_work
        + payoff_work_per_sample
    )
    return DefensiveProposalSelectionResult(
        proposal=proposals[selected_index],
        selected_index=selected_index,
        selected_id=candidate_ids[selected_index],
        estimates=tuple(estimates),
        validation_samples=sample_count,
        validation_defensive_mass=validation.defensive_mass,
        simultaneous_oracle_excess_bound=(
            radii[selected_index] + max(radii)
        ),
        algorithmic_work_units=sample_count * work_per_sample,
    )
