"""Margin-localized stability bounds for scalar Gaussian event thresholds.

The functions in this module encode deterministic inequalities.  They do not claim a
rough-Bergomi convergence rate on their own.  A model-level rate additionally requires
probabilistic control of coefficient errors, mesh-enrichment defects, and the bad event
where an active affine slope is too small.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

NORMAL_DENSITY_MAXIMUM = 1.0 / math.sqrt(2.0 * math.pi)


@dataclass(frozen=True)
class RatioCandidateStability:
    """Pathwise ratio-candidate errors and their good-event upper bounds."""

    denominator_floor: float
    good_event: torch.Tensor
    numerator_error: torch.Tensor
    denominator_error: torch.Tensor
    coarse_numerator_envelope: torch.Tensor
    observed_candidate_error: torch.Tensor
    candidate_error_bound: torch.Tensor
    maximum_good_event_violation: float


@dataclass(frozen=True)
class AggregateThresholdStability:
    """Common-grid plus mesh-enrichment stability of a max/min threshold."""

    kind: str
    common_candidate_error: torch.Tensor
    mesh_enrichment_defect: torch.Tensor
    threshold_error: torch.Tensor
    threshold_error_bound: torch.Tensor
    maximum_violation: float


@dataclass(frozen=True)
class DefensiveMomentBounds:
    """Upper bounds from good-event threshold moments and a bad-event probability."""

    natural_weight: float
    good_event_l1_threshold_bound: float
    good_event_l2_threshold_bound_squared: float
    bad_event_probability: float
    raw_second_moment_upper_bound: float
    dcs_second_moment_upper_bound: float


def _candidate_tensors(
    numerator_fine: torch.Tensor,
    denominator_fine: torch.Tensor,
    numerator_coarse: torch.Tensor,
    denominator_coarse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tensors = (
        numerator_fine,
        denominator_fine,
        numerator_coarse,
        denominator_coarse,
    )
    if any(tensor.ndim != 2 or tensor.shape[1] < 1 for tensor in tensors):
        raise ValueError("candidate tensors must have shape (paths, candidates)")
    if any(tensor.shape != numerator_fine.shape for tensor in tensors[1:]):
        raise ValueError("fine and coarse candidate tensors must have the same shape")
    if any(
        tensor.device != numerator_fine.device
        or tensor.dtype != numerator_fine.dtype
        or not tensor.is_floating_point()
        or not torch.isfinite(tensor).all()
        for tensor in tensors
    ):
        raise ValueError("candidate tensors must be finite matching floating tensors")
    if bool((denominator_fine <= 0.0).any()) or bool((denominator_coarse <= 0.0).any()):
        raise ValueError("candidate denominators must be strictly positive")
    return tensors


def ratio_candidate_stability(
    numerator_fine: torch.Tensor,
    denominator_fine: torch.Tensor,
    numerator_coarse: torch.Tensor,
    denominator_coarse: torch.Tensor,
    *,
    denominator_floor: float,
) -> RatioCandidateStability:
    """Evaluate ``|n_f/b_f-n_c/b_c|`` on a denominator-margin good event.

    On paths where every fine and coarse denominator is at least ``kappa``,

    ``sup_i |n_f/b_f-n_c/b_c|``
    ``<= ||n_f-n_c||_inf/kappa + ||n_c||_inf ||b_f-b_c||_inf/kappa^2``.

    No assertion is made on the complementary bad event.
    """

    fine_n, fine_b, coarse_n, coarse_b = _candidate_tensors(
        numerator_fine,
        denominator_fine,
        numerator_coarse,
        denominator_coarse,
    )
    if not math.isfinite(denominator_floor) or denominator_floor <= 0.0:
        raise ValueError("denominator_floor must be finite and positive")
    good = (torch.amin(fine_b, dim=1) >= denominator_floor) & (
        torch.amin(coarse_b, dim=1) >= denominator_floor
    )
    numerator_error = torch.amax(torch.abs(fine_n - coarse_n), dim=1)
    denominator_error = torch.amax(torch.abs(fine_b - coarse_b), dim=1)
    numerator_envelope = torch.amax(torch.abs(coarse_n), dim=1)
    observed = torch.amax(torch.abs(fine_n / fine_b - coarse_n / coarse_b), dim=1)
    bound = (
        numerator_error / denominator_floor
        + numerator_envelope * denominator_error / denominator_floor**2
    )
    violation = torch.where(good, torch.clamp(observed - bound, min=0.0), 0.0)
    return RatioCandidateStability(
        denominator_floor=denominator_floor,
        good_event=good,
        numerator_error=numerator_error,
        denominator_error=denominator_error,
        coarse_numerator_envelope=numerator_envelope,
        observed_candidate_error=observed,
        candidate_error_bound=bound,
        maximum_good_event_violation=float(torch.amax(violation)),
    )


def aggregate_threshold_stability(
    fine_all_candidates: torch.Tensor,
    fine_common_candidates: torch.Tensor,
    coarse_candidates: torch.Tensor,
    *,
    kind: str,
) -> AggregateThresholdStability:
    """Separate common-grid coefficient error from fine-grid enrichment.

    ``fine_common_candidates`` is the fine construction evaluated at the embedded
    coarse indices.  For a maximum, the enrichment defect is
    ``max(fine_all)-max(fine_common)``.  For a minimum its sign is reversed.  The
    resulting common-error-plus-defect bound is pathwise exact.
    """

    if kind not in {"max", "min"}:
        raise ValueError("kind must be max or min")
    tensors = (fine_all_candidates, fine_common_candidates, coarse_candidates)
    if any(tensor.ndim != 2 or tensor.shape[1] < 1 for tensor in tensors):
        raise ValueError("aggregate candidate tensors must be nonempty matrices")
    if fine_common_candidates.shape != coarse_candidates.shape:
        raise ValueError("common fine and coarse candidates must have the same shape")
    if fine_all_candidates.shape[0] != coarse_candidates.shape[0]:
        raise ValueError("all candidate tensors must have the same path count")
    if any(
        tensor.device != fine_all_candidates.device
        or tensor.dtype != fine_all_candidates.dtype
        or not tensor.is_floating_point()
        or not torch.isfinite(tensor).all()
        for tensor in tensors
    ):
        raise ValueError("aggregate candidates must be finite matching floating tensors")

    common_error = torch.amax(torch.abs(fine_common_candidates - coarse_candidates), dim=1)
    if kind == "max":
        fine_threshold = torch.amax(fine_all_candidates, dim=1)
        common_fine_threshold = torch.amax(fine_common_candidates, dim=1)
        coarse_threshold = torch.amax(coarse_candidates, dim=1)
        enrichment = torch.clamp(fine_threshold - common_fine_threshold, min=0.0)
    else:
        fine_threshold = torch.amin(fine_all_candidates, dim=1)
        common_fine_threshold = torch.amin(fine_common_candidates, dim=1)
        coarse_threshold = torch.amin(coarse_candidates, dim=1)
        enrichment = torch.clamp(common_fine_threshold - fine_threshold, min=0.0)
    threshold_error = torch.abs(fine_threshold - coarse_threshold)
    bound = common_error + enrichment
    violation = torch.clamp(threshold_error - bound, min=0.0)
    return AggregateThresholdStability(
        kind=kind,
        common_candidate_error=common_error,
        mesh_enrichment_defect=enrichment,
        threshold_error=threshold_error,
        threshold_error_bound=bound,
        maximum_violation=float(torch.amax(violation)),
    )


def combine_common_and_mesh_defect(
    common_candidate_bound: torch.Tensor, mesh_or_order_defect: torch.Tensor
) -> torch.Tensor:
    """Combine separately established nonnegative common and enrichment bounds."""

    if (
        common_candidate_bound.shape != mesh_or_order_defect.shape
        or not common_candidate_bound.is_floating_point()
        or mesh_or_order_defect.dtype != common_candidate_bound.dtype
        or mesh_or_order_defect.device != common_candidate_bound.device
        or not torch.isfinite(common_candidate_bound).all()
        or not torch.isfinite(mesh_or_order_defect).all()
        or bool((common_candidate_bound < 0.0).any())
        or bool((mesh_or_order_defect < 0.0).any())
    ):
        raise ValueError("threshold-bound terms must be finite matching nonnegative tensors")
    return common_candidate_bound + mesh_or_order_defect


def defensive_moment_upper_bounds(
    *,
    good_event_l1_threshold_bound: float,
    good_event_l2_threshold_bound_squared: float,
    bad_event_probability: float,
    natural_weight: float,
) -> DefensiveMomentBounds:
    """Apply defensive raw/DCS second-moment inequalities.

    The supplied good-event quantities are upper bounds on
    ``E[B 1_G]`` and ``E[B^2 1_G]`` under the target residual law.  If
    ``|A_f-A_c| <= B`` on ``G`` and the natural mixture weight is ``delta``, then

    ``E_Q[H_raw^2] <= delta^-1 (phi_max E[B 1_G] + P(G^c))`` and
    ``E_Q[H_dcs^2] <= delta^-1 (phi_max^2 E[B^2 1_G] + P(G^c))``.
    """

    values = (
        good_event_l1_threshold_bound,
        good_event_l2_threshold_bound_squared,
        bad_event_probability,
        natural_weight,
    )
    if any(not math.isfinite(value) for value in values):
        raise ValueError("moment-bound inputs must be finite")
    if good_event_l1_threshold_bound < 0.0:
        raise ValueError("good-event L1 threshold bound must be nonnegative")
    if good_event_l2_threshold_bound_squared < 0.0:
        raise ValueError("good-event squared L2 threshold bound must be nonnegative")
    if not 0.0 <= bad_event_probability <= 1.0:
        raise ValueError("bad_event_probability must lie in [0, 1]")
    if not 0.0 < natural_weight <= 1.0:
        raise ValueError("natural_weight must lie in (0, 1]")
    raw = (
        NORMAL_DENSITY_MAXIMUM * good_event_l1_threshold_bound + bad_event_probability
    ) / natural_weight
    dcs = (
        NORMAL_DENSITY_MAXIMUM**2 * good_event_l2_threshold_bound_squared + bad_event_probability
    ) / natural_weight
    return DefensiveMomentBounds(
        natural_weight=natural_weight,
        good_event_l1_threshold_bound=good_event_l1_threshold_bound,
        good_event_l2_threshold_bound_squared=good_event_l2_threshold_bound_squared,
        bad_event_probability=bad_event_probability,
        raw_second_moment_upper_bound=raw,
        dcs_second_moment_upper_bound=dcs,
    )
