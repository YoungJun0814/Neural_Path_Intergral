"""Pathwise threshold-coupling diagnostics for adjacent rBergomi grids.

This module deliberately diagnoses deterministic finite-grid identities only.  It
does not infer a strong rate from one sample, and it does not turn a pathwise bound
into a probabilistic convergence theorem without separate moment and bad-event
control.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from src.path_integral.path_functionals import (
    DiscreteBarrierHitTask,
    TerminalThresholdTask,
)
from src.path_integral.threshold_stability import (
    aggregate_threshold_stability,
    combine_common_and_mesh_defect,
    ratio_candidate_stability,
)


@dataclass(frozen=True)
class RBergomiThresholdCouplingDiagnostics:
    """Coefficient, candidate, and mesh terms in an adjacent threshold error."""

    task_kind: str
    denominator_floor: float
    fine_threshold: torch.Tensor
    coarse_threshold: torch.Tensor
    signed_threshold_difference: torch.Tensor
    threshold_error: torch.Tensor
    finite_threshold: torch.Tensor
    initially_hit: torch.Tensor
    good_event: torch.Tensor
    numerator_error: torch.Tensor
    denominator_error: torch.Tensor
    coarse_numerator_envelope: torch.Tensor
    common_candidate_error: torch.Tensor
    common_candidate_error_bound: torch.Tensor
    mesh_enrichment_defect: torch.Tensor
    threshold_error_bound: torch.Tensor
    fine_active_index: torch.Tensor
    coarse_active_index: torch.Tensor
    fine_active_time: torch.Tensor
    coarse_active_time: torch.Tensor
    maximum_good_event_bound_violation: float
    maximum_exact_decomposition_violation: float


def _validate_adjacent_affine_paths(
    fine_intercept: torch.Tensor,
    fine_slope: torch.Tensor,
    coarse_intercept: torch.Tensor,
    coarse_slope: torch.Tensor,
    *,
    fine_step_dt: float,
    coarse_step_dt: float,
    slope_tolerance: float,
) -> tuple[int, int]:
    tensors = (fine_intercept, fine_slope, coarse_intercept, coarse_slope)
    if any(tensor.ndim != 2 or tensor.shape[1] < 2 for tensor in tensors):
        raise ValueError("affine paths must have shape (paths, steps + 1)")
    if fine_intercept.shape != fine_slope.shape:
        raise ValueError("fine intercept and slope shapes differ")
    if coarse_intercept.shape != coarse_slope.shape:
        raise ValueError("coarse intercept and slope shapes differ")
    if fine_intercept.shape[0] != coarse_intercept.shape[0]:
        raise ValueError("fine and coarse path counts differ")
    if any(
        tensor.device != fine_intercept.device
        or tensor.dtype != fine_intercept.dtype
        or not tensor.is_floating_point()
        or not torch.isfinite(tensor).all()
        for tensor in tensors
    ):
        raise ValueError("affine paths must be finite matching floating tensors")
    if not math.isfinite(fine_step_dt) or fine_step_dt <= 0.0:
        raise ValueError("fine_step_dt must be finite and positive")
    if not math.isfinite(coarse_step_dt) or coarse_step_dt <= 0.0:
        raise ValueError("coarse_step_dt must be finite and positive")
    if not math.isfinite(slope_tolerance) or slope_tolerance < 0.0:
        raise ValueError("slope_tolerance must be finite and nonnegative")

    fine_steps = fine_intercept.shape[1] - 1
    coarse_steps = coarse_intercept.shape[1] - 1
    if fine_steps != 2 * coarse_steps:
        raise ValueError("the fine grid must contain exactly twice the coarse steps")
    if not math.isclose(coarse_step_dt, 2.0 * fine_step_dt, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError("coarse_step_dt must equal twice fine_step_dt")
    if not torch.allclose(fine_intercept[:, 0], coarse_intercept[:, 0], atol=1e-12, rtol=1e-12):
        raise ValueError("fine and coarse initial log spots differ")
    if (
        float(torch.amax(torch.abs(fine_slope[:, 0]))) > slope_tolerance
        or float(torch.amax(torch.abs(coarse_slope[:, 0]))) > slope_tolerance
    ):
        raise ValueError("time-zero affine slopes must be zero")
    if bool((fine_slope[:, 1:] <= 0.0).any()) or bool((coarse_slope[:, 1:] <= 0.0).any()):
        raise ValueError("all post-initial affine slopes must be strictly positive")
    return fine_steps, coarse_steps


def evaluate_rbergomi_threshold_coupling(
    fine_intercept: torch.Tensor,
    fine_slope: torch.Tensor,
    coarse_intercept: torch.Tensor,
    coarse_slope: torch.Tensor,
    *,
    fine_step_dt: float,
    coarse_step_dt: float,
    task: TerminalThresholdTask | DiscreteBarrierHitTask,
    denominator_floor: float,
    slope_tolerance: float = 1e-14,
) -> RBergomiThresholdCouplingDiagnostics:
    """Decompose an adjacent terminal/barrier threshold error path by path.

    The common-grid candidate error is bounded on the event where every relevant
    affine slope is at least ``denominator_floor``.  For a barrier, the remaining
    term is the exact nonnegative increase caused by fine-only monitoring times.
    The returned theoretical bound is asserted only on ``good_event``.
    """

    fine_steps, coarse_steps = _validate_adjacent_affine_paths(
        fine_intercept,
        fine_slope,
        coarse_intercept,
        coarse_slope,
        fine_step_dt=fine_step_dt,
        coarse_step_dt=coarse_step_dt,
        slope_tolerance=slope_tolerance,
    )
    if not math.isfinite(denominator_floor) or denominator_floor <= 0.0:
        raise ValueError("denominator_floor must be finite and positive")

    path_count = fine_intercept.shape[0]
    device = fine_intercept.device
    dtype = fine_intercept.dtype

    if isinstance(task, TerminalThresholdTask):
        fine_n = (math.log(task.level) - fine_intercept[:, -1]).unsqueeze(1)
        fine_b = fine_slope[:, -1:].clone()
        coarse_n = (math.log(task.level) - coarse_intercept[:, -1]).unsqueeze(1)
        coarse_b = coarse_slope[:, -1:].clone()
        ratio = ratio_candidate_stability(
            fine_n,
            fine_b,
            coarse_n,
            coarse_b,
            denominator_floor=denominator_floor,
        )
        fine_threshold = (fine_n / fine_b)[:, 0]
        coarse_threshold = (coarse_n / coarse_b)[:, 0]
        mesh_defect = torch.zeros(path_count, device=device, dtype=dtype)
        finite_threshold = torch.ones(path_count, device=device, dtype=torch.bool)
        initially_hit = torch.zeros(path_count, device=device, dtype=torch.bool)
        fine_active_index = torch.full((path_count,), fine_steps, device=device, dtype=torch.long)
        coarse_active_index = torch.full(
            (path_count,), coarse_steps, device=device, dtype=torch.long
        )
        task_kind = "terminal_threshold"
    elif isinstance(task, DiscreteBarrierHitTask):
        log_barrier = math.log(task.barrier)
        initially_hit = fine_intercept[:, 0] <= log_barrier
        coarse_initially_hit = coarse_intercept[:, 0] <= log_barrier
        if not torch.equal(initially_hit, coarse_initially_hit):
            raise AssertionError("fine and coarse initial barrier events differ")
        finite_threshold = ~initially_hit
        fine_n_all = log_barrier - fine_intercept[:, 1:]
        fine_b_all = fine_slope[:, 1:]
        fine_n = log_barrier - fine_intercept[:, 2::2]
        fine_b = fine_slope[:, 2::2]
        coarse_n = log_barrier - coarse_intercept[:, 1:]
        coarse_b = coarse_slope[:, 1:]
        ratio = ratio_candidate_stability(
            fine_n,
            fine_b,
            coarse_n,
            coarse_b,
            denominator_floor=denominator_floor,
        )
        fine_candidates = fine_n_all / fine_b_all
        fine_common_candidates = fine_n / fine_b
        coarse_candidates = coarse_n / coarse_b
        aggregate = aggregate_threshold_stability(
            fine_candidates,
            fine_common_candidates,
            coarse_candidates,
            kind="max",
        )
        if not torch.allclose(
            aggregate.common_candidate_error,
            ratio.observed_candidate_error,
            atol=5e-13,
            rtol=5e-13,
        ):
            raise AssertionError("common-grid candidate diagnostics disagree")
        finite_fine_threshold, fine_argmax = torch.max(fine_candidates, dim=1)
        finite_coarse_threshold, coarse_argmax = torch.max(coarse_candidates, dim=1)
        fine_threshold = torch.where(
            initially_hit,
            torch.full_like(finite_fine_threshold, math.inf),
            finite_fine_threshold,
        )
        coarse_threshold = torch.where(
            initially_hit,
            torch.full_like(finite_coarse_threshold, math.inf),
            finite_coarse_threshold,
        )
        mesh_defect = torch.where(
            initially_hit,
            torch.zeros_like(aggregate.mesh_enrichment_defect),
            aggregate.mesh_enrichment_defect,
        )
        fine_active_index = torch.where(initially_hit, 0, fine_argmax + 1)
        coarse_active_index = torch.where(initially_hit, 0, coarse_argmax + 1)
        task_kind = "discrete_barrier_hit"
    else:
        raise TypeError("threshold coupling supports terminal and discrete barrier tasks")

    signed_difference = torch.where(
        finite_threshold,
        fine_threshold - coarse_threshold,
        torch.zeros_like(fine_threshold),
    )
    threshold_error = torch.abs(signed_difference)
    common_candidate_error = torch.where(
        finite_threshold,
        ratio.observed_candidate_error,
        torch.zeros_like(ratio.observed_candidate_error),
    )
    common_candidate_bound = torch.where(
        finite_threshold,
        ratio.candidate_error_bound,
        torch.zeros_like(ratio.candidate_error_bound),
    )
    good_event = ratio.good_event & finite_threshold
    threshold_bound = combine_common_and_mesh_defect(common_candidate_bound, mesh_defect)
    good_violation = torch.where(
        good_event,
        torch.clamp(threshold_error - threshold_bound, min=0.0),
        torch.zeros_like(threshold_error),
    )
    exact_bound = combine_common_and_mesh_defect(common_candidate_error, mesh_defect)
    exact_violation = torch.clamp(threshold_error - exact_bound, min=0.0)
    fine_active_time = fine_active_index.to(dtype=dtype) * fine_step_dt
    coarse_active_time = coarse_active_index.to(dtype=dtype) * coarse_step_dt

    return RBergomiThresholdCouplingDiagnostics(
        task_kind=task_kind,
        denominator_floor=denominator_floor,
        fine_threshold=fine_threshold,
        coarse_threshold=coarse_threshold,
        signed_threshold_difference=signed_difference,
        threshold_error=threshold_error,
        finite_threshold=finite_threshold,
        initially_hit=initially_hit,
        good_event=good_event,
        numerator_error=ratio.numerator_error,
        denominator_error=ratio.denominator_error,
        coarse_numerator_envelope=ratio.coarse_numerator_envelope,
        common_candidate_error=common_candidate_error,
        common_candidate_error_bound=common_candidate_bound,
        mesh_enrichment_defect=mesh_defect,
        threshold_error_bound=threshold_bound,
        fine_active_index=fine_active_index,
        coarse_active_index=coarse_active_index,
        fine_active_time=fine_active_time,
        coarse_active_time=coarse_active_time,
        maximum_good_event_bound_violation=float(torch.amax(good_violation)),
        maximum_exact_decomposition_violation=float(torch.amax(exact_violation)),
    )
