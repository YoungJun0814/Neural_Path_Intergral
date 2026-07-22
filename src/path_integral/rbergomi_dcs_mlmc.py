"""Generic DCS-MGI adapters for finite-grid rBergomi level corrections."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeAlias

import torch

from src.path_integral.control_span_smoothing import rank_one_price_control_span
from src.path_integral.gaussian_smoothing import (
    downside_excursion_thresholds,
    positive_flat_direction,
)
from src.path_integral.gaussian_span_marginalization import (
    GaussianMixtureShiftSpec,
    MarginalLikelihoodEvaluation,
    build_orthonormal_control_span,
    evaluate_marginal_likelihood,
)
from src.path_integral.path_functionals import (
    DiscreteBarrierHitTask,
    DownsideExcursionTask,
    TerminalThresholdTask,
)
from src.path_integral.rbergomi_mixture import RBergomiMixtureSample
from src.path_integral.rbergomi_multilevel import CoupledRBergomiMixtureSample
from src.path_integral.rbergomi_smoothing import affine_rbergomi_log_spot
from src.path_integral.stable_gaussian import (
    scaled_normal_cdf,
    scaled_normal_cdf_difference,
)

RBergomiPathTask: TypeAlias = (
    TerminalThresholdTask | DiscreteBarrierHitTask | DownsideExcursionTask
)


@dataclass(frozen=True)
class RBergomiDCSLevelEvaluation:
    """One finite-grid raw/marginalized event estimate and exactness diagnostics."""

    task_kind: str
    log_spot_intercept: torch.Tensor
    log_spot_slope: torch.Tensor
    threshold: torch.Tensor
    coordinate: torch.Tensor
    hard_event: torch.Tensor
    threshold_event: torch.Tensor
    raw_contribution: torch.Tensor
    marginalized_contribution: torch.Tensor
    density: MarginalLikelihoodEvaluation
    maximum_path_reconstruction_error: float
    maximum_legacy_component_density_error: float
    maximum_legacy_mixture_density_error: float
    maximum_legacy_full_likelihood_error: float


@dataclass(frozen=True)
class RBergomiDCSAdjacentEvaluation:
    """One exact fine-minus-coarse DCS-MGI correction."""

    fine: RBergomiDCSLevelEvaluation
    coarse: RBergomiDCSLevelEvaluation
    raw_correction: torch.Tensor
    marginalized_correction: torch.Tensor
    threshold_difference: torch.Tensor
    maximum_coordinate_mismatch: float


def _task_kind(task: RBergomiPathTask) -> str:
    if isinstance(task, TerminalThresholdTask):
        return "terminal_threshold"
    if isinstance(task, DiscreteBarrierHitTask):
        return "discrete_barrier_hit"
    if isinstance(task, DownsideExcursionTask):
        return "hit_plus_occupation"
    raise TypeError("unsupported rBergomi path task")


def scalar_task_threshold(
    log_spot_intercept: torch.Tensor,
    log_spot_slope: torch.Tensor,
    *,
    step_dt: float,
    task: RBergomiPathTask,
    slope_tolerance: float = 1e-14,
) -> torch.Tensor:
    """Return the exact scalar-normal threshold for a supported finite-grid task."""

    if (
        log_spot_intercept.ndim != 2
        or log_spot_intercept.shape != log_spot_slope.shape
        or log_spot_intercept.shape[1] < 2
    ):
        raise ValueError("intercept and slope must have shape (paths, steps + 1)")
    if (
        log_spot_intercept.device != log_spot_slope.device
        or log_spot_intercept.dtype != log_spot_slope.dtype
        or not log_spot_intercept.is_floating_point()
        or not log_spot_slope.is_floating_point()
        or not torch.isfinite(log_spot_intercept).all()
        or not torch.isfinite(log_spot_slope).all()
    ):
        raise ValueError("intercept and slope must be finite matching floating tensors")
    if not math.isfinite(step_dt) or step_dt <= 0.0:
        raise ValueError("step_dt must be finite and positive")
    if not math.isfinite(slope_tolerance) or slope_tolerance < 0.0:
        raise ValueError("slope tolerance must be finite and nonnegative")
    if float(torch.max(torch.abs(log_spot_slope[:, 0]))) > slope_tolerance:
        raise ValueError("time-zero slope must be zero")
    monitoring_slope = log_spot_slope[:, 1:]
    if bool((monitoring_slope <= 0.0).any()):
        raise ValueError("all post-initial slopes must be strictly positive")

    if isinstance(task, DownsideExcursionTask):
        return downside_excursion_thresholds(
            log_spot_intercept,
            log_spot_slope,
            step_dt=step_dt,
            task=task,
            slope_tolerance=slope_tolerance,
        ).combined
    if isinstance(task, TerminalThresholdTask):
        return (
            math.log(task.level) - log_spot_intercept[:, -1]
        ) / log_spot_slope[:, -1]
    if isinstance(task, DiscreteBarrierHitTask):
        candidates = (
            math.log(task.barrier) - log_spot_intercept[:, 1:]
        ) / monitoring_slope
        threshold = torch.max(candidates, dim=1).values
        initially_hit = log_spot_intercept[:, 0] <= math.log(task.barrier)
        return torch.where(
            initially_hit, torch.full_like(threshold, math.inf), threshold
        )
    raise TypeError("unsupported rBergomi path task")


def _deterministic_schedules(
    all_expert_controls: torch.Tensor,
    *,
    tolerance: float,
) -> torch.Tensor:
    if all_expert_controls.ndim != 4 or all_expert_controls.shape[3] != 2:
        raise ValueError("expert controls must have shape (paths, experts, steps, 2)")
    if all_expert_controls.shape[0] < 1 or all_expert_controls.shape[1] < 1:
        raise ValueError("expert controls require paths and experts")
    if not all_expert_controls.is_floating_point() or not torch.isfinite(
        all_expert_controls
    ).all():
        raise ValueError("expert controls must be finite and floating point")
    schedules = all_expert_controls[0]
    error = float(torch.max(torch.abs(all_expert_controls - schedules.unsqueeze(0))))
    scale = max(1.0, float(torch.max(torch.abs(schedules))))
    if error > tolerance * scale:
        raise ValueError("DCS-MGI rejects path-dependent controls")
    return schedules


def _generic_density(
    *,
    target_increments: torch.Tensor,
    schedules: torch.Tensor,
    weights: torch.Tensor,
    step_dt: float,
    tolerance: float,
) -> tuple[MarginalLikelihoodEvaluation, torch.Tensor]:
    if target_increments.ndim != 3 or target_increments.shape[2] != 2:
        raise ValueError("target increments must have shape (paths, steps, 2)")
    if target_increments.shape[1:] != schedules.shape[1:]:
        raise ValueError("target increments and schedules use different grids")
    price_shift_norm = float(
        torch.linalg.vector_norm(math.sqrt(step_dt) * schedules[:, :, 1])
    )
    if price_shift_norm <= tolerance:
        direction = positive_flat_direction(
            schedules.shape[1], device=schedules.device, dtype=schedules.dtype
        )
    else:
        direction = rank_one_price_control_span(
            schedules, step_dt=step_dt, tolerance=tolerance
        ).direction
    dimension = int(target_increments.shape[1] * target_increments.shape[2])
    means = math.sqrt(step_dt) * schedules.reshape(schedules.shape[0], dimension)
    spec = GaussianMixtureShiftSpec(means, weights)
    full_basis = torch.zeros(
        (target_increments.shape[1], 2),
        device=target_increments.device,
        dtype=target_increments.dtype,
    )
    full_basis[:, 1] = direction
    span = build_orthonormal_control_span(
        spec, full_basis.reshape(dimension, 1), tolerance=tolerance
    )
    standardized_target = target_increments.reshape(target_increments.shape[0], dimension)
    standardized_target = standardized_target / math.sqrt(step_dt)
    density = evaluate_marginal_likelihood(
        standardized_target, spec, span, tolerance=tolerance
    )
    return density, direction


def _level_evaluation(
    *,
    spot: torch.Tensor,
    variance: torch.Tensor,
    step_dt: float,
    target_fine_increments: torch.Tensor,
    fine_step_dt: float,
    direction: torch.Tensor,
    rho: float,
    task: RBergomiPathTask,
    density: MarginalLikelihoodEvaluation,
    legacy_component_log_q_over_p: torch.Tensor,
    legacy_mixture_log_q_over_p: torch.Tensor,
    legacy_mixture_log_likelihood: torch.Tensor,
    coarse_from_fine_pairs: bool,
) -> RBergomiDCSLevelEvaluation:
    affine = affine_rbergomi_log_spot(
        spot=spot,
        variance=variance,
        proposal_fine_brownian_increments=target_fine_increments,
        fine_step_dt=fine_step_dt,
        rho=rho,
        direction=direction,
        coarse_from_fine_pairs=coarse_from_fine_pairs,
    )
    threshold = scalar_task_threshold(
        affine.intercept,
        affine.slope,
        step_dt=step_dt,
        task=task,
    )
    hard_event = task.hard_event(spot, step_dt)
    threshold_event = affine.coordinate <= threshold
    if not torch.equal(hard_event, threshold_event):
        mismatches = int(torch.count_nonzero(hard_event != threshold_event))
        raise AssertionError(f"scalar task threshold failed on {mismatches} paths")
    if not torch.allclose(affine.coordinate, density.coordinate[:, 0], atol=2e-13, rtol=0.0):
        raise AssertionError("affine price coordinate and generic span coordinate differ")
    raw = hard_event.to(spot.dtype) * density.full_likelihood
    marginalized = scaled_normal_cdf(density.residual_log_likelihood, threshold)
    if not torch.isfinite(raw).all() or not torch.isfinite(marginalized).all():
        raise FloatingPointError("rBergomi DCS contribution became nonfinite")
    return RBergomiDCSLevelEvaluation(
        task_kind=_task_kind(task),
        log_spot_intercept=affine.intercept,
        log_spot_slope=affine.slope,
        threshold=threshold,
        coordinate=affine.coordinate,
        hard_event=hard_event,
        threshold_event=threshold_event,
        raw_contribution=raw,
        marginalized_contribution=marginalized,
        density=density,
        maximum_path_reconstruction_error=affine.maximum_path_reconstruction_error,
        maximum_legacy_component_density_error=float(
            torch.max(
                torch.abs(
                    density.full_component_log_q_over_p
                    - legacy_component_log_q_over_p
                )
            )
        ),
        maximum_legacy_mixture_density_error=float(
            torch.max(
                torch.abs(density.full_log_q_over_p - legacy_mixture_log_q_over_p)
            )
        ),
        maximum_legacy_full_likelihood_error=float(
            torch.max(
                torch.abs(density.full_log_likelihood - legacy_mixture_log_likelihood)
            )
        ),
    )


def evaluate_rbergomi_dcs_level(
    sample: RBergomiMixtureSample,
    *,
    task: RBergomiPathTask,
    rho: float,
    tolerance: float = 1e-10,
) -> RBergomiDCSLevelEvaluation:
    """Evaluate generic DCS-MGI on a single-grid rBergomi mixture sample."""

    target = sample.paths.target_brownian_increments
    if target is None:
        raise ValueError("target Brownian increments must be recorded")
    schedules = _deterministic_schedules(
        sample.all_expert_controls, tolerance=tolerance
    )
    density, direction = _generic_density(
        target_increments=target,
        schedules=schedules,
        weights=sample.weights,
        step_dt=sample.paths.step_dt,
        tolerance=tolerance,
    )
    return _level_evaluation(
        spot=sample.paths.spot,
        variance=sample.paths.variance,
        step_dt=sample.paths.step_dt,
        target_fine_increments=target,
        fine_step_dt=sample.paths.step_dt,
        direction=direction,
        rho=rho,
        task=task,
        density=density,
        legacy_component_log_q_over_p=sample.component_log_q_over_p,
        legacy_mixture_log_q_over_p=sample.log_mixture_q_over_p,
        legacy_mixture_log_likelihood=sample.mixture_log_likelihood,
        coarse_from_fine_pairs=False,
    )


def evaluate_rbergomi_dcs_adjacent(
    sample: CoupledRBergomiMixtureSample,
    *,
    task: RBergomiPathTask,
    rho: float,
    tolerance: float = 1e-10,
) -> RBergomiDCSAdjacentEvaluation:
    """Evaluate a common-coordinate exact DCS-MGI adjacent correction."""

    target = sample.paths.target_fine_brownian_increments
    if target is None:
        raise ValueError("target fine Brownian increments must be recorded")
    schedules = _deterministic_schedules(
        sample.all_expert_controls, tolerance=tolerance
    )
    density, direction = _generic_density(
        target_increments=target,
        schedules=schedules,
        weights=sample.weights,
        step_dt=sample.paths.fine.step_dt,
        tolerance=tolerance,
    )
    common = {
        "target_fine_increments": target,
        "fine_step_dt": sample.paths.fine.step_dt,
        "direction": direction,
        "rho": rho,
        "task": task,
        "density": density,
        "legacy_component_log_q_over_p": sample.component_log_q_over_p,
        "legacy_mixture_log_q_over_p": sample.log_mixture_q_over_p,
        "legacy_mixture_log_likelihood": sample.mixture_log_likelihood,
    }
    fine = _level_evaluation(
        spot=sample.paths.fine.spot,
        variance=sample.paths.fine.variance,
        step_dt=sample.paths.fine.step_dt,
        coarse_from_fine_pairs=False,
        **common,
    )
    coarse = _level_evaluation(
        spot=sample.paths.coarse.spot,
        variance=sample.paths.coarse.variance,
        step_dt=sample.paths.coarse.step_dt,
        coarse_from_fine_pairs=True,
        **common,
    )
    coordinate_mismatch = float(torch.max(torch.abs(fine.coordinate - coarse.coordinate)))
    if coordinate_mismatch > tolerance:
        raise AssertionError("fine and coarse integrated coordinates differ")
    raw_correction = fine.raw_contribution - coarse.raw_contribution
    marginalized_correction = scaled_normal_cdf_difference(
        density.residual_log_likelihood,
        fine.threshold,
        coarse.threshold,
    )
    return RBergomiDCSAdjacentEvaluation(
        fine=fine,
        coarse=coarse,
        raw_correction=raw_correction,
        marginalized_correction=marginalized_correction,
        threshold_difference=fine.threshold - coarse.threshold,
        maximum_coordinate_mismatch=coordinate_mismatch,
    )
