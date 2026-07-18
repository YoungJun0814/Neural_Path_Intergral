"""Exact Gaussian smoothing of finite-grid rBergomi downside events.

The volatility driver, including the augmented BLP local integrals, is kept
fixed.  Only a positive one-dimensional direction in the independent price
Brownian driver is integrated analytically.  Exactness therefore requires a
deterministic time-only proposal control.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch

from src.path_integral.gaussian_smoothing import (
    decompose_gaussian_shift,
    downside_excursion_thresholds,
    orthogonal_gaussian_residual,
    positive_flat_direction,
    scaled_normal_cdf,
    scaled_normal_cdf_difference,
    validate_positive_unit_direction,
)
from src.path_integral.path_functionals import DownsideExcursionTask
from src.path_integral.rbergomi_coupling import (
    CoupledRBergomiPaths,
    RBergomiLevelPaths,
    simulate_coupled_rbergomi_adjacent,
)
from src.path_integral.rbergomi_fft import (
    simulate_coupled_rbergomi_adjacent_fft,
    simulate_rbergomi_fft,
)
from src.physics_engine import RBergomiSimulator, TwoDriverRBergomiPaths

RBergomiControl = Callable[..., torch.Tensor]
SimulationEngine = Literal["fft", "reference"]


@dataclass(frozen=True)
class AffineRBergomiLogSpot:
    """Log-spot representation ``intercept + slope * coordinate``."""

    intercept: torch.Tensor
    slope: torch.Tensor
    coordinate: torch.Tensor
    residual: torch.Tensor
    direction: torch.Tensor
    maximum_residual_projection: float
    maximum_path_reconstruction_error: float


@dataclass(frozen=True)
class SmoothedRBergomiLevel:
    """Raw and analytically smoothed contributions on one time grid."""

    threshold: torch.Tensor
    coordinate: torch.Tensor
    hard_event: torch.Tensor
    threshold_event: torch.Tensor
    raw_contribution: torch.Tensor
    smoothed_contribution: torch.Tensor


@dataclass(frozen=True)
class SmoothedRBergomiEstimate:
    """Single-grid MGVS output and exactness diagnostics."""

    level: SmoothedRBergomiLevel
    log_likelihood_perpendicular: torch.Tensor
    parallel_shift: torch.Tensor
    maximum_likelihood_reconstruction_error: float
    maximum_residual_projection: float
    maximum_path_reconstruction_error: float


@dataclass(frozen=True)
class SmoothedAdjacentRBergomiEstimate:
    """Adjacent-grid MGVS correction under the exact BLP coupling."""

    fine: SmoothedRBergomiLevel
    coarse: SmoothedRBergomiLevel
    raw_correction: torch.Tensor
    smoothed_correction: torch.Tensor
    log_likelihood_perpendicular: torch.Tensor
    parallel_shift: torch.Tensor
    maximum_likelihood_reconstruction_error: float
    maximum_residual_projection: float
    maximum_fine_path_reconstruction_error: float
    maximum_coarse_path_reconstruction_error: float


def _direction_for_sample(
    direction: torch.Tensor | None,
    *,
    steps: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    if direction is None:
        return positive_flat_direction(steps, device=reference.device, dtype=reference.dtype)
    validate_positive_unit_direction(direction, steps=steps)
    if direction.device != reference.device or direction.dtype != reference.dtype:
        raise ValueError("direction must match the sample device and dtype")
    return direction


def _validate_rho(rho: float) -> float:
    resolved = float(rho)
    if not math.isfinite(resolved) or not -1.0 < resolved < 1.0:
        raise ValueError("MGVS requires a finite rho strictly between -1 and 1")
    return resolved


def _deterministic_control_schedule(
    controls: torch.Tensor,
    *,
    declared_deterministic: bool,
    tolerance: float = 1e-12,
) -> torch.Tensor:
    if controls.ndim != 3 or controls.shape[0] < 1 or controls.shape[2] != 2:
        raise ValueError("recorded controls must have shape (paths, steps, 2)")
    if not controls.is_floating_point() or not torch.isfinite(controls).all():
        raise ValueError("recorded controls must be finite floating tensors")
    schedule = controls[0]
    maximum_error = float(torch.max(torch.abs(controls - schedule.unsqueeze(0))))
    scale = max(1.0, float(torch.max(torch.abs(schedule))))
    if maximum_error > tolerance * scale:
        raise ValueError("MGVS rejects path-dependent controls")
    if bool((schedule != 0.0).any()) and not declared_deterministic:
        raise ValueError(
            "nonzero controls require an explicit deterministic time-control declaration"
        )
    return schedule


def _validate_control_contract(control_fn: RBergomiControl | None) -> None:
    if control_fn is not None and not bool(
        getattr(control_fn, "is_deterministic_time_control", False)
    ):
        raise ValueError("MGVS accepts only controls declaring is_deterministic_time_control=True")


def affine_rbergomi_log_spot(
    *,
    spot: torch.Tensor,
    variance: torch.Tensor,
    proposal_fine_brownian_increments: torch.Tensor,
    fine_step_dt: float,
    rho: float,
    direction: torch.Tensor | None = None,
    coarse_from_fine_pairs: bool = False,
) -> AffineRBergomiLogSpot:
    """Decompose a fine or adjacent-coarse log-spot path along one W2 axis."""
    resolved_rho = _validate_rho(rho)
    if spot.ndim != 2 or variance.shape != spot.shape or spot.shape[1] < 2:
        raise ValueError("spot and variance must have shape (paths, level_steps + 1)")
    if (
        not spot.is_floating_point()
        or variance.device != spot.device
        or variance.dtype != spot.dtype
        or not torch.isfinite(spot).all()
        or not torch.isfinite(variance).all()
        or bool((spot <= 0.0).any())
        or bool((variance[:, :-1] <= 0.0).any())
    ):
        raise ValueError("spot and variance must be finite, positive, matching tensors")
    if (
        proposal_fine_brownian_increments.ndim != 3
        or proposal_fine_brownian_increments.shape[0] != spot.shape[0]
        or proposal_fine_brownian_increments.shape[2] != 2
        or proposal_fine_brownian_increments.device != spot.device
        or proposal_fine_brownian_increments.dtype != spot.dtype
        or not torch.isfinite(proposal_fine_brownian_increments).all()
    ):
        raise ValueError("fine Brownian increments must be finite and match the path sample")
    if not math.isfinite(fine_step_dt) or fine_step_dt <= 0.0:
        raise ValueError("fine_step_dt must be finite and positive")

    fine_steps = proposal_fine_brownian_increments.shape[1]
    level_steps = spot.shape[1] - 1
    expected_level_steps = fine_steps // 2 if coarse_from_fine_pairs else fine_steps
    if coarse_from_fine_pairs and fine_steps % 2 != 0:
        raise ValueError("adjacent coarse smoothing requires an even number of fine steps")
    if level_steps != expected_level_steps:
        raise ValueError("path grid is inconsistent with the supplied fine increments")

    resolved_direction = _direction_for_sample(
        direction,
        steps=fine_steps,
        reference=proposal_fine_brownian_increments,
    )
    standard_price_normal = proposal_fine_brownian_increments[:, :, 1] / math.sqrt(fine_step_dt)
    coordinate, residual = orthogonal_gaussian_residual(standard_price_normal, resolved_direction)
    if coarse_from_fine_pairs:
        direction_weight = resolved_direction.reshape(level_steps, 2).sum(dim=1)
    else:
        direction_weight = resolved_direction
    increment_slope = (
        math.sqrt(1.0 - resolved_rho**2)
        * torch.sqrt(variance[:, :-1])
        * math.sqrt(fine_step_dt)
        * direction_weight.unsqueeze(0)
    )
    slope = torch.cat(
        (
            torch.zeros(spot.shape[0], 1, device=spot.device, dtype=spot.dtype),
            torch.cumsum(increment_slope, dim=1),
        ),
        dim=1,
    )
    log_spot = torch.log(spot)
    intercept = log_spot - slope * coordinate.unsqueeze(1)
    reconstructed = intercept + slope * coordinate.unsqueeze(1)
    residual_projection = torch.sum(residual * resolved_direction, dim=1)
    return AffineRBergomiLogSpot(
        intercept=intercept,
        slope=slope,
        coordinate=coordinate,
        residual=residual,
        direction=resolved_direction,
        maximum_residual_projection=float(torch.max(torch.abs(residual_projection))),
        maximum_path_reconstruction_error=float(torch.max(torch.abs(reconstructed - log_spot))),
    )


def _likelihood_decomposition(
    *,
    proposal_fine_brownian_increments: torch.Tensor,
    controls: torch.Tensor,
    fine_step_dt: float,
    affine: AffineRBergomiLogSpot,
    declared_deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    schedule = _deterministic_control_schedule(
        controls, declared_deterministic=declared_deterministic
    )
    if schedule.shape[0] != proposal_fine_brownian_increments.shape[1]:
        raise ValueError("control schedule and Brownian grid have different lengths")
    standardized_shift = math.sqrt(fine_step_dt) * schedule[:, 1]
    shift = decompose_gaussian_shift(standardized_shift, affine.direction)
    first_driver_term = torch.sum(
        schedule[:, 0].unsqueeze(0) * proposal_fine_brownian_increments[:, :, 0],
        dim=1,
    )
    first_driver_energy = fine_step_dt * torch.sum(schedule[:, 0].square())
    orthogonal_term = torch.sum(affine.residual * shift.orthogonal_shift.unsqueeze(0), dim=1)
    orthogonal_energy = torch.sum(shift.orthogonal_shift.square())
    log_perpendicular = (
        -first_driver_term - 0.5 * first_driver_energy - orthogonal_term - 0.5 * orthogonal_energy
    )
    reconstructed = (
        log_perpendicular
        - shift.parallel_coefficient * affine.coordinate
        - 0.5 * shift.parallel_coefficient.square()
    )
    return log_perpendicular, shift.parallel_coefficient, reconstructed


def _level_estimate(
    *,
    level: RBergomiLevelPaths | TwoDriverRBergomiPaths,
    affine: AffineRBergomiLogSpot,
    task: DownsideExcursionTask,
    log_likelihood: torch.Tensor,
    log_perpendicular: torch.Tensor,
    parallel_shift: torch.Tensor,
) -> SmoothedRBergomiLevel:
    thresholds = downside_excursion_thresholds(
        affine.intercept,
        affine.slope,
        step_dt=level.step_dt,
        task=task,
    )
    hard_event = task.hard_event(level.spot, level.step_dt)
    threshold_event = affine.coordinate <= thresholds.combined
    if not torch.equal(hard_event, threshold_event):
        mismatches = int(torch.count_nonzero(hard_event != threshold_event))
        raise AssertionError(f"Gaussian threshold failed to reproduce {mismatches} hard events")
    raw = hard_event.to(level.spot.dtype) * torch.exp(log_likelihood)
    if not torch.isfinite(raw).all():
        raise FloatingPointError("raw importance-sampling contribution became nonfinite")
    smoothed = scaled_normal_cdf(
        log_perpendicular,
        thresholds.combined + parallel_shift,
    )
    return SmoothedRBergomiLevel(
        threshold=thresholds.combined,
        coordinate=affine.coordinate,
        hard_event=hard_event,
        threshold_event=threshold_event,
        raw_contribution=raw,
        smoothed_contribution=smoothed,
    )


def evaluate_smoothed_rbergomi_sample(
    sample: TwoDriverRBergomiPaths,
    *,
    task: DownsideExcursionTask,
    rho: float,
    direction: torch.Tensor | None = None,
    declared_deterministic_control: bool = False,
) -> SmoothedRBergomiEstimate:
    """Evaluate MGVS from an augmented single-grid sample."""
    if sample.proposal_brownian_increments is None or sample.controls is None:
        raise ValueError("MGVS requires record_augmented=True")
    affine = affine_rbergomi_log_spot(
        spot=sample.spot,
        variance=sample.variance,
        proposal_fine_brownian_increments=sample.proposal_brownian_increments,
        fine_step_dt=sample.step_dt,
        rho=rho,
        direction=direction,
    )
    log_perpendicular, parallel_shift, reconstructed_likelihood = _likelihood_decomposition(
        proposal_fine_brownian_increments=sample.proposal_brownian_increments,
        controls=sample.controls,
        fine_step_dt=sample.step_dt,
        affine=affine,
        declared_deterministic=declared_deterministic_control,
    )
    likelihood_error = float(torch.max(torch.abs(reconstructed_likelihood - sample.log_likelihood)))
    level = _level_estimate(
        level=sample,
        affine=affine,
        task=task,
        log_likelihood=sample.log_likelihood,
        log_perpendicular=log_perpendicular,
        parallel_shift=parallel_shift,
    )
    return SmoothedRBergomiEstimate(
        level=level,
        log_likelihood_perpendicular=log_perpendicular,
        parallel_shift=parallel_shift,
        maximum_likelihood_reconstruction_error=likelihood_error,
        maximum_residual_projection=affine.maximum_residual_projection,
        maximum_path_reconstruction_error=affine.maximum_path_reconstruction_error,
    )


def simulate_smoothed_rbergomi(
    simulator: RBergomiSimulator,
    *,
    S0: float,
    T: float,
    dt: float,
    num_paths: int,
    task: DownsideExcursionTask,
    mu: float = 0.0,
    control_fn: RBergomiControl | None = None,
    override_params: dict | None = None,
    direction: torch.Tensor | None = None,
    engine: SimulationEngine = "fft",
    dtype: torch.dtype = torch.float64,
) -> SmoothedRBergomiEstimate:
    """Simulate and analytically smooth a finite-grid rBergomi estimator."""
    _validate_control_contract(control_fn)
    params = simulator._resolved(override_params)
    if engine == "fft":
        sample = simulate_rbergomi_fft(
            simulator,
            S0=S0,
            T=T,
            dt=dt,
            num_paths=num_paths,
            mu=mu,
            control_fn=control_fn,
            override_params=override_params,
            method="fft",
            dtype=dtype,
        )
    elif engine == "reference":
        sample = simulator.simulate_controlled_two_driver(
            S0=S0,
            T=T,
            dt=dt,
            num_paths=num_paths,
            mu=mu,
            control_fn=control_fn,
            override_params=override_params,
            record_augmented=True,
            dtype=dtype,
        )
    else:
        raise ValueError("engine must be 'fft' or 'reference'")
    return evaluate_smoothed_rbergomi_sample(
        sample,
        task=task,
        rho=params["rho"],
        direction=direction,
        declared_deterministic_control=True,
    )


def evaluate_smoothed_adjacent_rbergomi_sample(
    sample: CoupledRBergomiPaths,
    *,
    task: DownsideExcursionTask,
    rho: float,
    direction: torch.Tensor | None = None,
    declared_deterministic_control: bool = False,
) -> SmoothedAdjacentRBergomiEstimate:
    """Evaluate the exact MGVS fine-minus-coarse BLP correction."""
    if sample.proposal_fine_brownian_increments is None or sample.fine_controls is None:
        raise ValueError("MGVS requires record_augmented=True")
    fine_affine = affine_rbergomi_log_spot(
        spot=sample.fine.spot,
        variance=sample.fine.variance,
        proposal_fine_brownian_increments=sample.proposal_fine_brownian_increments,
        fine_step_dt=sample.fine.step_dt,
        rho=rho,
        direction=direction,
    )
    coarse_affine = affine_rbergomi_log_spot(
        spot=sample.coarse.spot,
        variance=sample.coarse.variance,
        proposal_fine_brownian_increments=sample.proposal_fine_brownian_increments,
        fine_step_dt=sample.fine.step_dt,
        rho=rho,
        direction=fine_affine.direction,
        coarse_from_fine_pairs=True,
    )
    log_perpendicular, parallel_shift, reconstructed_likelihood = _likelihood_decomposition(
        proposal_fine_brownian_increments=sample.proposal_fine_brownian_increments,
        controls=sample.fine_controls,
        fine_step_dt=sample.fine.step_dt,
        affine=fine_affine,
        declared_deterministic=declared_deterministic_control,
    )
    fine = _level_estimate(
        level=sample.fine,
        affine=fine_affine,
        task=task,
        log_likelihood=sample.log_likelihood,
        log_perpendicular=log_perpendicular,
        parallel_shift=parallel_shift,
    )
    coarse = _level_estimate(
        level=sample.coarse,
        affine=coarse_affine,
        task=task,
        log_likelihood=sample.log_likelihood,
        log_perpendicular=log_perpendicular,
        parallel_shift=parallel_shift,
    )
    raw_correction = fine.raw_contribution - coarse.raw_contribution
    smoothed_correction = scaled_normal_cdf_difference(
        log_perpendicular,
        fine.threshold + parallel_shift,
        coarse.threshold + parallel_shift,
    )
    return SmoothedAdjacentRBergomiEstimate(
        fine=fine,
        coarse=coarse,
        raw_correction=raw_correction,
        smoothed_correction=smoothed_correction,
        log_likelihood_perpendicular=log_perpendicular,
        parallel_shift=parallel_shift,
        maximum_likelihood_reconstruction_error=float(
            torch.max(torch.abs(reconstructed_likelihood - sample.log_likelihood))
        ),
        maximum_residual_projection=max(
            fine_affine.maximum_residual_projection,
            coarse_affine.maximum_residual_projection,
        ),
        maximum_fine_path_reconstruction_error=(fine_affine.maximum_path_reconstruction_error),
        maximum_coarse_path_reconstruction_error=(coarse_affine.maximum_path_reconstruction_error),
    )


def simulate_smoothed_adjacent_rbergomi(
    simulator: RBergomiSimulator,
    *,
    S0: float,
    T: float,
    fine_steps: int,
    num_paths: int,
    task: DownsideExcursionTask,
    mu: float = 0.0,
    control_fn: RBergomiControl | None = None,
    override_params: dict | None = None,
    direction: torch.Tensor | None = None,
    engine: SimulationEngine = "fft",
    dtype: torch.dtype = torch.float64,
) -> SmoothedAdjacentRBergomiEstimate:
    """Simulate exact adjacent BLP paths and smooth their correction."""
    _validate_control_contract(control_fn)
    params = simulator._resolved(override_params)
    if engine == "fft":
        sample = simulate_coupled_rbergomi_adjacent_fft(
            simulator,
            S0=S0,
            T=T,
            fine_steps=fine_steps,
            num_paths=num_paths,
            mu=mu,
            control_fn=control_fn,
            override_params=override_params,
            method="fft",
            dtype=dtype,
        )
    elif engine == "reference":
        sample = simulate_coupled_rbergomi_adjacent(
            simulator,
            S0=S0,
            T=T,
            fine_steps=fine_steps,
            num_paths=num_paths,
            mu=mu,
            control_fn=control_fn,
            override_params=override_params,
            record_augmented=True,
            dtype=dtype,
        )
    else:
        raise ValueError("engine must be 'fft' or 'reference'")
    return evaluate_smoothed_adjacent_rbergomi_sample(
        sample,
        task=task,
        rho=params["rho"],
        direction=direction,
        declared_deterministic_control=True,
    )
