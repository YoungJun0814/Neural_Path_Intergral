"""Stable one-dimensional Gaussian smoothing for monotone path events."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from src.path_integral.path_functionals import DownsideExcursionTask
from src.path_integral.stable_gaussian import (
    scaled_normal_cdf as _scaled_normal_cdf,
)
from src.path_integral.stable_gaussian import (
    scaled_normal_cdf_difference as _scaled_normal_cdf_difference,
)
from src.path_integral.stable_gaussian import (
    signed_log_normal_cdf_difference as _signed_log_normal_cdf_difference,
)
from src.path_integral.stable_gaussian import (
    stable_normal_cdf_difference as _stable_normal_cdf_difference,
)


@dataclass(frozen=True)
class DownsideThresholds:
    """Gaussian thresholds equivalent to a finite-grid downside event."""

    combined: torch.Tensor
    hit: torch.Tensor
    occupation: torch.Tensor
    required_occupation_count: int


@dataclass(frozen=True)
class GaussianShiftDecomposition:
    """Parallel and orthogonal parts of a deterministic Gaussian shift."""

    parallel_coefficient: torch.Tensor
    orthogonal_shift: torch.Tensor


def positive_flat_direction(
    steps: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return the positive unit direction used by the primary MGVS protocol."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("dtype must be floating point")
    return torch.full(
        (steps,),
        1.0 / math.sqrt(steps),
        device=device,
        dtype=dtype,
    )


def positive_exponential_direction(
    steps: int,
    *,
    decay: float,
    device: torch.device | str,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return ``q_i proportional to exp(-decay*(i+1/2)/steps)``.

    Positive decay emphasizes early increments; negative decay emphasizes late
    increments.  The direction is deterministic and strictly positive.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not math.isfinite(decay):
        raise ValueError("decay must be finite")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("dtype must be floating point")
    midpoint = (torch.arange(steps, device=device, dtype=dtype) + 0.5) / steps
    log_weight = -float(decay) * midpoint
    log_weight = log_weight - torch.max(log_weight)
    direction = torch.exp(log_weight)
    return direction / torch.linalg.vector_norm(direction)


def validate_positive_unit_direction(
    direction: torch.Tensor,
    *,
    steps: int | None = None,
    tolerance: float = 1e-10,
) -> None:
    """Validate the deterministic direction required for path monotonicity."""
    if direction.ndim != 1 or (steps is not None and direction.shape[0] != steps):
        raise ValueError("direction must be one-dimensional with the declared step count")
    if not direction.is_floating_point() or not torch.isfinite(direction).all():
        raise ValueError("direction must be finite and floating point")
    if bool((direction <= 0.0).any()):
        raise ValueError("all direction entries must be strictly positive")
    norm_error = abs(float(torch.linalg.vector_norm(direction)) - 1.0)
    if norm_error > tolerance:
        raise ValueError(f"direction must have unit norm; error={norm_error:.3e}")


def orthogonal_gaussian_residual(
    standard_normal: torch.Tensor,
    direction: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decompose a standard Gaussian vector into ``q Z + R``."""
    if standard_normal.ndim < 1:
        raise ValueError("standard_normal must have at least one dimension")
    validate_positive_unit_direction(direction, steps=standard_normal.shape[-1])
    if standard_normal.shape[-1] != direction.shape[0]:
        raise ValueError("standard_normal and direction dimensions must match")
    if standard_normal.device != direction.device or standard_normal.dtype != direction.dtype:
        raise ValueError("standard_normal and direction must share device and dtype")
    if not standard_normal.is_floating_point() or not torch.isfinite(standard_normal).all():
        raise ValueError("standard_normal must be finite and floating point")
    coordinate = torch.sum(standard_normal * direction, dim=-1)
    residual = standard_normal - coordinate.unsqueeze(-1) * direction
    return coordinate, residual


def decompose_gaussian_shift(
    shift: torch.Tensor,
    direction: torch.Tensor,
) -> GaussianShiftDecomposition:
    """Decompose a deterministic standard-normal mean-shift coefficient."""
    if shift.ndim != 1:
        raise ValueError("shift must be one-dimensional")
    validate_positive_unit_direction(direction, steps=shift.shape[-1])
    if shift.ndim != 1 or shift.shape != direction.shape:
        raise ValueError("shift and direction must be matching one-dimensional tensors")
    if shift.device != direction.device or shift.dtype != direction.dtype:
        raise ValueError("shift and direction must share device and dtype")
    if not torch.isfinite(shift).all():
        raise ValueError("shift must be finite")
    parallel = torch.dot(shift, direction)
    orthogonal = shift - parallel * direction
    return GaussianShiftDecomposition(parallel, orthogonal)


def downside_excursion_thresholds(
    log_spot_intercept: torch.Tensor,
    log_spot_slope: torch.Tensor,
    *,
    step_dt: float,
    task: DownsideExcursionTask,
    slope_tolerance: float = 1e-14,
) -> DownsideThresholds:
    """Return the exact scalar threshold for the current finite-grid event.

    The inputs include time zero.  Slopes after time zero must be strictly
    positive, while the time-zero slope must be zero.
    """
    if (
        log_spot_intercept.ndim != 2
        or log_spot_intercept.shape != log_spot_slope.shape
        or log_spot_intercept.shape[1] < 2
    ):
        raise ValueError("intercept and slope must have shape (paths, steps + 1)")
    if (
        not log_spot_intercept.is_floating_point()
        or not log_spot_slope.is_floating_point()
        or log_spot_intercept.device != log_spot_slope.device
        or log_spot_intercept.dtype != log_spot_slope.dtype
        or not torch.isfinite(log_spot_intercept).all()
        or not torch.isfinite(log_spot_slope).all()
    ):
        raise ValueError("intercept and slope must be finite matching floating tensors")
    if not math.isfinite(step_dt) or step_dt <= 0.0:
        raise ValueError("step_dt must be finite and positive")
    if float(torch.max(torch.abs(log_spot_slope[:, 0]))) > slope_tolerance:
        raise ValueError("the time-zero slope must be zero")
    monitoring_slope = log_spot_slope[:, 1:]
    if bool((monitoring_slope <= 0.0).any()):
        raise ValueError("all post-initial slopes must be strictly positive")

    monitoring_intercept = log_spot_intercept[:, 1:]
    hit_candidates = (math.log(task.hit_barrier) - monitoring_intercept) / monitoring_slope
    hit_threshold = torch.max(hit_candidates, dim=1).values
    initially_hit = log_spot_intercept[:, 0] <= math.log(task.hit_barrier)
    hit_threshold = torch.where(
        initially_hit,
        torch.full_like(hit_threshold, math.inf),
        hit_threshold,
    )

    steps = monitoring_intercept.shape[1]
    required = math.ceil((task.minimum_occupation - 1e-15) / step_dt)
    if required <= 0:
        occupation_threshold = torch.full_like(hit_threshold, math.inf)
    elif required > steps:
        occupation_threshold = torch.full_like(hit_threshold, -math.inf)
    else:
        occupation_candidates = (
            math.log(task.stress_level) - monitoring_intercept
        ) / monitoring_slope
        occupation_threshold = torch.topk(
            occupation_candidates,
            k=required,
            dim=1,
            largest=True,
            sorted=True,
        ).values[:, -1]
    return DownsideThresholds(
        combined=torch.minimum(hit_threshold, occupation_threshold),
        hit=hit_threshold,
        occupation=occupation_threshold,
        required_occupation_count=required,
    )


def signed_log_normal_cdf_difference(
    left: torch.Tensor,
    right: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compatibility wrapper for the audited G11 stable implementation."""
    return _signed_log_normal_cdf_difference(left, right)


def stable_normal_cdf_difference(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Compatibility wrapper for the audited G11 stable implementation."""
    return _stable_normal_cdf_difference(left, right)


def scaled_normal_cdf_difference(
    log_scale: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Compatibility wrapper for the audited G11 stable implementation."""
    return _scaled_normal_cdf_difference(log_scale, left, right)


def scaled_normal_cdf(
    log_scale: torch.Tensor,
    argument: torch.Tensor,
) -> torch.Tensor:
    """Compatibility wrapper for the audited G11 stable implementation."""
    return _scaled_normal_cdf(log_scale, argument)
