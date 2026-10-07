"""Pilot-frozen MLMC allocation and error-budget utilities for V15 mesh studies."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MLMCLevelPilot:
    level: int
    mean: float
    variance: float
    work_per_sample: float

    def __post_init__(self) -> None:
        if self.level < 0 or not math.isfinite(self.mean):
            raise ValueError("invalid MLMC pilot level")
        if not math.isfinite(self.variance) or self.variance < 0.0:
            raise ValueError("MLMC pilot variance must be finite and nonnegative")
        if not math.isfinite(self.work_per_sample) or self.work_per_sample <= 0.0:
            raise ValueError("MLMC pilot work must be finite and positive")


@dataclass(frozen=True)
class MLMCPilotAllocation:
    target_rmse: float
    sampling_variance_budget: float
    sample_counts: tuple[int, ...]
    predicted_sampling_variance: float
    predicted_work: float
    bias_proxy: float
    bias_budget: float
    bias_gate_pass: bool


def allocate_pilot_frozen_mlmc(
    pilots: tuple[MLMCLevelPilot, ...],
    *,
    target_rmse: float,
    sampling_fraction: float = 0.5,
    bias_multiplier: float = 1.0,
) -> MLMCPilotAllocation:
    """Freeze the standard variance-cost allocation before final evaluation."""

    if not pilots or tuple(item.level for item in pilots) != tuple(range(len(pilots))):
        raise ValueError("MLMC pilot levels must be consecutive from zero")
    if not math.isfinite(target_rmse) or target_rmse <= 0.0:
        raise ValueError("target RMSE must be finite and positive")
    if not 0.0 < sampling_fraction < 1.0:
        raise ValueError("sampling fraction must lie in (0, 1)")
    if not math.isfinite(bias_multiplier) or bias_multiplier <= 0.0:
        raise ValueError("bias multiplier must be finite and positive")
    variance_budget = sampling_fraction * target_rmse**2
    square_root_products = [
        math.sqrt(item.variance * item.work_per_sample) for item in pilots
    ]
    normalization = sum(square_root_products)
    counts = []
    for item in pilots:
        if item.variance == 0.0:
            counts.append(2)
        else:
            ideal = normalization * math.sqrt(item.variance / item.work_per_sample) / variance_budget
            counts.append(max(2, math.ceil(ideal)))
    predicted_variance = sum(
        item.variance / count for item, count in zip(pilots, counts, strict=True)
    )
    predicted_work = sum(
        item.work_per_sample * count for item, count in zip(pilots, counts, strict=True)
    )
    bias_proxy = bias_multiplier * abs(pilots[-1].mean)
    bias_budget = math.sqrt(1.0 - sampling_fraction) * target_rmse
    return MLMCPilotAllocation(
        target_rmse=target_rmse,
        sampling_variance_budget=variance_budget,
        sample_counts=tuple(counts),
        predicted_sampling_variance=predicted_variance,
        predicted_work=predicted_work,
        bias_proxy=bias_proxy,
        bias_budget=bias_budget,
        bias_gate_pass=bias_proxy <= bias_budget,
    )
