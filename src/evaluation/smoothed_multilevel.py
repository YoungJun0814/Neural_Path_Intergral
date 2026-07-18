"""Paired statistical diagnostics for raw and Gaussian-smoothed MLMC levels."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PairedLevelDiagnostics:
    """A paired raw-versus-smoothed finite-grid level comparison."""

    paths: int
    raw_mean: float
    smoothed_mean: float
    paired_mean_difference: float
    paired_mean_difference_z: float
    raw_variance: float
    smoothed_variance: float
    smoothed_over_raw_variance: float
    variance_ratio_ci_lower: float
    variance_ratio_ci_upper: float
    confidence_level: float
    raw_excess_kurtosis: float
    smoothed_excess_kurtosis: float
    raw_cost_per_path: float
    smoothed_cost_per_path: float
    raw_work_coefficient: float
    smoothed_work_coefficient: float
    raw_over_smoothed_work_ratio: float


@dataclass(frozen=True)
class PairedMLMCDiagnostics:
    """Independent-level MLMC work and estimator summaries."""

    levels: int
    raw_estimate: float
    smoothed_estimate: float
    raw_standard_error: float
    smoothed_standard_error: float
    raw_work_coefficient: float
    smoothed_work_coefficient: float
    raw_over_smoothed_work_ratio: float


def _excess_kurtosis(values: torch.Tensor) -> float:
    centered = values - values.mean()
    second = torch.mean(centered.square())
    if float(second) == 0.0:
        return 0.0
    return float(torch.mean(centered.pow(4)) / second.square() - 3.0)


def _paired_bootstrap_variance_ratio_interval(
    raw: torch.Tensor,
    smoothed: torch.Tensor,
    *,
    confidence_level: float,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    bootstrap_batch_size: int = 64,
) -> tuple[float, float]:
    if bootstrap_replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    generator = torch.Generator(device="cpu").manual_seed(bootstrap_seed)
    raw_cpu = raw.detach().to(device="cpu", dtype=torch.float64)
    smoothed_cpu = smoothed.detach().to(device="cpu", dtype=torch.float64)
    paths = raw_cpu.numel()
    ratios: list[torch.Tensor] = []
    completed = 0
    while completed < bootstrap_replicates:
        batch = min(bootstrap_batch_size, bootstrap_replicates - completed)
        indices = torch.randint(paths, (batch, paths), generator=generator)
        raw_variance = torch.var(raw_cpu[indices], dim=1, unbiased=True)
        smoothed_variance = torch.var(smoothed_cpu[indices], dim=1, unbiased=True)
        if bool((raw_variance <= 0.0).any()):
            raise ValueError("bootstrap raw variance became zero")
        ratios.append(smoothed_variance / raw_variance)
        completed += batch
    ratio = torch.cat(ratios)
    tail = (1.0 - confidence_level) / 2.0
    return (
        float(torch.quantile(ratio, tail)),
        float(torch.quantile(ratio, 1.0 - tail)),
    )


def paired_level_diagnostics(
    raw: torch.Tensor,
    smoothed: torch.Tensor,
    *,
    raw_cost_per_path: float,
    smoothed_cost_per_path: float,
    confidence_level: float = 0.95,
    bootstrap_replicates: int = 1_000,
    bootstrap_seed: int = 0,
) -> PairedLevelDiagnostics:
    """Summarize paired estimators without treating them as independent."""
    if raw.ndim != 1 or smoothed.shape != raw.shape or raw.numel() < 3:
        raise ValueError("raw and smoothed must be matching vectors with at least three values")
    if (
        not raw.is_floating_point()
        or raw.device != smoothed.device
        or raw.dtype != smoothed.dtype
        or not torch.isfinite(raw).all()
        or not torch.isfinite(smoothed).all()
    ):
        raise ValueError("raw and smoothed must be finite matching floating tensors")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    if not math.isfinite(raw_cost_per_path) or raw_cost_per_path <= 0.0:
        raise ValueError("raw_cost_per_path must be finite and positive")
    if not math.isfinite(smoothed_cost_per_path) or smoothed_cost_per_path <= 0.0:
        raise ValueError("smoothed_cost_per_path must be finite and positive")

    raw_variance = float(raw.var(unbiased=True))
    smoothed_variance = float(smoothed.var(unbiased=True))
    if raw_variance <= 0.0:
        raise ValueError("raw estimator variance must be positive")
    difference = smoothed - raw
    difference_standard_error = math.sqrt(float(difference.var(unbiased=True)) / raw.numel())
    difference_mean = float(difference.mean())
    difference_z = (
        difference_mean / difference_standard_error
        if difference_standard_error > 0.0
        else (0.0 if difference_mean == 0.0 else math.copysign(math.inf, difference_mean))
    )
    interval = _paired_bootstrap_variance_ratio_interval(
        raw,
        smoothed,
        confidence_level=confidence_level,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    raw_work = raw_variance * raw_cost_per_path
    smoothed_work = smoothed_variance * smoothed_cost_per_path
    ratio = math.inf if smoothed_work == 0.0 else raw_work / smoothed_work
    return PairedLevelDiagnostics(
        paths=raw.numel(),
        raw_mean=float(raw.mean()),
        smoothed_mean=float(smoothed.mean()),
        paired_mean_difference=difference_mean,
        paired_mean_difference_z=difference_z,
        raw_variance=raw_variance,
        smoothed_variance=smoothed_variance,
        smoothed_over_raw_variance=smoothed_variance / raw_variance,
        variance_ratio_ci_lower=interval[0],
        variance_ratio_ci_upper=interval[1],
        confidence_level=confidence_level,
        raw_excess_kurtosis=_excess_kurtosis(raw),
        smoothed_excess_kurtosis=_excess_kurtosis(smoothed),
        raw_cost_per_path=raw_cost_per_path,
        smoothed_cost_per_path=smoothed_cost_per_path,
        raw_work_coefficient=raw_work,
        smoothed_work_coefficient=smoothed_work,
        raw_over_smoothed_work_ratio=ratio,
    )


def paired_mlmc_diagnostics(
    reports: Sequence[PairedLevelDiagnostics],
) -> PairedMLMCDiagnostics:
    """Combine independent-level pilot moments using optimal-work coefficients."""
    if not reports:
        raise ValueError("at least one level report is required")
    paths = [report.paths for report in reports]
    raw_variance_of_mean = sum(
        report.raw_variance / paths[index] for index, report in enumerate(reports)
    )
    smoothed_variance_of_mean = sum(
        report.smoothed_variance / paths[index] for index, report in enumerate(reports)
    )
    raw_coefficient = (
        sum(math.sqrt(report.raw_variance * report.raw_cost_per_path) for report in reports) ** 2
    )
    smoothed_coefficient = (
        sum(
            math.sqrt(report.smoothed_variance * report.smoothed_cost_per_path)
            for report in reports
        )
        ** 2
    )
    ratio = math.inf if smoothed_coefficient == 0.0 else raw_coefficient / smoothed_coefficient
    return PairedMLMCDiagnostics(
        levels=len(reports),
        raw_estimate=sum(report.raw_mean for report in reports),
        smoothed_estimate=sum(report.smoothed_mean for report in reports),
        raw_standard_error=math.sqrt(raw_variance_of_mean),
        smoothed_standard_error=math.sqrt(smoothed_variance_of_mean),
        raw_work_coefficient=raw_coefficient,
        smoothed_work_coefficient=smoothed_coefficient,
        raw_over_smoothed_work_ratio=ratio,
    )
