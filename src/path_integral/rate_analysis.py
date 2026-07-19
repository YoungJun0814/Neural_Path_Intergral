"""Seed-clustered diagnostics and predeclared window selection for G11 rates."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class CorrectionRateObservation:
    """Path-aggregated diagnostics from one independent seed cluster and level."""

    level: int
    replicate: int
    paths: int
    threshold_l1: float
    threshold_l2: float
    raw_second_moment: float
    dcs_second_moment: float
    raw_variance: float
    dcs_variance: float
    raw_kurtosis: float
    dcs_kurtosis: float
    raw_zero_fraction: float
    dcs_zero_fraction: float
    raw_positive_fraction: float
    raw_negative_fraction: float
    dcs_positive_fraction: float
    dcs_negative_fraction: float
    raw_work_units: float
    dcs_work_units: float

    def __post_init__(self) -> None:
        if self.level < 1 or self.replicate < 0 or self.paths < 2:
            raise ValueError("invalid correction observation index or path count")
        for name, value in asdict(self).items():
            if name in {"level", "replicate", "paths"}:
                continue
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")


def _variance(values: torch.Tensor) -> float:
    return float(torch.var(values, unbiased=True))


def _kurtosis(values: torch.Tensor) -> float:
    centered = values - torch.mean(values)
    variance = torch.mean(centered * centered)
    if float(variance) == 0.0:
        return 0.0
    return float(torch.mean(centered**4) / (variance * variance))


def correction_rate_observation(
    *,
    level: int,
    replicate: int,
    threshold_difference: torch.Tensor,
    raw_correction: torch.Tensor,
    dcs_correction: torch.Tensor,
    raw_work_units: float,
    dcs_work_units: float,
    zero_tolerance: float = 0.0,
) -> CorrectionRateObservation:
    """Construct one validated cluster record without treating paths as replicates."""

    tensors = (threshold_difference, raw_correction, dcs_correction)
    if (
        any(item.ndim != 1 for item in tensors)
        or len({item.shape for item in tensors}) != 1
        or threshold_difference.numel() < 2
    ):
        raise ValueError("diagnostic tensors must be matching vectors")
    if any(
        not item.is_floating_point() or not torch.isfinite(item).all()
        for item in tensors
    ):
        raise ValueError("diagnostic tensors must be finite and floating")
    if not math.isfinite(zero_tolerance) or zero_tolerance < 0.0:
        raise ValueError("zero_tolerance must be finite and nonnegative")

    def fractions(values: torch.Tensor) -> tuple[float, float, float]:
        zero = torch.abs(values) <= zero_tolerance
        return (
            float(torch.mean(zero.to(torch.float64))),
            float(torch.mean((values > zero_tolerance).to(torch.float64))),
            float(torch.mean((values < -zero_tolerance).to(torch.float64))),
        )

    raw_zero, raw_positive, raw_negative = fractions(raw_correction)
    dcs_zero, dcs_positive, dcs_negative = fractions(dcs_correction)
    return CorrectionRateObservation(
        level=level,
        replicate=replicate,
        paths=int(threshold_difference.numel()),
        threshold_l1=float(torch.mean(torch.abs(threshold_difference))),
        threshold_l2=float(torch.mean(threshold_difference**2)),
        raw_second_moment=float(torch.mean(raw_correction**2)),
        dcs_second_moment=float(torch.mean(dcs_correction**2)),
        raw_variance=_variance(raw_correction),
        dcs_variance=_variance(dcs_correction),
        raw_kurtosis=_kurtosis(raw_correction),
        dcs_kurtosis=_kurtosis(dcs_correction),
        raw_zero_fraction=raw_zero,
        dcs_zero_fraction=dcs_zero,
        raw_positive_fraction=raw_positive,
        raw_negative_fraction=raw_negative,
        dcs_positive_fraction=dcs_positive,
        dcs_negative_fraction=dcs_negative,
        raw_work_units=raw_work_units,
        dcs_work_units=dcs_work_units,
    )


_RATE_METRICS = (
    "threshold_l1",
    "threshold_l2",
    "raw_second_moment",
    "dcs_second_moment",
    "raw_variance",
    "dcs_variance",
)


@dataclass(frozen=True)
class RateWindowAnalysis:
    identified: bool
    reason: str
    levels: tuple[int, ...]
    exponents: Mapping[str, float]
    confidence_intervals_95: Mapping[str, tuple[float, float]]
    level_variance_cv: Mapping[str, float]
    bootstrap_repetitions: int


def _aggregate(
    observations: Iterable[CorrectionRateObservation],
) -> dict[int, dict[str, float]]:
    grouped: dict[int, list[CorrectionRateObservation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.level].append(observation)
    result: dict[int, dict[str, float]] = {}
    for level, items in grouped.items():
        total_paths = sum(item.paths for item in items)
        result[level] = {
            metric: math.fsum(getattr(item, metric) * item.paths for item in items)
            / total_paths
            for metric in _RATE_METRICS
        }
    return result


def _exponent(levels: tuple[int, ...], values: Mapping[int, float]) -> float:
    y = torch.tensor(
        [math.log2(values[level]) for level in levels], dtype=torch.float64
    )
    x = torch.tensor(levels, dtype=torch.float64)
    centered = x - torch.mean(x)
    slope = torch.sum(centered * (y - torch.mean(y))) / torch.sum(centered**2)
    return -float(slope)


def _candidate_stable(
    levels: tuple[int, ...], aggregate: Mapping[int, Mapping[str, float]], margin: float
) -> bool:
    for metric in ("raw_variance", "dcs_variance"):
        values = {level: aggregate[level][metric] for level in levels}
        if any(value <= 0.0 for value in values.values()):
            return False
        full = _exponent(levels, values)
        if len(levels) > 3:
            left = _exponent(levels[1:], values)
            right = _exponent(levels[:-1], values)
            if max(abs(full - left), abs(full - right)) > margin:
                return False
    return True


def _cluster_bootstrap_aggregates(
    observations: tuple[CorrectionRateObservation, ...],
    repetitions: int,
    seed: int,
) -> list[dict[int, dict[str, float]]]:
    replicates = sorted({item.replicate for item in observations})
    by_replicate: dict[int, tuple[CorrectionRateObservation, ...]] = {
        replicate: tuple(item for item in observations if item.replicate == replicate)
        for replicate in replicates
    }
    generator = torch.Generator().manual_seed(seed)
    results: list[dict[int, dict[str, float]]] = []
    for _ in range(repetitions):
        indices = torch.randint(
            len(replicates), (len(replicates),), generator=generator
        ).tolist()
        draw: list[CorrectionRateObservation] = []
        for index in indices:
            draw.extend(by_replicate[replicates[index]])
        results.append(_aggregate(draw))
    return results


def identify_rate_window(
    observations: Iterable[CorrectionRateObservation],
    *,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
    minimum_levels: int = 4,
    endpoint_slope_margin: float = 0.15,
    maximum_variance_cv: float = 0.20,
) -> RateWindowAnalysis:
    """Apply one common, predeclared window rule to raw and DCS corrections."""

    data = tuple(observations)
    if bootstrap_repetitions < 200:
        raise ValueError("at least 200 bootstrap repetitions are required")
    if len({item.replicate for item in data}) < 3:
        raise ValueError("at least three independent seed clusters are required")
    aggregate = _aggregate(data)
    levels = tuple(sorted(aggregate))
    if len(levels) < minimum_levels or any(
        right != left + 1 for left, right in zip(levels, levels[1:], strict=False)
    ):
        return RateWindowAnalysis(
            False,
            "rate unidentified: insufficient consecutive levels",
            (),
            {},
            {},
            {},
            bootstrap_repetitions,
        )
    bootstraps = _cluster_bootstrap_aggregates(
        data, bootstrap_repetitions, bootstrap_seed
    )
    level_cv: dict[str, float] = {}
    for metric in ("raw_variance", "dcs_variance"):
        for level in levels:
            draws = torch.tensor(
                [item[level][metric] for item in bootstraps], dtype=torch.float64
            )
            mean = float(torch.mean(draws))
            level_cv[f"{metric}:L{level}"] = (
                float(torch.std(draws, unbiased=True)) / mean if mean > 0.0 else math.inf
            )

    candidates: list[tuple[int, ...]] = []
    for length in range(len(levels), minimum_levels - 1, -1):
        for start in range(0, len(levels) - length + 1):
            candidate = levels[start : start + length]
            if not _candidate_stable(candidate, aggregate, endpoint_slope_margin):
                continue
            if any(
                level_cv[f"{metric}:L{level}"] > maximum_variance_cv
                for metric in ("raw_variance", "dcs_variance")
                for level in candidate
            ):
                continue
            candidates.append(candidate)
        if candidates:
            break
    if not candidates:
        return RateWindowAnalysis(
            False,
            "rate unidentified: no common stable raw/DCS window",
            (),
            {},
            {},
            level_cv,
            bootstrap_repetitions,
        )
    selected = candidates[0]
    exponents = {
        metric: _exponent(
            selected, {level: aggregate[level][metric] for level in selected}
        )
        for metric in _RATE_METRICS
    }
    draws_by_metric: dict[str, list[float]] = {metric: [] for metric in _RATE_METRICS}
    for item in bootstraps:
        for metric in _RATE_METRICS:
            values = {level: item[level][metric] for level in selected}
            if all(value > 0.0 for value in values.values()):
                draws_by_metric[metric].append(_exponent(selected, values))
    intervals: dict[str, tuple[float, float]] = {}
    for metric, draws in draws_by_metric.items():
        tensor = torch.tensor(draws, dtype=torch.float64)
        intervals[metric] = (
            float(torch.quantile(tensor, 0.025)),
            float(torch.quantile(tensor, 0.975)),
        )
    derived_draws = {
        "dcs_second_minus_threshold_l2": [
            dcs - threshold
            for dcs, threshold in zip(
                draws_by_metric["dcs_second_moment"],
                draws_by_metric["threshold_l2"],
                strict=True,
            )
        ],
        "dcs_second_minus_raw_second": [
            dcs - raw
            for dcs, raw in zip(
                draws_by_metric["dcs_second_moment"],
                draws_by_metric["raw_second_moment"],
                strict=True,
            )
        ],
    }
    exponents["dcs_second_minus_threshold_l2"] = (
        exponents["dcs_second_moment"] - exponents["threshold_l2"]
    )
    exponents["dcs_second_minus_raw_second"] = (
        exponents["dcs_second_moment"] - exponents["raw_second_moment"]
    )
    for metric, draws in derived_draws.items():
        tensor = torch.tensor(draws, dtype=torch.float64)
        intervals[metric] = (
            float(torch.quantile(tensor, 0.025)),
            float(torch.quantile(tensor, 0.975)),
        )
    return RateWindowAnalysis(
        True,
        "common predeclared stability rule passed",
        selected,
        exponents,
        intervals,
        level_cv,
        bootstrap_repetitions,
    )
