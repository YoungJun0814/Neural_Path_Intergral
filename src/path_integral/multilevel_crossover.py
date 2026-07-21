"""Finite-level work crossover diagnostics for multilevel estimators.

The comparison is deliberately non-asymptotic.  Given independently estimated
single-level and correction variances and per-sample costs, it compares the exact
optimal-allocation work coefficients at every admissible starting level.  It is an
implementation of the standard MLMC allocation identity, used here to prevent an
MLMC method from being declared efficient without comparison to its single-level
counterpart.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class MultilevelCrossoverDecision:
    """Work coefficients and the cheapest admissible estimator construction."""

    finest_level: int
    single_level_work_coefficient: float
    multilevel_work_coefficients: tuple[float, ...]
    optimal_start_level: int
    optimal_work_coefficient: float
    multilevel_strictly_better: bool
    single_over_optimal_work_ratio: float


@dataclass(frozen=True)
class TotalWorkCrossoverDecision:
    """Training-inclusive finite-RMSE crossover decision."""

    sampling_variance_target: float
    online: MultilevelCrossoverDecision
    preprocessing_work_by_start_level: tuple[float, ...]
    total_work_by_start_level: tuple[float, ...]
    optimal_start_level: int
    optimal_total_work: float
    multilevel_strictly_better: bool
    single_over_optimal_total_work_ratio: float


def _validate_positive(name: str, values: Sequence[float]) -> tuple[float, ...]:
    resolved = tuple(float(value) for value in values)
    if not resolved or any(not math.isfinite(value) or value <= 0.0 for value in resolved):
        raise ValueError(f"{name} must contain finite positive values")
    return resolved


def optimal_sampling_work_coefficient(variances: Sequence[float], costs: Sequence[float]) -> float:
    """Return ``(sum_l sqrt(V_l C_l))^2`` for optimal independent allocation."""

    resolved_variances = _validate_positive("variances", variances)
    resolved_costs = _validate_positive("costs", costs)
    if len(resolved_variances) != len(resolved_costs):
        raise ValueError("variances and costs must have the same length")
    return (
        math.fsum(
            math.sqrt(variance * cost)
            for variance, cost in zip(resolved_variances, resolved_costs, strict=True)
        )
        ** 2
    )


def evaluate_multilevel_crossover(
    *,
    single_level_variances: Sequence[float],
    single_level_costs: Sequence[float],
    correction_variances: Sequence[float],
    correction_costs: Sequence[float],
) -> MultilevelCrossoverDecision:
    """Compare finest-grid single-level work with every MLMC starting level.

    ``single_level_*[l]`` describes the level-``l`` estimator.  The correction
    arrays describe levels ``1,...,L`` and therefore have length ``L`` when the
    single-level arrays have length ``L+1``.  Starting at ``l0`` uses the
    single-level term at ``l0`` and corrections ``l0+1,...,L``.
    """

    single_variances = _validate_positive("single_level_variances", single_level_variances)
    single_costs = _validate_positive("single_level_costs", single_level_costs)
    correction_variance = _validate_positive("correction_variances", correction_variances)
    correction_cost = _validate_positive("correction_costs", correction_costs)
    if len(single_variances) != len(single_costs):
        raise ValueError("single-level variance and cost lengths differ")
    if len(correction_variance) != len(correction_cost):
        raise ValueError("correction variance and cost lengths differ")
    if len(correction_variance) + 1 != len(single_variances):
        raise ValueError("corrections must cover levels one through the finest level")

    finest_level = len(single_variances) - 1
    single_coefficient = single_variances[-1] * single_costs[-1]
    candidates: list[float] = []
    for start in range(finest_level + 1):
        if start == finest_level:
            candidates.append(single_coefficient)
        else:
            variances = (single_variances[start],) + correction_variance[start:]
            costs = (single_costs[start],) + correction_cost[start:]
            candidates.append(optimal_sampling_work_coefficient(variances, costs))
    optimal_start = min(range(len(candidates)), key=lambda index: (candidates[index], index))
    optimal = candidates[optimal_start]
    return MultilevelCrossoverDecision(
        finest_level=finest_level,
        single_level_work_coefficient=single_coefficient,
        multilevel_work_coefficients=tuple(candidates),
        optimal_start_level=optimal_start,
        optimal_work_coefficient=optimal,
        multilevel_strictly_better=optimal_start < finest_level and optimal < single_coefficient,
        single_over_optimal_work_ratio=single_coefficient / optimal,
    )


def evaluate_total_work_crossover(
    *,
    single_level_variances: Sequence[float],
    single_level_costs: Sequence[float],
    correction_variances: Sequence[float],
    correction_costs: Sequence[float],
    preprocessing_work_by_start_level: Sequence[float],
    sampling_variance_target: float,
) -> TotalWorkCrossoverDecision:
    """Add declared preprocessing work to every finite-level candidate.

    The online term uses the continuous optimal-allocation coefficient divided by the
    requested sampling variance.  The preprocessing vector has one entry for each
    start level, including the finest-level single estimator.  It must include every
    method-specific training, tuning, pilot, and setup cost that is not already part
    of the per-sample costs.
    """

    if not math.isfinite(sampling_variance_target) or sampling_variance_target <= 0.0:
        raise ValueError("sampling_variance_target must be finite and positive")
    online = evaluate_multilevel_crossover(
        single_level_variances=single_level_variances,
        single_level_costs=single_level_costs,
        correction_variances=correction_variances,
        correction_costs=correction_costs,
    )
    preprocessing = tuple(float(value) for value in preprocessing_work_by_start_level)
    if len(preprocessing) != len(online.multilevel_work_coefficients):
        raise ValueError("preprocessing work must have one entry per start level")
    if any(not math.isfinite(value) or value < 0.0 for value in preprocessing):
        raise ValueError("preprocessing work must be finite and nonnegative")
    totals = tuple(
        fixed + coefficient / sampling_variance_target
        for fixed, coefficient in zip(
            preprocessing, online.multilevel_work_coefficients, strict=True
        )
    )
    optimal_start = min(range(len(totals)), key=lambda index: (totals[index], index))
    optimal = totals[optimal_start]
    single = totals[online.finest_level]
    return TotalWorkCrossoverDecision(
        sampling_variance_target=sampling_variance_target,
        online=online,
        preprocessing_work_by_start_level=preprocessing,
        total_work_by_start_level=totals,
        optimal_start_level=optimal_start,
        optimal_total_work=optimal,
        multilevel_strictly_better=optimal_start < online.finest_level and optimal < single,
        single_over_optimal_total_work_ratio=single / optimal,
    )
