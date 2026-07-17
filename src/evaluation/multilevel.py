"""Allocation and work accounting for independent multilevel corrections."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class MLMCAllocation:
    sample_counts: tuple[int, ...]
    predicted_variance: float
    predicted_online_work: float
    continuous_optimal_work: float


def optimal_mlmc_sample_counts(
    variances: Sequence[float],
    costs: Sequence[float],
    *,
    variance_budget: float,
    minimum_samples: int = 2,
) -> MLMCAllocation:
    r"""Return the variance-constrained square-root MLMC allocation.

    The continuous solution is

    ``N_l = eps^-2 sqrt(V_l/C_l) sum_j sqrt(V_j C_j)``.

    Counts are rounded upward, so the returned allocation remains within the
    requested Monte Carlo variance budget when the supplied pilot moments are
    treated as exact.
    """
    if len(variances) == 0 or len(variances) != len(costs):
        raise ValueError("variances and costs must be nonempty and equally sized")
    if not math.isfinite(variance_budget) or variance_budget <= 0.0:
        raise ValueError("variance_budget must be finite and positive")
    if minimum_samples <= 0:
        raise ValueError("minimum_samples must be positive")
    resolved_variances = tuple(float(value) for value in variances)
    resolved_costs = tuple(float(value) for value in costs)
    if not all(math.isfinite(value) and value >= 0.0 for value in resolved_variances):
        raise ValueError("variances must be finite and nonnegative")
    if not all(math.isfinite(value) and value > 0.0 for value in resolved_costs):
        raise ValueError("costs must be finite and positive")

    normalizer = sum(
        math.sqrt(variance * cost)
        for variance, cost in zip(resolved_variances, resolved_costs, strict=True)
    )
    counts: list[int] = []
    for variance, cost in zip(resolved_variances, resolved_costs, strict=True):
        if variance == 0.0 or normalizer == 0.0:
            counts.append(minimum_samples)
        else:
            continuous = normalizer / variance_budget * math.sqrt(variance / cost)
            counts.append(max(minimum_samples, math.ceil(continuous)))
    predicted_variance = sum(
        variance / count
        for variance, count in zip(resolved_variances, counts, strict=True)
    )
    predicted_work = sum(
        count * cost for count, cost in zip(counts, resolved_costs, strict=True)
    )
    continuous_work = normalizer * normalizer / variance_budget
    return MLMCAllocation(
        sample_counts=tuple(counts),
        predicted_variance=predicted_variance,
        predicted_online_work=predicted_work,
        continuous_optimal_work=continuous_work,
    )


def single_level_online_work(
    variance: float,
    cost_per_path: float,
    *,
    variance_budget: float,
    minimum_samples: int = 2,
) -> tuple[int, float]:
    """Return the rounded finest-level sample count and online work."""
    if not math.isfinite(variance) or variance < 0.0:
        raise ValueError("variance must be finite and nonnegative")
    if not math.isfinite(cost_per_path) or cost_per_path <= 0.0:
        raise ValueError("cost_per_path must be finite and positive")
    if not math.isfinite(variance_budget) or variance_budget <= 0.0:
        raise ValueError("variance_budget must be finite and positive")
    if minimum_samples <= 0:
        raise ValueError("minimum_samples must be positive")
    count = max(minimum_samples, math.ceil(variance / variance_budget))
    return count, count * cost_per_path


def break_even_query_count(
    training_work: float,
    baseline_online_work: float,
    candidate_online_work: float,
) -> float:
    """Return amortization queries, or infinity when no online saving exists."""
    values = (training_work, baseline_online_work, candidate_online_work)
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("work values must be finite and nonnegative")
    saving = baseline_online_work - candidate_online_work
    if saving <= 0.0:
        return math.inf
    return training_work / saving
