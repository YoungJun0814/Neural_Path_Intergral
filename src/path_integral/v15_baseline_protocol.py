"""Frozen comparator and work-accounting contract for V15 experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from src.path_integral.baseline_framework import BaselineCostLedger

REQUIRED_PRIMARY_COMPARATORS = frozenset(
    {
        "conditional_rbergomi",
        "v14_local_volterra",
        "defensive_cem",
        "ld_subspace_is",
        "smoothing_rqmc",
    }
)


@dataclass(frozen=True)
class V15BaselineProtocol:
    primary_comparators: tuple[str, ...]
    query_counts: tuple[int, ...] = (1, 10, 100, 1_000)
    ordinary_is_only: bool = True
    include_training_cost: bool = True
    paired_problem_cells: bool = True

    def __post_init__(self) -> None:
        missing = REQUIRED_PRIMARY_COMPARATORS - set(self.primary_comparators)
        if missing:
            raise ValueError(f"V15 protocol is missing primary comparators: {sorted(missing)}")
        if len(set(self.primary_comparators)) != len(self.primary_comparators):
            raise ValueError("V15 comparator names must be unique")
        if not self.query_counts or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in self.query_counts
        ):
            raise ValueError("V15 query counts must be positive integers")
        if tuple(sorted(set(self.query_counts))) != self.query_counts:
            raise ValueError("V15 query counts must be strictly increasing")
        if not (self.ordinary_is_only and self.include_training_cost and self.paired_problem_cells):
            raise ValueError("V15 cannot disable exactness, training cost, or paired cells")


@dataclass(frozen=True)
class V15MethodSummary:
    method: str
    estimate: float
    sample_variance: float
    standard_error: float
    samples: int
    training_work: float
    evaluation_work_per_sample: float

    def total_work(self, query_count: int) -> float:
        if query_count < 1:
            raise ValueError("query count must be positive")
        return self.training_work + query_count * self.samples * self.evaluation_work_per_sample

    def work_normalized_variance(self, query_count: int) -> float:
        return self.sample_variance * self.total_work(query_count) / self.samples


def summarize_v15_method(
    method: str,
    contributions: torch.Tensor,
    *,
    training_cost: BaselineCostLedger,
    evaluation_cost: BaselineCostLedger,
) -> V15MethodSummary:
    if contributions.ndim != 1 or contributions.numel() < 2:
        raise ValueError("V15 contributions must contain at least two inferential units")
    if not torch.isfinite(contributions).all():
        raise ValueError("V15 contributions must be finite")
    samples = int(contributions.numel())
    variance = float(torch.var(contributions, unbiased=True))
    evaluation_work = evaluation_cost.algorithmic_work_units / samples
    if not math.isfinite(evaluation_work) or evaluation_work <= 0.0:
        raise ValueError("V15 evaluation work per sample must be finite and positive")
    return V15MethodSummary(
        method=method,
        estimate=float(torch.mean(contributions)),
        sample_variance=variance,
        standard_error=math.sqrt(variance / samples),
        samples=samples,
        training_work=training_cost.algorithmic_work_units,
        evaluation_work_per_sample=evaluation_work,
    )


def work_efficiency_ratio(
    baseline: V15MethodSummary,
    candidate: V15MethodSummary,
    *,
    query_count: int,
) -> float:
    """Return baseline/candidate work-normalized variance; values above one favor V15."""

    candidate_value = candidate.work_normalized_variance(query_count)
    if candidate_value <= 0.0:
        return math.inf
    return baseline.work_normalized_variance(query_count) / candidate_value
