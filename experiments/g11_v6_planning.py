"""Shared replicated point-variance planning for V6 direct estimators.

The routines in this module deliberately separate three statements:

1. the final estimator is exact on the frozen grid because its streams are
   independent of every planning stream;
2. the replicated pilot statistic is an engineering allocation rule, not a
   finite-sample upper confidence bound for the variance;
3. a deterministic bounded-importance variance bound may be recorded as a
   diagnostic certificate, but is not used to inflate the final allocation.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from typing import Any, Literal

import torch

from src.path_integral import SeedKey, SeedLedger

PlanningVarianceStatistic = Literal[
    "mean_replicate_variance", "median_replicate_variance"
]


@dataclass(frozen=True)
class ReplicatedDirectPilot:
    """Independent replicate summaries plus the pooled observations."""

    values: torch.Tensor
    replicate_variances: tuple[float, ...]
    planning_variance: float
    replicates: int
    samples_per_replicate: int
    work_units: float
    wall_seconds: float
    cpu_seconds: float


def validate_replicated_planning(
    *,
    replicates: Any,
    samples_per_replicate: Any,
    variance_statistic: Any,
) -> tuple[int, int, PlanningVarianceStatistic]:
    """Validate and normalize a replicated direct-planning declaration."""

    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or replicates < 3
    ):
        raise ValueError("replicated direct planning requires at least three replicates")
    if (
        isinstance(samples_per_replicate, bool)
        or not isinstance(samples_per_replicate, int)
        or samples_per_replicate < 2
    ):
        raise ValueError(
            "replicated direct planning requires at least two samples per replicate"
        )
    if variance_statistic not in {
        "mean_replicate_variance",
        "median_replicate_variance",
    }:
        raise ValueError("unsupported replicated direct-planning variance statistic")
    return replicates, samples_per_replicate, variance_statistic


def replicated_variance_statistic(
    replicate_variances: tuple[float, ...],
    statistic: PlanningVarianceStatistic,
) -> float:
    """Return the frozen point statistic used by the allocation design."""

    if len(replicate_variances) < 3 or any(
        not math.isfinite(value) or value < 0.0 for value in replicate_variances
    ):
        raise ValueError("replicate variances must contain at least three finite values")
    if statistic == "mean_replicate_variance":
        return float(statistics.fmean(replicate_variances))
    if statistic == "median_replicate_variance":
        return float(statistics.median(replicate_variances))
    raise ValueError("unsupported replicated direct-planning variance statistic")


def collect_replicated_direct_pilot(
    *,
    sampler: Any,
    ledger: SeedLedger,
    protocol: str,
    cell_id: str,
    cluster: int,
    method: str,
    profile_id: str,
    level: int,
    streams: tuple[str, ...],
    replicates: int,
    samples_per_replicate: int,
    variance_statistic: PlanningVarianceStatistic,
) -> ReplicatedDirectPilot:
    """Collect disjoint planning replicates without touching final streams."""

    replicates, samples_per_replicate, variance_statistic = (
        validate_replicated_planning(
            replicates=replicates,
            samples_per_replicate=samples_per_replicate,
            variance_statistic=variance_statistic,
        )
    )
    batches: list[torch.Tensor] = []
    variances: list[float] = []
    work_units = 0.0
    wall_seconds = 0.0
    cpu_seconds = 0.0
    for replicate in range(replicates):
        seeds = {
            stream: ledger.allocate(
                SeedKey(
                    protocol,
                    "allocation-pilot",
                    f"{cell_id}:cluster-{cluster}",
                    method,
                    level,
                    replicate,
                    stream,
                )
            )
            for stream in streams
        }
        cpu_started = time.process_time()
        batch = sampler(profile_id, "pilot", samples_per_replicate, seeds)
        cpu_seconds += time.process_time() - cpu_started
        values = batch.values.detach().clone()
        if (
            values.ndim != 1
            or values.numel() != samples_per_replicate
            or not torch.isfinite(values).all()
        ):
            raise ValueError("replicated planning sampler returned malformed values")
        batches.append(values)
        variances.append(float(torch.var(values, unbiased=True)))
        work_units += float(batch.work_units)
        wall_seconds += float(batch.wall_seconds)
    replicate_tuple = tuple(variances)
    return ReplicatedDirectPilot(
        values=torch.cat(batches),
        replicate_variances=replicate_tuple,
        planning_variance=replicated_variance_statistic(
            replicate_tuple, variance_statistic
        ),
        replicates=replicates,
        samples_per_replicate=samples_per_replicate,
        work_units=work_units,
        wall_seconds=wall_seconds,
        cpu_seconds=cpu_seconds,
    )


def plugin_design_variance(
    planning_variance: float,
    *,
    safety_factor: float,
    zero_variance_fallback: float,
) -> float:
    """Apply the frozen engineering safety factor and positive fallback."""

    if not math.isfinite(planning_variance) or planning_variance < 0.0:
        raise ValueError("planning variance must be finite and nonnegative")
    if not math.isfinite(safety_factor) or safety_factor < 1.0:
        raise ValueError("allocation safety factor must be finite and at least one")
    if not math.isfinite(zero_variance_fallback) or zero_variance_fallback <= 0.0:
        raise ValueError("zero-variance fallback must be finite and positive")
    return max(safety_factor * planning_variance, zero_variance_fallback)


def replicated_direct_certificate(
    pilot: ReplicatedDirectPilot,
    *,
    method: str,
    variance_statistic: PlanningVarianceStatistic,
    safety_factor: float,
    zero_variance_fallback: float,
    selected_design_variance: float,
    absolute_bound: float,
    nominal_probability: float,
    nominal_probability_upper_multiplier: float,
    reference_probability: float,
    reference_standard_error: float,
    reference_certificate_z: float,
) -> dict[str, Any]:
    """Serialize replayable point-design arithmetic and a non-allocation bound."""

    if (
        not math.isfinite(absolute_bound)
        or absolute_bound <= 0.0
        or not 0.0 < nominal_probability <= 1.0
        or not math.isfinite(nominal_probability_upper_multiplier)
        or nominal_probability_upper_multiplier < 1.0
        or not math.isfinite(reference_probability)
        or reference_probability < 0.0
        or not math.isfinite(reference_standard_error)
        or reference_standard_error < 0.0
        or not math.isfinite(reference_certificate_z)
        or reference_certificate_z <= 0.0
    ):
        raise ValueError("invalid replicated direct-planning certificate inputs")
    probability_upper = min(
        1.0, nominal_probability_upper_multiplier * nominal_probability
    )
    reference_upper = (
        reference_probability + reference_certificate_z * reference_standard_error
    )
    bounded_values = bool(
        torch.all(pilot.values >= -1e-12)
        and torch.all(pilot.values <= absolute_bound + 1e-12)
    )
    return {
        "schema": "npi.g11.v6-replicated-direct-plugin-certificate.v1",
        "method": method,
        "planning_replicates": pilot.replicates,
        "samples_per_replicate": pilot.samples_per_replicate,
        "variance_statistic": variance_statistic,
        "replicate_variances": list(pilot.replicate_variances),
        "planning_variance": pilot.planning_variance,
        "pooled_count": int(pilot.values.numel()),
        "pooled_mean": float(torch.mean(pilot.values)),
        "pooled_variance": float(torch.var(pilot.values, unbiased=True)),
        "variance_safety_factor": safety_factor,
        "zero_variance_fallback": zero_variance_fallback,
        "selected_design_variance": selected_design_variance,
        "absolute_bound": absolute_bound,
        "pilot_values_within_nonnegative_bound": bounded_values,
        "nominal_probability": nominal_probability,
        "nominal_probability_upper_multiplier": (
            nominal_probability_upper_multiplier
        ),
        "probability_upper_bound": probability_upper,
        "reference_certificate_z": reference_certificate_z,
        "reference_upper_bound": reference_upper,
        "structural_variance_upper_diagnostic": absolute_bound
        * probability_upper,
        "structural_bound_used_for_allocation": False,
        "certified": bounded_values and reference_upper <= probability_upper,
    }
