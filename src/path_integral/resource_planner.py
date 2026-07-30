"""Conservative resource forecasts for sharded path-simulation workloads."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReferenceBenchmarkObservation:
    paths: int
    steps: int
    wall_seconds: float
    cpu_seconds: float
    peak_resident_memory_bytes: int
    artifact_bytes: int

    def __post_init__(self) -> None:
        if self.paths < 1 or self.steps < 1:
            raise ValueError("benchmark paths and steps must be positive")
        if (
            not math.isfinite(self.wall_seconds)
            or self.wall_seconds <= 0.0
            or not math.isfinite(self.cpu_seconds)
            or self.cpu_seconds < 0.0
        ):
            raise ValueError("benchmark times must be finite with positive wall time")
        if self.peak_resident_memory_bytes < 1 or self.artifact_bytes < 1:
            raise ValueError("benchmark memory and artifact size must be positive")

    @property
    def path_steps_per_wall_second(self) -> float:
        return self.paths * self.steps / self.wall_seconds

    @property
    def cpu_seconds_per_path_step(self) -> float:
        return self.cpu_seconds / (self.paths * self.steps)

    @property
    def artifact_bytes_per_path(self) -> float:
        return self.artifact_bytes / self.paths


def forecast_reference_resources(
    observations: list[ReferenceBenchmarkObservation],
    *,
    total_paths: int,
    steps: int,
    workers: int,
    parallel_efficiency: float,
    safety_factor: float,
    available_memory_bytes: int,
    maximum_wall_seconds: float,
) -> dict[str, Any]:
    """Forecast with the slowest observed throughput and largest memory footprint."""

    if not observations:
        raise ValueError("at least one benchmark observation is required")
    if total_paths < 1 or steps < 1 or workers < 1:
        raise ValueError("forecast paths, steps, and workers must be positive")
    if (
        not math.isfinite(parallel_efficiency)
        or not 0.0 < parallel_efficiency <= 1.0
        or not math.isfinite(safety_factor)
        or safety_factor < 1.0
    ):
        raise ValueError("invalid parallel efficiency or safety factor")
    if available_memory_bytes < 1 or maximum_wall_seconds <= 0.0:
        raise ValueError("resource limits must be positive")

    throughput = min(item.path_steps_per_wall_second for item in observations)
    effective_throughput = throughput * workers * parallel_efficiency
    wall = safety_factor * total_paths * steps / effective_throughput
    cpu = safety_factor * total_paths * steps * max(
        item.cpu_seconds_per_path_step for item in observations
    )
    memory = math.ceil(
        safety_factor
        * workers
        * max(item.peak_resident_memory_bytes for item in observations)
    )
    storage = math.ceil(
        safety_factor
        * total_paths
        * max(item.artifact_bytes_per_path for item in observations)
    )
    memory_feasible = memory <= available_memory_bytes
    wall_feasible = wall <= maximum_wall_seconds
    return {
        "schema": "npi.g11.v8-reference-resource-forecast.v1",
        "observation_count": len(observations),
        "total_paths": total_paths,
        "steps": steps,
        "workers": workers,
        "parallel_efficiency": parallel_efficiency,
        "safety_factor": safety_factor,
        "conservative_single_worker_path_steps_per_second": throughput,
        "predicted_wall_seconds": wall,
        "predicted_cpu_seconds": cpu,
        "predicted_peak_memory_bytes": memory,
        "predicted_storage_bytes": storage,
        "available_memory_bytes": available_memory_bytes,
        "maximum_wall_seconds": maximum_wall_seconds,
        "memory_feasible": memory_feasible,
        "wall_feasible": wall_feasible,
        "launch_authorized": memory_feasible and wall_feasible,
    }
