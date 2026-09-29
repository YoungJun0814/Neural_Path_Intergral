"""Measure each research stage with consistent CPU, wall, and memory fields."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from src.path_integral.provenance import process_peak_resident_memory_bytes
from src.path_integral.research_result_contract import StageCost

T = TypeVar("T")


def measure_stage(
    stage: str, action: Callable[[], T], *, proxy_work_units: float = 0.0
) -> tuple[T, StageCost]:
    """The peak is process-lifetime RSS, not an allocated-memory delta."""

    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    result = action()
    cost = StageCost(
        stage=stage,
        wall_seconds=time.perf_counter() - wall_start,
        cpu_seconds=time.process_time() - cpu_start,
        peak_memory_bytes=process_peak_resident_memory_bytes(),
        proxy_work_units=proxy_work_units,
    )
    return result, cost
