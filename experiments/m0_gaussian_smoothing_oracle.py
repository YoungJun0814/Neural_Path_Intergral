"""Independent M0 algebraic oracle for MGVS thresholds and shifted Gaussian integration."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist

import torch

from src.path_integral.gaussian_smoothing import (
    downside_excursion_thresholds,
    stable_normal_cdf_difference,
)
from src.path_integral.path_functionals import DownsideExcursionTask


def _simpson_shifted_integral(threshold: float, shift: float, intervals: int = 100_000) -> float:
    lower = -12.0
    if threshold <= lower:
        raise ValueError("oracle threshold must exceed the finite integration boundary")
    if intervals % 2 != 0:
        raise ValueError("Simpson integration requires an even interval count")
    spacing = (threshold - lower) / intervals
    normalizer = 1.0 / math.sqrt(2.0 * math.pi)

    def integrand(value: float) -> float:
        return normalizer * math.exp(-0.5 * value * value - shift * value - 0.5 * shift * shift)

    total = integrand(lower) + integrand(threshold)
    total += 4.0 * sum(integrand(lower + spacing * index) for index in range(1, intervals, 2))
    total += 2.0 * sum(integrand(lower + spacing * index) for index in range(2, intervals, 2))
    return total * spacing / 3.0


def run() -> dict[str, object]:
    torch.manual_seed(9701)
    paths, steps, step_dt = 5_000, 8, 0.1
    coordinate = torch.randn(paths, dtype=torch.float64)
    innovations = 0.03 * torch.randn(paths, steps, dtype=torch.float64)
    intercept = torch.cat(
        (
            torch.full((paths, 1), math.log(100.0), dtype=torch.float64),
            math.log(100.0) + torch.cumsum(innovations - 0.012, dim=1),
        ),
        dim=1,
    )
    slope = torch.cat(
        (
            torch.zeros(paths, 1, dtype=torch.float64),
            torch.cumsum(0.012 + 0.02 * torch.rand(paths, steps, dtype=torch.float64), dim=1),
        ),
        dim=1,
    )
    task = DownsideExcursionTask(90.0, 95.0, 0.2, 2.0, 0.05)
    threshold = downside_excursion_thresholds(intercept, slope, step_dt=step_dt, task=task).combined
    spot = torch.exp(intercept + slope * coordinate.unsqueeze(1))
    mismatch = int(torch.count_nonzero(task.hard_event(spot, step_dt) != (coordinate <= threshold)))

    normal = NormalDist()
    quadrature_cases = ((-2.3, -1.1), (-0.4, 0.8), (1.7, 2.2))
    quadrature_errors = [
        abs(_simpson_shifted_integral(boundary, shift) - normal.cdf(boundary + shift))
        for boundary, shift in quadrature_cases
    ]
    tail_actual = float(
        stable_normal_cdf_difference(
            torch.tensor([-11.0], dtype=torch.float64),
            torch.tensor([-12.0], dtype=torch.float64),
        )[0]
    )
    tail_reference = 0.5 * (math.erfc(11.0 / math.sqrt(2.0)) - math.erfc(12.0 / math.sqrt(2.0)))
    tail_relative_error = abs(tail_actual - tail_reference) / tail_reference
    gates = {
        "event_threshold": mismatch == 0,
        "shifted_integral": max(quadrature_errors) < 1e-11,
        "tail_difference": tail_relative_error < 1e-12,
    }
    return {
        "protocol_id": "g9-m0-gaussian-smoothing-oracle-v1",
        "seed": 9701,
        "paths": paths,
        "event_mismatches": mismatch,
        "quadrature_absolute_errors": quadrature_errors,
        "tail_relative_error": tail_relative_error,
        "gates": gates,
        "passed": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = run()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"passed": result["passed"], "gates": result["gates"]}, indent=2))


if __name__ == "__main__":
    main()
