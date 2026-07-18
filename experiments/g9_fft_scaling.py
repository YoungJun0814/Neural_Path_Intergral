"""Frozen correctness and scaling audit for the G9 FFT BLP engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from src.path_integral.rbergomi_fft import (
    RBergomiFFTInnovations,
    simulate_rbergomi_fft,
)
from src.physics_engine import RBergomiSimulator


def _load(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("expected a G9 FFT schema-version-1 config")
    if payload.get("frozen") is not True:
        raise ValueError("the FFT scaling protocol must be frozen")
    return payload, hashlib.sha256(raw).hexdigest()


def _simulator(config: dict[str, Any]) -> RBergomiSimulator:
    model = config["model"]
    return RBergomiSimulator(
        H=float(model["H"]),
        eta=float(model["eta"]),
        xi=float(model["xi"]),
        rho=float(model["rho"]),
        device="cpu",
    )


def _fit_log_slope_upper_95(steps: list[int], seconds: list[float]) -> tuple[float, float]:
    x = [math.log(value) for value in steps]
    y = [math.log(value) for value in seconds]
    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)
    centered_square = sum((value - mean_x) ** 2 for value in x)
    if centered_square <= 0.0:
        raise ValueError("scaling steps must not all be equal")
    slope = (
        sum((left - mean_x) * (right - mean_y) for left, right in zip(x, y, strict=True))
        / centered_square
    )
    intercept = mean_y - slope * mean_x
    residual = [right - intercept - slope * left for left, right in zip(x, y, strict=True)]
    degrees = len(x) - 2
    if degrees <= 0:
        raise ValueError("at least three scaling levels are required")
    residual_variance = sum(value * value for value in residual) / degrees
    standard_error = math.sqrt(max(residual_variance / centered_square, 0.0))
    # One-sided 95% Student-t critical values. Scaling protocols use df=3;
    # the table keeps the helper valid for smaller/larger diagnostic windows.
    critical_by_df = {
        1: 6.314,
        2: 2.920,
        3: 2.353,
        4: 2.132,
        5: 2.015,
        6: 1.943,
        7: 1.895,
        8: 1.860,
        9: 1.833,
        10: 1.812,
        11: 1.796,
        12: 1.782,
        13: 1.771,
        14: 1.761,
        15: 1.753,
        16: 1.746,
        17: 1.740,
        18: 1.734,
        19: 1.729,
        20: 1.725,
        21: 1.721,
        22: 1.717,
        23: 1.714,
        24: 1.711,
        25: 1.708,
        26: 1.706,
        27: 1.703,
        28: 1.701,
        29: 1.699,
        30: 1.697,
    }
    critical = critical_by_df.get(degrees, 1.645)
    upper = slope + critical * standard_error
    return slope, upper


def _maximum_replay_error(config: dict[str, Any], simulator: RBergomiSimulator) -> float:
    validation = config["validation"]
    paths = int(validation["replay_paths"])
    steps = int(validation["replay_steps"])
    generator = torch.Generator().manual_seed(int(validation["replay_seed"]))
    innovations = RBergomiFFTInnovations(
        local_standard_normal=torch.randn(
            paths, steps, 2, dtype=torch.float64, generator=generator
        ),
        price_standard_normal=torch.randn(paths, steps, dtype=torch.float64, generator=generator),
    )
    arguments = dict(
        S0=float(config["model"]["spot"]),
        T=float(config["model"]["maturity"]),
        dt=float(config["model"]["maturity"]) / steps,
        num_paths=paths,
        innovations=innovations,
    )
    direct = simulate_rbergomi_fft(simulator, method="direct", **arguments)
    fast = simulate_rbergomi_fft(simulator, method="fft", **arguments)
    return max(
        float(torch.max(torch.abs(left - right)))
        for left, right in (
            (fast.spot, direct.spot),
            (fast.variance, direct.variance),
            (fast.volterra, direct.volterra),
            (fast.log_likelihood, direct.log_likelihood),
        )
    )


def run(config_path: Path, *, smoke: bool = False) -> dict[str, Any]:
    config, digest = _load(config_path)
    simulator = _simulator(config)
    benchmark = config["benchmark"]
    steps = [int(value) for value in benchmark["steps"]]
    repetitions = int(benchmark["timed_repetitions"])
    paths = int(benchmark["paths"])
    if smoke:
        steps = steps[:3]
        repetitions = min(repetitions, 2)
        paths = min(paths, 16)
    torch.set_num_threads(int(benchmark["thread_count"]))
    timing_seeds = [int(value) for value in config["seeds"]["timing"]][:repetitions]
    records: list[dict[str, Any]] = []
    for level_steps in steps:
        for _ in range(int(benchmark["warmup_repetitions"])):
            torch.manual_seed(timing_seeds[0])
            simulate_rbergomi_fft(
                simulator,
                S0=float(config["model"]["spot"]),
                T=float(config["model"]["maturity"]),
                dt=float(config["model"]["maturity"]) / level_steps,
                num_paths=paths,
            )
        fft_times: list[float] = []
        reference_times: list[float] = []
        for seed in timing_seeds:
            torch.manual_seed(seed)
            start = time.perf_counter()
            simulate_rbergomi_fft(
                simulator,
                S0=float(config["model"]["spot"]),
                T=float(config["model"]["maturity"]),
                dt=float(config["model"]["maturity"]) / level_steps,
                num_paths=paths,
            )
            fft_times.append(time.perf_counter() - start)
            torch.manual_seed(seed + 1_000_000)
            start = time.perf_counter()
            simulator.simulate_controlled_two_driver(
                S0=float(config["model"]["spot"]),
                T=float(config["model"]["maturity"]),
                dt=float(config["model"]["maturity"]) / level_steps,
                num_paths=paths,
            )
            reference_times.append(time.perf_counter() - start)
        fft_median = statistics.median(fft_times)
        reference_median = statistics.median(reference_times)
        records.append(
            {
                "steps": level_steps,
                "paths": paths,
                "fft_seconds": fft_times,
                "reference_seconds": reference_times,
                "fft_median_seconds": fft_median,
                "reference_median_seconds": reference_median,
                "speedup": reference_median / fft_median,
            }
        )
    replay_error = _maximum_replay_error(config, simulator)
    slope, slope_upper = _fit_log_slope_upper_95(
        [int(record["steps"]) for record in records],
        [float(record["fft_median_seconds"]) for record in records],
    )
    validation = config["validation"]
    terminal_speedup = float(records[-1]["speedup"])
    gates = {
        "pathwise_replay": replay_error <= float(validation["maximum_path_error"]),
        "cost_exponent": slope_upper < float(validation["maximum_cost_exponent_upper_95"]),
        "terminal_speedup": terminal_speedup > float(validation["minimum_speedup_at_1024"]),
    }
    return {
        "protocol_id": config["protocol_id"],
        "protocol_sha256": digest,
        "smoke": smoke,
        "environment": {
            "torch_version": torch.__version__,
            "thread_count": torch.get_num_threads(),
            "device": "cpu",
            "dtype": "float64",
        },
        "replay_maximum_absolute_error": replay_error,
        "scaling": records,
        "fft_log_cost_slope": slope,
        "fft_log_cost_slope_upper_95_one_sided": slope_upper,
        "terminal_speedup": terminal_speedup,
        "gates": gates,
        "passed": all(gates.values()),
        "claim_scope": "measured finite-window wall-clock scaling; not an asymptotic theorem",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/g9_fft_scaling.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    result = run(arguments.config, smoke=arguments.smoke)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"passed": result["passed"], "gates": result["gates"]}, indent=2))


if __name__ == "__main__":
    main()
