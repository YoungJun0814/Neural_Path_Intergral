"""G0 falsification: time-piecewise CEM versus the strongest constant CEM."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from src.path_integral import DownsideExcursionTask, TimePiecewiseTwoDriverControl
from src.physics_engine import RBergomiSimulator
from src.training import PiecewiseCEMResult, fit_rbergomi_piecewise_cem


def _load(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("expected a G0 schema-version-1 config")
    if payload.get("frozen") is not True:
        raise ValueError("G0 protocol must be frozen")
    return payload


def _task(config: dict[str, Any]) -> DownsideExcursionTask:
    values = config["task"]
    return DownsideExcursionTask(
        hit_barrier=float(values["hit_barrier"]),
        stress_level=float(values["stress_level"]),
        minimum_occupation=float(values["minimum_occupation"]),
        hit_scale=float(values["hit_scale"]),
        occupation_scale=float(values["occupation_scale"]),
    )


def _simulator(config: dict[str, Any]) -> RBergomiSimulator:
    model = config["model"]
    return RBergomiSimulator(
        H=float(model["H"]),
        eta=float(model["eta"]),
        xi=float(model["xi"]),
        rho=float(model["rho"]),
        device="cpu",
    )


def _fit(
    simulator: RBergomiSimulator,
    task: DownsideExcursionTask,
    config: dict[str, Any],
    *,
    segments: int,
    seed: int,
    smoke: bool,
) -> tuple[PiecewiseCEMResult, float]:
    model, cem = config["model"], config["cem"]
    initial_pair = tuple(float(value) for value in cem["initial_segment"])
    if len(initial_pair) != 2:
        raise ValueError("initial_segment must contain two coordinates")
    start = time.perf_counter()
    result = fit_rbergomi_piecewise_cem(
        simulator,
        task,
        spot=float(model["spot"]),
        maturity=float(model["maturity"]),
        dt=1.0 / 16.0 if smoke else float(model["dt"]),
        initial_control=tuple((initial_pair[0], initial_pair[1]) for _ in range(segments)),
        num_paths=1_500 if smoke else int(cem["paths_per_iteration"]),
        seed=seed,
        max_iterations=3 if smoke else int(cem["max_iterations"]),
        elite_quantile=float(cem["elite_quantile"]),
        smoothing=float(cem["smoothing"]),
        min_elite_paths=32 if smoke else int(cem["min_elite_paths"]),
        control_bound=float(cem["control_bound"]),
        target_level_repetitions=(
            1 if smoke else int(cem["target_level_repetitions"])
        ),
    )
    return result, time.perf_counter() - start


def _statistics(
    contribution: torch.Tensor,
    *,
    event_fraction: float,
    elapsed: float,
    paths: int,
    method: str,
    seed: int,
) -> dict[str, float | str]:
    variance = float(contribution.var(unbiased=True))
    return {
        "method": method,
        "seed": float(seed),
        "estimate": float(contribution.mean()),
        "standard_error": math.sqrt(variance / paths),
        "single_path_variance": variance,
        "second_moment": float(contribution.square().mean()),
        "event_fraction": event_fraction,
        "cost_per_path": elapsed / paths,
        "online_work_proxy": variance * elapsed / paths,
    }


def _evaluate(
    simulator: RBergomiSimulator,
    task: DownsideExcursionTask,
    config: dict[str, Any],
    *,
    method: str,
    seed: int,
    paths: int,
    timing_repeats: int,
    control: TimePiecewiseTwoDriverControl | None,
) -> dict[str, float | str]:
    model = config["model"]
    elapsed: list[float] = []
    retained = None
    for repeat in range(timing_repeats):
        torch.manual_seed(seed)
        start = time.perf_counter()
        with torch.no_grad():
            sample = simulator.simulate_controlled_two_driver(
                S0=float(model["spot"]),
                T=float(model["maturity"]),
                dt=float(model["dt"]),
                num_paths=paths,
                control_fn=control,
                dtype=torch.float64,
            )
        elapsed.append(time.perf_counter() - start)
        if repeat == 0:
            retained = sample
    assert retained is not None
    event = task.hard_event(retained.spot, retained.step_dt)
    contribution = event.double() * torch.exp(retained.log_likelihood)
    return _statistics(
        contribution,
        event_fraction=float(event.double().mean()),
        elapsed=statistics.median(elapsed),
        paths=paths,
        method=method,
        seed=seed,
    )


def _mean_summary(runs: list[dict[str, float | str]], method: str) -> dict[str, float]:
    selected = [run for run in runs if run["method"] == method]
    keys = (
        "estimate",
        "standard_error",
        "single_path_variance",
        "second_moment",
        "event_fraction",
        "cost_per_path",
        "online_work_proxy",
    )
    result = {key: statistics.mean(float(run[key]) for run in selected) for key in keys}
    result["combined_standard_error"] = math.sqrt(
        sum(float(run["standard_error"]) ** 2 for run in selected)
    ) / len(selected)
    return result


def run(config_path: Path, *, smoke: bool = False) -> dict[str, Any]:
    config = _load(config_path)
    model, cem, validation, seeds = (
        config["model"],
        config["cem"],
        config["validation"],
        config["seeds"],
    )
    simulator, task = _simulator(config), _task(config)
    constant_fit, constant_seconds = _fit(
        simulator,
        task,
        config,
        segments=int(cem["constant_segments"]),
        seed=int(seeds["constant_training"]),
        smoke=smoke,
    )
    piecewise_fit, piecewise_seconds = _fit(
        simulator,
        task,
        config,
        segments=int(cem["piecewise_segments"]),
        seed=int(seeds["piecewise_training"]),
        smoke=smoke,
    )
    constant = TimePiecewiseTwoDriverControl(
        constant_fit.control, maturity=float(model["maturity"])
    )
    piecewise = TimePiecewiseTwoDriverControl(
        piecewise_fit.control, maturity=float(model["maturity"])
    )

    paths = 4_000 if smoke else int(validation["paths_per_seed"])
    repeats = 1 if smoke else int(validation["timing_repeats"])
    validation_seeds = list(seeds["validation"][:2] if smoke else seeds["validation"])
    runs: list[dict[str, float | str]] = []
    for seed in validation_seeds:
        for method, control in (
            ("natural", None),
            ("constant_cem", constant),
            ("piecewise_cem", piecewise),
        ):
            runs.append(
                _evaluate(
                    simulator,
                    task,
                    config,
                    method=method,
                    seed=int(seed),
                    paths=paths,
                    timing_repeats=repeats,
                    control=control,
                )
            )

    methods = ("natural", "constant_cem", "piecewise_cem")
    summary = {method: _mean_summary(runs, method) for method in methods}
    by_method = {
        method: {
            int(float(run["seed"])): run for run in runs if run["method"] == method
        }
        for method in methods
    }
    log_raw_ratios = [
        math.log(
            float(by_method["constant_cem"][seed]["single_path_variance"])
            / max(float(by_method["piecewise_cem"][seed]["single_path_variance"]), 1e-300)
        )
        for seed in by_method["constant_cem"]
    ]
    log_work_ratios = [
        math.log(
            float(by_method["constant_cem"][seed]["online_work_proxy"])
            / max(float(by_method["piecewise_cem"][seed]["online_work_proxy"]), 1e-300)
        )
        for seed in by_method["constant_cem"]
    ]
    raw_vrf = math.exp(statistics.mean(log_raw_ratios))
    work_vrf = math.exp(statistics.mean(log_work_ratios))
    improving_seeds = sum(value > 0.0 for value in log_raw_ratios)

    natural, candidate = summary["natural"], summary["piecewise_cem"]
    difference_z = (candidate["estimate"] - natural["estimate"]) / max(
        math.sqrt(
            candidate["combined_standard_error"] ** 2
            + natural["combined_standard_error"] ** 2
        ),
        1e-300,
    )
    epsilon = float(validation["target_relative_error"]) * max(natural["estimate"], 1e-12)
    constant_query_seconds = summary["constant_cem"]["online_work_proxy"] / epsilon**2
    piecewise_query_seconds = candidate["online_work_proxy"] / epsilon**2
    incremental_training = max(0.0, piecewise_seconds - constant_seconds)
    break_even = (
        math.ceil(incremental_training / (constant_query_seconds - piecewise_query_seconds))
        if piecewise_query_seconds < constant_query_seconds
        else None
    )
    gates = {
        "raw_vrf": raw_vrf >= float(validation["minimum_raw_vrf"]),
        "work_vrf": work_vrf >= float(validation["minimum_work_vrf"]),
        "paired_consistency": improving_seeds
        >= (1 if smoke else int(validation["minimum_improving_seeds"])),
        "reported_bias": abs(difference_z)
        <= float(validation["maximum_absolute_difference_z"]),
        "break_even": break_even is not None
        and break_even <= int(validation["maximum_break_even_queries"]),
    }
    return {
        "protocol_id": config["protocol_id"],
        "protocol_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "smoke": smoke,
        "task": config["task"],
        "fits": {
            "constant": {
                "control": constant_fit.control,
                "converged": constant_fit.converged,
                "seconds": constant_seconds,
                "history": [asdict(item) for item in constant_fit.history],
            },
            "piecewise": {
                "control": piecewise_fit.control,
                "converged": piecewise_fit.converged,
                "seconds": piecewise_seconds,
                "history": [asdict(item) for item in piecewise_fit.history],
            },
        },
        "validation": {
            "runs": runs,
            "summary": summary,
            "paired_log_raw_ratios": log_raw_ratios,
            "paired_log_work_ratios": log_work_ratios,
            "geometric_raw_vrf": raw_vrf,
            "geometric_work_vrf": work_vrf,
            "improving_seeds": improving_seeds,
            "difference_z_vs_natural": difference_z,
        },
        "break_even": {
            "queries": break_even,
            "incremental_training_seconds": incremental_training,
            "constant_query_seconds": constant_query_seconds,
            "piecewise_query_seconds": piecewise_query_seconds,
        },
        "gates": gates,
        "passed": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/g0_piecewise_cem.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    result = run(args.config, smoke=args.smoke)
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(args.output)


if __name__ == "__main__":
    main()
