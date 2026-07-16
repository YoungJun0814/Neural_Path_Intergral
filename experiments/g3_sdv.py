"""G3 falsification: SDV against the frozen four-segment CEM baseline."""

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

from src.path_integral import (
    DownsideExcursionTask,
    SpectralDoobVolterraControl,
    TimePiecewiseTwoDriverControl,
    brownian_log_likelihood,
    simulate_rbergomi_mixture,
)
from src.physics_engine import RBergomiSimulator
from src.training import train_sdv_regression


def _load(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("expected a G3 schema-version-1 config")
    if payload.get("frozen") is not True:
        raise ValueError("G3 protocol must be frozen")
    return payload


def _anchor_values(config: dict[str, Any]) -> tuple[tuple[float, float], ...]:
    values: list[tuple[float, float]] = []
    for pair in config["anchor"]["values"]:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("anchor values must be two-dimensional")
        values.append((float(pair[0]), float(pair[1])))
    if not values:
        raise ValueError("anchor values must be two-dimensional")
    return tuple(values)


def _pair(values: Any, *, name: str) -> tuple[float, float]:
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError(f"{name} must contain two values")
    return float(values[0]), float(values[1])


def _objects(
    config: dict[str, Any],
) -> tuple[
    RBergomiSimulator,
    DownsideExcursionTask,
    TimePiecewiseTwoDriverControl,
    SpectralDoobVolterraControl,
]:
    model, task_values, architecture = (
        config["model"],
        config["task"],
        config["architecture"],
    )
    simulator = RBergomiSimulator(
        H=float(model["H"]),
        eta=float(model["eta"]),
        xi=float(model["xi"]),
        rho=float(model["rho"]),
        device="cpu",
    )
    task = DownsideExcursionTask(
        hit_barrier=float(task_values["hit_barrier"]),
        stress_level=float(task_values["stress_level"]),
        minimum_occupation=float(task_values["minimum_occupation"]),
        hit_scale=float(task_values["hit_scale"]),
        occupation_scale=float(task_values["occupation_scale"]),
    )
    anchor_values = _anchor_values(config)
    anchor = TimePiecewiseTwoDriverControl(
        anchor_values, maturity=float(model["maturity"])
    )
    torch.manual_seed(int(config["seeds"]["initialization"]))
    sdv = SpectralDoobVolterraControl(
        H=float(model["H"]),
        spot=float(model["spot"]),
        xi=float(model["xi"]),
        maturity=float(model["maturity"]),
        hit_barrier=float(task_values["hit_barrier"]),
        stress_level=float(task_values["stress_level"]),
        minimum_occupation=float(task_values["minimum_occupation"]),
        minimum_dt=float(model["dt"]),
        anchor_values=anchor_values,
        soe_terms=int(architecture["soe_terms"]),
        hidden_dim=int(architecture["hidden_dim"]),
        control_bound=_pair(architecture["control_bound"], name="control_bound"),
        residual_bound=_pair(architecture["residual_bound"], name="residual_bound"),
        desirability_floor=float(architecture["desirability_floor"]),
        initial_desirability=float(architecture["initial_desirability"]),
    ).double()
    return simulator, task, anchor, sdv


def _evaluate(
    simulator: RBergomiSimulator,
    task: DownsideExcursionTask,
    config: dict[str, Any],
    *,
    method: str,
    control: Any | None,
    seed: int,
    paths: int,
    repeats: int,
) -> dict[str, float | str]:
    model = config["model"]
    timings: list[float] = []
    retained = None
    for repeat in range(repeats):
        torch.manual_seed(seed)
        start = time.perf_counter()
        with torch.no_grad():
            result = simulator.simulate_controlled_two_driver(
                S0=float(model["spot"]),
                T=float(model["maturity"]),
                dt=float(model["dt"]),
                num_paths=paths,
                control_fn=control,
                record_augmented=False,
                dtype=torch.float64,
            )
        timings.append(time.perf_counter() - start)
        if repeat == 0:
            retained = result
    assert retained is not None
    event = task.hard_event(retained.spot, retained.step_dt)
    contribution = event.double() * torch.exp(retained.log_likelihood)
    variance = float(contribution.var(unbiased=True))
    cost = statistics.median(timings) / paths
    return {
        "method": method,
        "seed": float(seed),
        "estimate": float(contribution.mean()),
        "standard_error": math.sqrt(variance / paths),
        "single_path_variance": variance,
        "event_fraction": float(event.double().mean()),
        "cost_per_path": cost,
        "online_work_proxy": variance * cost,
        "maximum_likelihood_reconstruction_error": 0.0,
    }


def _summary(runs: list[dict[str, float | str]], method: str) -> dict[str, float]:
    chosen = [run for run in runs if run["method"] == method]
    keys = (
        "estimate",
        "standard_error",
        "single_path_variance",
        "event_fraction",
        "cost_per_path",
        "online_work_proxy",
        "maximum_likelihood_reconstruction_error",
    )
    result = {key: statistics.mean(float(run[key]) for run in chosen) for key in keys}
    result["combined_standard_error"] = math.sqrt(
        sum(float(run["standard_error"]) ** 2 for run in chosen)
    ) / len(chosen)
    return result


def run(
    config_path: Path,
    *,
    smoke: bool = False,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    config = _load(config_path)
    model, training, validation, seeds = (
        config["model"],
        config["training"],
        config["validation"],
        config["seeds"],
    )
    simulator, task, anchor, sdv = _objects(config)
    audit_paths = 256
    torch.manual_seed(int(seeds["initialization"]))
    with torch.no_grad():
        anchor_audit = simulator.simulate_controlled_two_driver(
            S0=float(model["spot"]),
            T=float(model["maturity"]),
            dt=float(model["dt"]),
            num_paths=audit_paths,
            control_fn=anchor,
            record_augmented=True,
            dtype=torch.float64,
        )
    torch.manual_seed(int(seeds["initialization"]))
    with torch.no_grad():
        sdv_audit = simulator.simulate_controlled_two_driver(
            S0=float(model["spot"]),
            T=float(model["maturity"]),
            dt=float(model["dt"]),
            num_paths=audit_paths,
            control_fn=sdv,
            record_augmented=True,
            dtype=torch.float64,
        )
    assert anchor_audit.controls is not None and sdv_audit.controls is not None
    initialization_control_error = float(
        torch.max(torch.abs(anchor_audit.controls - sdv_audit.controls))
    )
    initialization_path_error = float(
        torch.max(torch.abs(anchor_audit.spot - sdv_audit.spot))
    )

    start = time.perf_counter()
    records = train_sdv_regression(
        simulator,
        sdv,
        task,
        updates=3 if smoke else int(training["updates"]),
        learning_rate=float(training["learning_rate"]),
        seed=int(seeds["training"]),
        gradient_clip=float(training["gradient_clip"]),
        spot=float(model["spot"]),
        maturity=float(model["maturity"]),
        dt=1.0 / 16.0 if smoke else float(model["dt"]),
        num_paths=256 if smoke else int(training["paths_per_update"]),
        natural_behavior_mass=float(training["natural_behavior_mass"]),
        moment_loss_weight=float(training["moment_loss_weight"]),
        anchor_loss_weight=float(training["anchor_loss_weight"]),
    )
    training_seconds = time.perf_counter() - start
    frozen_sdv = sdv.frozen_copy()
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "protocol_id": config["protocol_id"],
                "protocol_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "state_dict": frozen_sdv.state_dict(),
            },
            checkpoint_path,
        )

    torch.manual_seed(int(seeds["replay_audit"]))
    with torch.no_grad():
        replay_audit = simulate_rbergomi_mixture(
            simulator,
            (anchor, frozen_sdv),
            torch.tensor((0.5, 0.5), dtype=torch.float64),
            spot=float(model["spot"]),
            maturity=float(model["maturity"]),
            dt=float(model["dt"]),
            num_paths=256,
            label_generator=torch.Generator().manual_seed(
                int(seeds["replay_audit"]) + 1
            ),
        )
    assert replay_audit.paths.controls is not None
    assert replay_audit.paths.proposal_brownian_increments is not None
    reconstructed_selected = brownian_log_likelihood(
        replay_audit.paths.controls,
        replay_audit.paths.proposal_brownian_increments,
        replay_audit.paths.step_dt,
    )
    direct_likelihood_error = float(
        torch.max(
            torch.abs(reconstructed_selected - replay_audit.paths.log_likelihood)
        )
    )

    paths = 3_000 if smoke else int(validation["paths_per_seed"])
    repeats = 1 if smoke else int(validation["timing_repeats"])
    validation_seeds = list(seeds["validation"][:2] if smoke else seeds["validation"])
    runs: list[dict[str, float | str]] = []
    for seed in validation_seeds:
        for method, control in (
            ("natural", None),
            ("piecewise_cem", anchor),
            ("sdv", frozen_sdv),
        ):
            runs.append(
                _evaluate(
                    simulator,
                    task,
                    config,
                    method=method,
                    control=control,
                    seed=int(seed),
                    paths=paths,
                    repeats=repeats,
                )
            )
    methods = ("natural", "piecewise_cem", "sdv")
    summaries = {method: _summary(runs, method) for method in methods}
    indexed = {
        method: {
            int(float(run["seed"])): run for run in runs if run["method"] == method
        }
        for method in methods
    }
    log_raw = [
        math.log(
            float(indexed["piecewise_cem"][seed]["single_path_variance"])
            / max(float(indexed["sdv"][seed]["single_path_variance"]), 1e-300)
        )
        for seed in indexed["piecewise_cem"]
    ]
    log_work = [
        math.log(
            float(indexed["piecewise_cem"][seed]["online_work_proxy"])
            / max(float(indexed["sdv"][seed]["online_work_proxy"]), 1e-300)
        )
        for seed in indexed["piecewise_cem"]
    ]
    raw_vrf = math.exp(statistics.mean(log_raw))
    work_vrf = math.exp(statistics.mean(log_work))
    improving = sum(value > 0.0 for value in log_raw)
    natural, candidate = summaries["natural"], summaries["sdv"]
    difference_z = (candidate["estimate"] - natural["estimate"]) / max(
        math.sqrt(
            candidate["combined_standard_error"] ** 2
            + natural["combined_standard_error"] ** 2
        ),
        1e-300,
    )
    epsilon = float(validation["target_relative_error"]) * max(natural["estimate"], 1e-12)
    base_query = summaries["piecewise_cem"]["online_work_proxy"] / epsilon**2
    sdv_query = candidate["online_work_proxy"] / epsilon**2
    break_even = (
        math.ceil(training_seconds / (base_query - sdv_query))
        if sdv_query < base_query
        else None
    )
    maximum_replay_error = max(
        replay_audit.maximum_selected_replay_error,
        direct_likelihood_error,
    )
    gates = {
        "initialization_matches_anchor": initialization_control_error <= 2e-14,
        "exact_replay_and_likelihood": maximum_replay_error
        <= float(validation["maximum_replay_error"]),
        "raw_vrf": raw_vrf >= float(validation["minimum_raw_vrf"]),
        "work_vrf": work_vrf >= float(validation["minimum_work_vrf"]),
        "paired_consistency": improving
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
        "initialization": {
            "maximum_control_error": initialization_control_error,
            "maximum_spot_path_error": initialization_path_error,
        },
        "training": {
            "seconds": training_seconds,
            "records": [asdict(record) for record in records],
        },
        "validation": {
            "runs": runs,
            "summary": summaries,
            "geometric_raw_vrf": raw_vrf,
            "geometric_work_vrf": work_vrf,
            "improving_seeds": improving,
            "paired_log_raw_ratios": log_raw,
            "paired_log_work_ratios": log_work,
            "difference_z_vs_natural": difference_z,
            "maximum_replay_error": maximum_replay_error,
        },
        "break_even": {
            "queries": break_even,
            "training_seconds": training_seconds,
            "base_query_seconds": base_query,
            "sdv_query_seconds": sdv_query,
        },
        "gates": gates,
        "passed": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/g3_sdv.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    result = run(args.config, smoke=args.smoke, checkpoint_path=args.checkpoint)
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(args.output)


if __name__ == "__main__":
    main()
