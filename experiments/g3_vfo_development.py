"""Staged development runner for the first task-specific VFO prototype."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from src.path_integral.controllers import VolterraFollmerOperator
from src.physics_engine import RBergomiSimulator
from src.training.vfo import train_vfo_stage


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("expected a G3 schema-version-1 config")
    return payload


def _evaluate(
    simulator: RBergomiSimulator,
    control: VolterraFollmerOperator | None,
    *,
    model: dict[str, Any],
    barrier: float,
    paths: int,
    seed: int,
    method: str,
    event_type: str,
) -> dict[str, float | str]:
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
    elapsed = time.perf_counter() - start
    event_value = (
        result.spot[:, -1]
        if event_type == "terminal"
        else result.running_minimum[:, -1]
    )
    if event_type not in ("terminal", "down_barrier"):
        raise ValueError("unknown event_type")
    event = (event_value <= barrier).double()
    contribution = event * torch.exp(result.log_likelihood)
    estimate = float(contribution.mean())
    standard_error = float(contribution.std(unbiased=True) / math.sqrt(paths))
    ess = float(
        contribution.sum().square() / contribution.square().sum().clamp_min(1e-300)
    )
    return {
        "method": method,
        "seed": float(seed),
        "estimate": estimate,
        "standard_error": standard_error,
        "single_path_variance": float(contribution.var(unbiased=True)),
        "event_fraction": float(event.mean()),
        "contribution_ess_fraction": ess / paths,
        "seconds": elapsed,
        "cost_per_path": elapsed / paths,
    }


def run(config_path: Path, *, smoke: bool = False) -> dict[str, Any]:
    config = _load(config_path)
    model = config["model"]
    task = config["task"]
    architecture = config["controller"]
    training = config["training"]
    seeds = config["seeds"]
    simulator = RBergomiSimulator(
        H=float(model["H"]),
        eta=float(model["eta"]),
        xi=float(model["xi"]),
        rho=float(model["rho"]),
        device="cpu",
    )
    torch.manual_seed(int(seeds["initialization"]))
    control = VolterraFollmerOperator(
        H=float(model["H"]),
        rho=float(model["rho"]),
        eta=float(model["eta"]),
        xi=float(model["xi"]),
        maturity=float(model["maturity"]),
        barrier=float(task["hard_barrier"]),
        minimum_dt=float(model["dt"]),
        soe_terms=int(architecture["soe_terms"]),
        hidden_dim=int(architecture["hidden_dim"]),
        residual_dim=int(architecture["residual_dim"]),
        control_bound=tuple(float(value) for value in architecture["control_bound"]),
    ).double()
    updates = 2 if smoke else None
    paths_per_batch = 256 if smoke else int(training["paths_per_batch"])
    common_soft = {
        "spot": float(model["spot"]),
        "maturity": float(model["maturity"]),
        "dt": 1.0 / 16.0 if smoke else float(model["dt"]),
        "num_paths": paths_per_batch,
        "barrier": float(task["soft_barrier"]),
        "soft_scale": float(task["soft_scale"]),
        "event_type": str(task["event_type"]),
    }
    records: dict[str, Any] = {}
    timings: dict[str, float] = {}

    def execute(
        target_control: VolterraFollmerOperator,
        name: str,
        **kwargs: Any,
    ) -> None:
        start = time.perf_counter()
        stage_records = train_vfo_stage(simulator, target_control, **kwargs)
        timings[name] = time.perf_counter() - start
        records[name] = [asdict(record) for record in stage_records]

    execute(
        control,
        "instant_pi",
        stage="instant",
        objective="pi",
        updates=updates or int(training["instant_pi_updates"]),
        learning_rate=float(training["pi_learning_rate"]),
        seed=int(seeds["instant_pi"]),
        gradient_clip=float(training["gradient_clip"]),
        **common_soft,
    )
    # Fair ablations branch from the same B0 checkpoint and receive the same
    # number and sequence of objective updates. Architecture is the only
    # intended difference.
    instant_matched = control.frozen_copy()
    execute(
        instant_matched,
        "instant_matched_pi",
        stage="instant",
        objective="pi",
        updates=updates or int(training["structural_pi_updates"]),
        learning_rate=float(training["pi_learning_rate"]),
        seed=int(seeds["structural_pi"]),
        gradient_clip=float(training["gradient_clip"]),
        **common_soft,
    )
    execute(
        instant_matched,
        "instant_matched_pice_1",
        stage="instant",
        objective="pice",
        updates=updates or int(training["structural_pice_updates"]),
        learning_rate=float(training["pice_learning_rate"]),
        seed=int(seeds["structural_pice"]),
        gradient_clip=float(training["gradient_clip"]),
        behavior_refresh=int(training["behavior_refresh"]),
        **common_soft,
    )
    execute(
        instant_matched,
        "instant_matched_pice_2",
        stage="instant",
        objective="pice",
        updates=updates or int(training["residual_pice_updates"]),
        learning_rate=float(training["pice_learning_rate"]),
        seed=int(seeds["residual_pice"]),
        gradient_clip=float(training["gradient_clip"]),
        behavior_refresh=int(training["behavior_refresh"]),
        **common_soft,
    )

    hard_common = dict(common_soft)
    hard_common.pop("soft_scale")
    hard_common["barrier"] = float(task["hard_barrier"])
    execute(
        instant_matched,
        "instant_matched_j2",
        stage="instant",
        objective="j2",
        updates=updates or int(training["joint_j2_updates"]),
        learning_rate=float(training["j2_learning_rate"]),
        seed=int(seeds["joint_j2"]),
        gradient_clip=float(training["gradient_clip"]),
        **hard_common,
    )
    instant_matched = instant_matched.frozen_copy()

    execute(
        control,
        "structural_pi",
        stage="structural",
        objective="pi",
        updates=updates or int(training["structural_pi_updates"]),
        learning_rate=float(training["pi_learning_rate"]),
        seed=int(seeds["structural_pi"]),
        gradient_clip=float(training["gradient_clip"]),
        **common_soft,
    )
    execute(
        control,
        "structural_pice",
        stage="structural",
        objective="pice",
        updates=updates or int(training["structural_pice_updates"]),
        learning_rate=float(training["pice_learning_rate"]),
        seed=int(seeds["structural_pice"]),
        gradient_clip=float(training["gradient_clip"]),
        behavior_refresh=int(training["behavior_refresh"]),
        **common_soft,
    )
    structural_matched = control.frozen_copy()
    execute(
        structural_matched,
        "structural_matched_pice",
        stage="structural",
        objective="pice",
        updates=updates or int(training["residual_pice_updates"]),
        learning_rate=float(training["pice_learning_rate"]),
        seed=int(seeds["residual_pice"]),
        gradient_clip=float(training["gradient_clip"]),
        behavior_refresh=int(training["behavior_refresh"]),
        **common_soft,
    )
    execute(
        structural_matched,
        "structural_matched_j2",
        stage="structural",
        objective="j2",
        updates=updates or int(training["joint_j2_updates"]),
        learning_rate=float(training["j2_learning_rate"]),
        seed=int(seeds["joint_j2"]),
        gradient_clip=float(training["gradient_clip"]),
        **hard_common,
    )
    structural_matched = structural_matched.frozen_copy()

    execute(
        control,
        "residual_pice",
        stage="residual",
        objective="pice",
        updates=updates or int(training["residual_pice_updates"]),
        learning_rate=float(training["pice_learning_rate"]),
        seed=int(seeds["residual_pice"]),
        gradient_clip=float(training["gradient_clip"]),
        behavior_refresh=int(training["behavior_refresh"]),
        **common_soft,
    )
    execute(
        control,
        "joint_j2",
        stage="joint",
        objective="j2",
        updates=updates or int(training["joint_j2_updates"]),
        learning_rate=float(training["j2_learning_rate"]),
        seed=int(seeds["joint_j2"]),
        gradient_clip=float(training["gradient_clip"]),
        **hard_common,
    )
    joint = control.frozen_copy()

    validation_paths = 3_000 if smoke else int(config["validation"]["paths_per_seed"])
    validation_seeds = list(
        seeds["validation"][:2] if smoke else seeds["validation"]
    )
    methods = {
        "natural": None,
        "instant_matched": instant_matched,
        "structural_matched": structural_matched,
        "full_vfo": joint,
    }
    runs = [
        _evaluate(
            simulator,
            method_control,
            model=model,
            barrier=float(task["hard_barrier"]),
            paths=validation_paths,
            seed=int(seed),
            method=method_name,
            event_type=str(task["event_type"]),
        )
        for method_name, method_control in methods.items()
        for seed in validation_seeds
    ]
    natural_estimate = sum(
        float(run["estimate"]) for run in runs if run["method"] == "natural"
    ) / len(validation_seeds)
    summary: dict[str, dict[str, float]] = {}
    for method_name in methods:
        selected = [run for run in runs if run["method"] == method_name]
        variance = sum(float(run["single_path_variance"]) for run in selected) / len(
            selected
        )
        cost = sum(float(run["cost_per_path"]) for run in selected) / len(selected)
        estimate = sum(float(run["estimate"]) for run in selected) / len(selected)
        standard_error = math.sqrt(
            sum(float(run["standard_error"]) ** 2 for run in selected)
        ) / len(selected)
        summary[method_name] = {
            "mean_estimate": estimate,
            "combined_standard_error": standard_error,
            "difference_z_vs_natural": (estimate - natural_estimate)
            / max(
                math.sqrt(
                    standard_error**2
                    + (
                        summary.get("natural", {}).get(
                            "combined_standard_error", standard_error
                        )
                    )
                    ** 2
                ),
                1e-300,
            ),
            "mean_single_path_variance": variance,
            "mean_cost_per_path": cost,
            "online_work_proxy": variance * cost,
        }
    instant_work = summary["instant_matched"]["online_work_proxy"]
    for values in summary.values():
        values["online_work_vrf_vs_instant"] = instant_work / max(
            values["online_work_proxy"], 1e-300
        )

    all_records = [record for stage in records.values() for record in stage]
    maximum_residual_fraction = max(
        float(record["residual_energy_fraction"]) for record in all_records
    )
    takeover = any(bool(record["takeover_alarm"]) for record in all_records)
    return {
        "protocol_id": config["protocol_id"],
        "protocol_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "protocol_frozen": bool(config.get("frozen", False)),
        "smoke": smoke,
        "soe_fit": {
            "relative_l2_error": control.soe_bank.relative_l2_error,
            "maximum_relative_error": control.soe_bank.maximum_relative_error,
        },
        "training": {"records": records, "seconds": timings},
        "validation": {"runs": runs, "summary": summary},
        "diagnostics": {
            "maximum_residual_energy_fraction": maximum_residual_fraction,
            "takeover_alarm": takeover,
        },
        "development_gates": {
            "no_takeover": not takeover
            and maximum_residual_fraction
            <= float(config["validation"]["maximum_residual_energy_fraction"]),
            "structural_online_work_improves": summary["structural_matched"][
                "online_work_vrf_vs_instant"
            ]
            > 1.0,
            "joint_online_work_improves": summary["full_vfo"][
                "online_work_vrf_vs_instant"
            ]
            > 1.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/g3_vfo_development.yaml")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    result = run(args.config, smoke=args.smoke)
    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.output is None:
        print(serialized)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
        print(args.output)


if __name__ == "__main__":
    main()
