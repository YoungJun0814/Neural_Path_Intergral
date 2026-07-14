"""One-shot matched-work pilot for CEM-anchored neural residual experts."""

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
    CEMAnchoredResidualControl,
    ConstantTwoDriverControl,
    RBergomiTaskMode,
    simulate_rbergomi_mixture,
)
from src.physics_engine import RBergomiSimulator
from src.training.path_mixture import train_lean_pi_pice


def _load(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("expected a G5 schema-version-1 config")
    return payload


def _pair(values: Any, *, name: str) -> tuple[float, float]:
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    pair = (float(values[0]), float(values[1]))
    if not all(math.isfinite(value) for value in pair):
        raise ValueError(f"{name} must contain finite values")
    return pair


def _new_residual(
    config: dict[str, Any], mode: RBergomiTaskMode, base_control: tuple[float, float]
) -> CEMAnchoredResidualControl:
    model = config["model"]
    task = config["task"]
    architecture = config["residual"]
    torch.manual_seed(int(config["seeds"]["initialization"]))
    return CEMAnchoredResidualControl(
        spot=float(model["spot"]),
        xi=float(model["xi"]),
        maturity=float(model["maturity"]),
        lower_threshold=float(task["lower_threshold"]),
        upper_threshold=float(task["upper_threshold"]),
        mode=mode,
        hidden_dim=int(architecture["hidden_dim"]),
        control_bound=_pair(architecture["global_control_bound"], name="global_control_bound"),
        base_control=base_control,
        residual_bound=_pair(architecture["residual_bound"], name="residual_bound"),
    ).double()


def _statistics(
    contribution: torch.Tensor,
    terminal: torch.Tensor,
    *,
    lower: float,
    upper: float,
    cost_per_path: float,
    method: str,
    seed: int,
) -> dict[str, float | str]:
    left = terminal <= lower
    right = terminal >= upper
    total = contribution.sum().clamp_min(1e-300)
    variance = float(contribution.var(unbiased=True))
    return {
        "method": method,
        "seed": float(seed),
        "estimate": float(contribution.mean()),
        "standard_error": math.sqrt(variance / contribution.numel()),
        "single_path_variance": variance,
        "cost_per_path": cost_per_path,
        "online_work_proxy": variance * cost_per_path,
        "event_fraction": float((left | right).double().mean()),
        "left_contribution_share": float(contribution[left].sum() / total),
        "right_contribution_share": float(contribution[right].sum() / total),
        "contribution_ess_fraction": float(
            contribution.sum().square()
            / contribution.square().sum().clamp_min(1e-300)
            / contribution.numel()
        ),
    }


def _evaluate_natural(
    simulator: RBergomiSimulator,
    config: dict[str, Any],
    *,
    seed: int,
    paths: int,
    repeats: int,
) -> dict[str, float | str]:
    model = config["model"]
    task = config["task"]
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
                control_fn=None,
                dtype=torch.float64,
            )
        timings.append(time.perf_counter() - start)
        if repeat == 0:
            retained = result
    assert retained is not None
    terminal = retained.spot[:, -1]
    event = (terminal <= float(task["lower_threshold"])) | (
        terminal >= float(task["upper_threshold"])
    )
    return _statistics(
        event.double(),
        terminal,
        lower=float(task["lower_threshold"]),
        upper=float(task["upper_threshold"]),
        cost_per_path=statistics.median(timings) / paths,
        method="natural",
        seed=seed,
    )


def _evaluate_mixture(
    simulator: RBergomiSimulator,
    controls: list[Any],
    weights: torch.Tensor,
    config: dict[str, Any],
    *,
    seed: int,
    paths: int,
    repeats: int,
    method: str,
) -> tuple[dict[str, float | str], float]:
    model = config["model"]
    task = config["task"]
    timings: list[float] = []
    retained = None
    for repeat in range(repeats):
        torch.manual_seed(seed)
        start = time.perf_counter()
        with torch.no_grad():
            sample = simulate_rbergomi_mixture(
                simulator,
                controls,
                weights,
                spot=float(model["spot"]),
                maturity=float(model["maturity"]),
                dt=float(model["dt"]),
                num_paths=paths,
                dtype=torch.float64,
                label_generator=torch.Generator().manual_seed(seed + 700_000),
            )
        timings.append(time.perf_counter() - start)
        if repeat == 0:
            retained = sample
    assert retained is not None
    terminal = retained.paths.spot[:, -1]
    event = (terminal <= float(task["lower_threshold"])) | (
        terminal >= float(task["upper_threshold"])
    )
    contribution = event.double() * torch.exp(retained.mixture_log_likelihood)
    return (
        _statistics(
            contribution,
            terminal,
            lower=float(task["lower_threshold"]),
            upper=float(task["upper_threshold"]),
            cost_per_path=statistics.median(timings) / paths,
            method=method,
            seed=seed,
        ),
        retained.maximum_selected_replay_error,
    )


def _mean_summary(runs: list[dict[str, float | str]], method: str) -> dict[str, float]:
    selected = [run for run in runs if run["method"] == method]
    if not selected:
        raise ValueError(f"no runs found for method {method!r}")
    keys = (
        "estimate",
        "single_path_variance",
        "cost_per_path",
        "online_work_proxy",
        "left_contribution_share",
        "right_contribution_share",
        "contribution_ess_fraction",
    )
    result = {key: statistics.mean(float(run[key]) for run in selected) for key in keys}
    result["mean_per_seed_standard_error"] = statistics.mean(
        float(run["standard_error"]) for run in selected
    )
    result["combined_standard_error"] = math.sqrt(
        sum(float(run["standard_error"]) ** 2 for run in selected)
    ) / len(selected)
    return result


def run(config_path: Path, *, smoke: bool = False) -> dict[str, Any]:
    config = _load(config_path)
    model = config["model"]
    task = config["task"]
    base = config["base"]
    training = config["training"]
    validation = config["validation"]
    simulator = RBergomiSimulator(
        H=float(model["H"]),
        eta=float(model["eta"]),
        xi=float(model["xi"]),
        rho=float(model["rho"]),
        device="cpu",
    )
    bases: list[tuple[float, float]] = [
        _pair(base["left_control"], name="left_control"),
        _pair(base["right_control"], name="right_control"),
    ]
    base_controls = [ConstantTwoDriverControl(*value) for value in bases]
    modes: tuple[RBergomiTaskMode, RBergomiTaskMode] = ("left", "right")
    residual_controls = [
        _new_residual(config, mode, value) for mode, value in zip(modes, bases, strict=True)
    ]
    weights = torch.tensor(base["mixture_weights"], dtype=torch.float64)

    audit_seed = int(config["seeds"]["initialization_audit"])
    torch.manual_seed(audit_seed)
    with torch.no_grad():
        base_audit = simulate_rbergomi_mixture(
            simulator,
            base_controls,
            weights,
            spot=float(model["spot"]),
            maturity=float(model["maturity"]),
            dt=float(model["dt"]),
            num_paths=512,
            label_generator=torch.Generator().manual_seed(audit_seed + 1),
        )
    torch.manual_seed(audit_seed)
    with torch.no_grad():
        residual_audit = simulate_rbergomi_mixture(
            simulator,
            residual_controls,
            weights,
            spot=float(model["spot"]),
            maturity=float(model["maturity"]),
            dt=float(model["dt"]),
            num_paths=512,
            label_generator=torch.Generator().manual_seed(audit_seed + 1),
        )
    initialization_exact = bool(
        torch.equal(base_audit.paths.spot, residual_audit.paths.spot)
        and torch.equal(
            base_audit.component_log_q_over_p,
            residual_audit.component_log_q_over_p,
        )
        and torch.equal(
            base_audit.mixture_log_likelihood,
            residual_audit.mixture_log_likelihood,
        )
    )

    train_paths = 256 if smoke else int(training["paths_per_batch"])
    pi_updates = 2 if smoke else int(training["pi_updates"])
    pice_updates = 2 if smoke else int(training["pice_updates"])
    records: dict[str, list[dict[str, Any]]] = {}
    training_seconds: dict[str, float] = {}
    for mode, control in zip(modes, residual_controls, strict=True):
        start = time.perf_counter()
        history = train_lean_pi_pice(
            simulator,
            control,
            spot=float(model["spot"]),
            maturity=float(model["maturity"]),
            dt=1.0 / 16.0 if smoke else float(model["dt"]),
            num_paths=train_paths,
            lower_threshold=float(task["soft_lower_threshold"]),
            upper_threshold=float(task["soft_upper_threshold"]),
            soft_scale=float(task["soft_scale"]),
            mode=mode,
            pi_updates=pi_updates,
            pice_updates=pice_updates,
            pi_learning_rate=float(training["pi_learning_rate"]),
            pice_learning_rate=float(training["pice_learning_rate"]),
            gradient_clip=float(training["gradient_clip"]),
            seed=int(config["seeds"][f"{mode}_training"]),
            behavior_refresh=int(training["behavior_refresh"]),
        )
        training_seconds[mode] = time.perf_counter() - start
        records[mode] = [asdict(item) for item in history]

    paths = 3_000 if smoke else int(validation["paths_per_seed"])
    repeats = 1 if smoke else int(validation["timing_repeats"])
    seeds = list(config["seeds"]["validation"][:2] if smoke else config["seeds"]["validation"])
    runs: list[dict[str, float | str]] = []
    max_replay_error = 0.0
    for seed in seeds:
        runs.append(
            _evaluate_natural(simulator, config, seed=int(seed), paths=paths, repeats=repeats)
        )
        base_run, base_error = _evaluate_mixture(
            simulator,
            base_controls,
            weights,
            config,
            seed=int(seed),
            paths=paths,
            repeats=repeats,
            method="cem_base_mixture",
        )
        residual_run, residual_error = _evaluate_mixture(
            simulator,
            residual_controls,
            weights,
            config,
            seed=int(seed),
            paths=paths,
            repeats=repeats,
            method="cem_residual_mixture",
        )
        runs.extend((base_run, residual_run))
        max_replay_error = max(max_replay_error, base_error, residual_error)

    summary = {
        method: _mean_summary(runs, method)
        for method in ("natural", "cem_base_mixture", "cem_residual_mixture")
    }
    base_by_seed = {
        int(float(run["seed"])): float(run["online_work_proxy"])
        for run in runs
        if run["method"] == "cem_base_mixture"
    }
    residual_by_seed = {
        int(float(run["seed"])): float(run["online_work_proxy"])
        for run in runs
        if run["method"] == "cem_residual_mixture"
    }
    paired_log_ratios = [
        math.log(base_by_seed[seed] / residual_by_seed[seed]) for seed in base_by_seed
    ]
    geometric_work_vrf = math.exp(statistics.mean(paired_log_ratios))
    improving_seeds = sum(value > 0.0 for value in paired_log_ratios)

    natural = summary["natural"]
    residual = summary["cem_residual_mixture"]
    difference_z = (residual["estimate"] - natural["estimate"]) / max(
        math.sqrt(
            residual["combined_standard_error"] ** 2 + natural["combined_standard_error"] ** 2
        ),
        1e-300,
    )
    minimum_mode_share = min(
        residual["left_contribution_share"], residual["right_contribution_share"]
    )
    epsilon = float(validation["target_relative_error"]) * natural["estimate"]
    base_query_seconds = summary["cem_base_mixture"]["online_work_proxy"] / max(epsilon**2, 1e-300)
    residual_query_seconds = residual["online_work_proxy"] / max(epsilon**2, 1e-300)
    incremental_training = sum(training_seconds.values())
    if residual_query_seconds < base_query_seconds:
        break_even = math.ceil(incremental_training / (base_query_seconds - residual_query_seconds))
    else:
        break_even = None

    gates = {
        "exact_cem_initialization": initialization_exact,
        "all_expert_replay_exact": max_replay_error <= float(validation["maximum_replay_error"]),
        "both_modes_contribute": minimum_mode_share
        >= float(validation["minimum_mode_contribution_share"]),
        "reported_bias_within_gate": abs(difference_z)
        <= float(validation["maximum_absolute_difference_z"]),
        "raw_variance_improves": residual["single_path_variance"]
        < summary["cem_base_mixture"]["single_path_variance"],
        "work_improves": geometric_work_vrf >= float(validation["minimum_work_vrf"]),
        "paired_direction_consistent": improving_seeds
        >= int(validation["minimum_improving_seeds"]),
        "incremental_break_even": break_even is not None
        and break_even <= int(validation["maximum_break_even_queries"]),
    }
    return {
        "protocol_id": config["protocol_id"],
        "protocol_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "protocol_frozen": bool(config.get("frozen", False)),
        "smoke": smoke,
        "initialization_exact": initialization_exact,
        "training": {"seconds": training_seconds, "records": records},
        "validation": {
            "runs": runs,
            "summary": summary,
            "paired_log_work_ratios": paired_log_ratios,
            "geometric_work_vrf_vs_cem": geometric_work_vrf,
            "improving_seeds": improving_seeds,
            "maximum_replay_error": max_replay_error,
            "difference_z_vs_natural": difference_z,
        },
        "break_even": {
            "target_relative_error": float(validation["target_relative_error"]),
            "base_query_seconds": base_query_seconds,
            "residual_query_seconds": residual_query_seconds,
            "incremental_training_seconds": incremental_training,
            "queries": break_even,
        },
        "gates": gates,
        "passed": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/g5_cem_anchored_residual.yaml")
    )
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
