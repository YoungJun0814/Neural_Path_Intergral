"""Development gate for exact multimodal path-integral mixtures in rBergomi."""

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
    ConstantTwoDriverControl,
    LeanRBergomiControl,
    simulate_rbergomi_mixture,
)
from src.physics_engine import RBergomiSimulator
from src.training.path_mixture import train_lean_pi_pice, train_mixture_weight_j2
from src.training.rbergomi_cem import fit_rbergomi_two_driver_cem


def _load(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("expected a G4 rBergomi schema-version-1 config")
    return payload


def _new_control(config: dict[str, Any], mode: str) -> LeanRBergomiControl:
    model = config["model"]
    task = config["task"]
    architecture = config["controller"]
    torch.manual_seed(int(config["seeds"]["initialization"]))
    return LeanRBergomiControl(
        spot=float(model["spot"]),
        xi=float(model["xi"]),
        maturity=float(model["maturity"]),
        lower_threshold=float(task["lower_threshold"]),
        upper_threshold=float(task["upper_threshold"]),
        mode=mode,  # type: ignore[arg-type]
        hidden_dim=int(architecture["hidden_dim"]),
        control_bound=tuple(float(value) for value in architecture["control_bound"]),
    ).double()


def _contribution_summary(
    contribution: torch.Tensor,
    terminal: torch.Tensor,
    *,
    lower: float,
    upper: float,
    method: str,
    seed: int,
    seconds_per_path: float,
) -> dict[str, float | str]:
    paths = contribution.numel()
    left = terminal <= lower
    right = terminal >= upper
    total = contribution.sum()
    variance = float(contribution.var(unbiased=True))
    standard_error = math.sqrt(variance / paths)
    ess = float(total.square() / contribution.square().sum().clamp_min(1e-300))
    return {
        "method": method,
        "seed": float(seed),
        "estimate": float(contribution.mean()),
        "standard_error": standard_error,
        "single_path_variance": variance,
        "event_fraction": float((left | right).double().mean()),
        "left_event_fraction": float(left.double().mean()),
        "right_event_fraction": float(right.double().mean()),
        "left_contribution_share": float(
            contribution[left].sum() / total.clamp_min(1e-300)
        ),
        "right_contribution_share": float(
            contribution[right].sum() / total.clamp_min(1e-300)
        ),
        "contribution_ess_fraction": ess / paths,
        "cost_per_path": seconds_per_path,
        "online_work_proxy": variance * seconds_per_path,
    }


def _evaluate_single(
    simulator: RBergomiSimulator,
    control: LeanRBergomiControl | None,
    *,
    config: dict[str, Any],
    seed: int,
    method: str,
    paths: int,
    repeats: int,
) -> dict[str, float | str]:
    model = config["model"]
    task = config["task"]
    elapsed: list[float] = []
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
        elapsed.append(time.perf_counter() - start)
        if repeat == 0:
            retained = result
    assert retained is not None
    terminal = retained.spot[:, -1]
    event = (terminal <= float(task["lower_threshold"])) | (
        terminal >= float(task["upper_threshold"])
    )
    contribution = event.double() * torch.exp(retained.log_likelihood)
    return _contribution_summary(
        contribution,
        terminal,
        lower=float(task["lower_threshold"]),
        upper=float(task["upper_threshold"]),
        method=method,
        seed=seed,
        seconds_per_path=statistics.median(elapsed) / paths,
    )


def _evaluate_mixture(
    simulator: RBergomiSimulator,
    controls: list[Any],
    weights: torch.Tensor,
    *,
    config: dict[str, Any],
    seed: int,
    method: str,
    paths: int,
    repeats: int,
) -> tuple[dict[str, float | str], dict[str, float | str], float]:
    model = config["model"]
    task = config["task"]
    elapsed: list[float] = []
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
                label_generator=torch.Generator().manual_seed(seed + 900_000),
            )
        elapsed.append(time.perf_counter() - start)
        if repeat == 0:
            retained = sample
    assert retained is not None
    terminal = retained.paths.spot[:, -1]
    event = (terminal <= float(task["lower_threshold"])) | (
        terminal >= float(task["upper_threshold"])
    )
    balance = event.double() * torch.exp(retained.mixture_log_likelihood)
    component = event.double() * torch.exp(retained.selected_component_log_likelihood)
    cost = statistics.median(elapsed) / paths
    balance_result = _contribution_summary(
        balance,
        terminal,
        lower=float(task["lower_threshold"]),
        upper=float(task["upper_threshold"]),
        method=method,
        seed=seed,
        seconds_per_path=cost,
    )
    component_result = _contribution_summary(
        component,
        terminal,
        lower=float(task["lower_threshold"]),
        upper=float(task["upper_threshold"]),
        method=f"{method}_component_weight",
        seed=seed,
        seconds_per_path=cost,
    )
    return balance_result, component_result, retained.maximum_selected_replay_error


def _aggregate(
    runs: list[dict[str, float | str]], methods: list[str]
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for method in methods:
        selected = [run for run in runs if run["method"] == method]
        output[method] = {
            key: sum(float(run[key]) for run in selected) / len(selected)
            for key in (
                "estimate",
                "standard_error",
                "single_path_variance",
                "cost_per_path",
                "online_work_proxy",
                "contribution_ess_fraction",
                "left_contribution_share",
                "right_contribution_share",
            )
        }
    return output


def run(config_path: Path, *, smoke: bool = False) -> dict[str, Any]:
    config = _load(config_path)
    model = config["model"]
    task = config["task"]
    training = config["training"]
    validation = config["validation"]
    simulator = RBergomiSimulator(
        H=float(model["H"]),
        eta=float(model["eta"]),
        xi=float(model["xi"]),
        rho=float(model["rho"]),
        device="cpu",
    )
    controls = {
        mode: _new_control(config, mode) for mode in ("union", "left", "right")
    }
    training_paths = 256 if smoke else int(training["paths_per_batch"])
    pi_updates = 2 if smoke else int(training["pi_updates"])
    pice_updates = 2 if smoke else int(training["pice_updates"])
    training_records: dict[str, list[dict[str, Any]]] = {}
    training_seconds: dict[str, float] = {}
    for mode in ("union", "left", "right"):
        start = time.perf_counter()
        records = train_lean_pi_pice(
            simulator,
            controls[mode],
            spot=float(model["spot"]),
            maturity=float(model["maturity"]),
            dt=1.0 / 16.0 if smoke else float(model["dt"]),
            num_paths=training_paths,
            lower_threshold=float(
                task.get("soft_lower_threshold", task["lower_threshold"])
            ),
            upper_threshold=float(
                task.get("soft_upper_threshold", task["upper_threshold"])
            ),
            soft_scale=float(task["soft_scale"]),
            mode=mode,  # type: ignore[arg-type]
            pi_updates=pi_updates,
            pice_updates=pice_updates,
            pi_learning_rate=float(training["pi_learning_rate"]),
            pice_learning_rate=float(training["pice_learning_rate"]),
            gradient_clip=float(training["gradient_clip"]),
            seed=int(config["seeds"][f"{mode}_training"]),
            behavior_refresh=int(training["behavior_refresh"]),
        )
        training_seconds[mode] = time.perf_counter() - start
        training_records[mode] = [asdict(record) for record in records]

    fixed_weights = torch.full((2,), 0.5, dtype=torch.float64)
    weight_start = time.perf_counter()
    learned_weights, weight_records = train_mixture_weight_j2(
        simulator,
        [controls["left"], controls["right"]],
        fixed_weights,
        spot=float(model["spot"]),
        maturity=float(model["maturity"]),
        dt=1.0 / 16.0 if smoke else float(model["dt"]),
        num_paths=(
            512 if smoke else int(training["mixture_weight_paths_per_batch"])
        ),
        lower_threshold=float(task["lower_threshold"]),
        upper_threshold=float(task["upper_threshold"]),
        minimum_weight=float(training["mixture_minimum_weight"]),
        updates=2 if smoke else int(training["mixture_weight_updates"]),
        learning_rate=float(training["mixture_weight_learning_rate"]),
        gradient_clip=float(training["gradient_clip"]),
        seed=int(config["seeds"]["weight_training"]),
    )
    training_seconds["mixture_weight"] = time.perf_counter() - weight_start
    training_records["mixture_weight"] = [asdict(record) for record in weight_records]

    cem_start = time.perf_counter()
    left_cem = fit_rbergomi_two_driver_cem(
        simulator,
        spot=float(model["spot"]),
        maturity=float(model["maturity"]),
        dt=1.0 / 16.0 if smoke else float(model["dt"]),
        threshold=float(task["lower_threshold"]),
        mode="left",
        initial_control=(1.0, -1.0),
        num_paths=512 if smoke else int(training["cem_paths_per_iteration"]),
        seed=int(config["seeds"]["left_cem"]),
        max_iterations=2 if smoke else int(training["cem_max_iterations"]),
        elite_quantile=float(training["cem_elite_quantile"]),
        smoothing=float(training["cem_smoothing"]),
        min_elite_paths=32,
        control_bound=float(config["controller"]["control_bound"][0]),
        target_level_repetitions=1 if smoke else 2,
    )
    right_cem = fit_rbergomi_two_driver_cem(
        simulator,
        spot=float(model["spot"]),
        maturity=float(model["maturity"]),
        dt=1.0 / 16.0 if smoke else float(model["dt"]),
        threshold=float(task["upper_threshold"]),
        mode="right",
        initial_control=(-1.0, 1.0),
        num_paths=512 if smoke else int(training["cem_paths_per_iteration"]),
        seed=int(config["seeds"]["right_cem"]),
        max_iterations=2 if smoke else int(training["cem_max_iterations"]),
        elite_quantile=float(training["cem_elite_quantile"]),
        smoothing=float(training["cem_smoothing"]),
        min_elite_paths=32,
        control_bound=float(config["controller"]["control_bound"][0]),
        target_level_repetitions=1 if smoke else 2,
    )
    training_seconds["cem_constant_mixture"] = time.perf_counter() - cem_start
    training_records["left_cem"] = [asdict(record) for record in left_cem.history]
    training_records["right_cem"] = [asdict(record) for record in right_cem.history]
    cem_controls = [
        ConstantTwoDriverControl(*left_cem.control),
        ConstantTwoDriverControl(*right_cem.control),
    ]

    paths = 3_000 if smoke else int(validation["paths_per_seed"])
    repeats = 1 if smoke else int(validation["timing_repeats"])
    seeds = list(config["seeds"]["validation"][:2] if smoke else config["seeds"]["validation"])
    runs: list[dict[str, float | str]] = []
    maximum_replay_error = 0.0
    for seed in seeds:
        runs.extend(
            [
                _evaluate_single(
                    simulator,
                    None,
                    config=config,
                    seed=int(seed),
                    method="natural",
                    paths=paths,
                    repeats=repeats,
                ),
                _evaluate_single(
                    simulator,
                    controls["union"],
                    config=config,
                    seed=int(seed),
                    method="single_union_feedback",
                    paths=paths,
                    repeats=repeats,
                ),
                _evaluate_single(
                    simulator,
                    controls["left"],
                    config=config,
                    seed=int(seed),
                    method="left_expert_only",
                    paths=paths,
                    repeats=repeats,
                ),
                _evaluate_single(
                    simulator,
                    controls["right"],
                    config=config,
                    seed=int(seed),
                    method="right_expert_only",
                    paths=paths,
                    repeats=repeats,
                ),
            ]
        )
        for method, weights in (
            ("fixed_mixture", fixed_weights),
            ("learned_mixture", learned_weights),
            ("cem_constant_mixture", fixed_weights),
        ):
            method_controls = (
                cem_controls
                if method == "cem_constant_mixture"
                else [controls["left"], controls["right"]]
            )
            balance, component, error = _evaluate_mixture(
                simulator,
                method_controls,
                weights,
                config=config,
                seed=int(seed),
                method=method,
                paths=paths,
                repeats=repeats,
            )
            runs.extend((balance, component))
            maximum_replay_error = max(maximum_replay_error, error)

    methods = list(dict.fromkeys(str(run["method"]) for run in runs))
    summary = _aggregate(runs, methods)
    union_by_seed = {
        int(float(run["seed"])): float(run["online_work_proxy"])
        for run in runs
        if run["method"] == "single_union_feedback"
    }
    mixture_by_seed = {
        int(float(run["seed"])): float(run["online_work_proxy"])
        for run in runs
        if run["method"] == "learned_mixture"
    }
    paired_log_ratios = [
        math.log(union_by_seed[seed] / mixture_by_seed[seed]) for seed in union_by_seed
    ]
    geometric_work_vrf = math.exp(statistics.mean(paired_log_ratios))
    improving_seeds = sum(value > 0.0 for value in paired_log_ratios)
    strong_baseline_methods = (
        "natural",
        "single_union_feedback",
        "cem_constant_mixture",
    )
    work_by_method_seed = {
        method: {
            int(float(run["seed"])): float(run["online_work_proxy"])
            for run in runs
            if run["method"] == method
        }
        for method in strong_baseline_methods
    }
    paired_log_vs_best = [
        math.log(
            min(work_by_method_seed[method][seed] for method in strong_baseline_methods)
            / mixture_by_seed[seed]
        )
        for seed in mixture_by_seed
    ]
    geometric_work_vrf_vs_best = math.exp(statistics.mean(paired_log_vs_best))
    improving_vs_best_seeds = sum(value > 0.0 for value in paired_log_vs_best)

    natural = summary["natural"]
    for values in summary.values():
        difference_se = math.sqrt(
            values["standard_error"] ** 2 + natural["standard_error"] ** 2
        )
        values["difference_z_vs_natural"] = (
            values["estimate"] - natural["estimate"]
        ) / max(difference_se, 1e-300)

    mixture_values = summary["learned_mixture"]
    union_values = summary["single_union_feedback"]
    work_vrf_vs_natural = natural["online_work_proxy"] / mixture_values[
        "online_work_proxy"
    ]
    epsilon_absolute = (
        float(validation["target_relative_error"]) * natural["estimate"]
    )
    union_query_seconds = union_values["online_work_proxy"] / max(
        epsilon_absolute**2, 1e-300
    )
    mixture_query_seconds = mixture_values["online_work_proxy"] / max(
        epsilon_absolute**2, 1e-300
    )
    union_training_seconds = training_seconds["union"]
    mixture_training_seconds = (
        training_seconds["left"]
        + training_seconds["right"]
        + training_seconds["mixture_weight"]
    )
    if mixture_query_seconds < union_query_seconds:
        break_even_queries_vs_single = max(
            0,
            math.ceil(
                (mixture_training_seconds - union_training_seconds)
                / (union_query_seconds - mixture_query_seconds)
            ),
        )
    else:
        break_even_queries_vs_single = None
    natural_query_seconds = natural["online_work_proxy"] / max(
        epsilon_absolute**2, 1e-300
    )
    if mixture_query_seconds < natural_query_seconds:
        break_even_queries_vs_natural = math.ceil(
            mixture_training_seconds
            / (natural_query_seconds - mixture_query_seconds)
        )
    else:
        break_even_queries_vs_natural = None
    cem_values = summary["cem_constant_mixture"]
    cem_query_seconds = cem_values["online_work_proxy"] / max(
        epsilon_absolute**2, 1e-300
    )
    cem_training_seconds = training_seconds["cem_constant_mixture"]
    if mixture_query_seconds < cem_query_seconds:
        break_even_queries_vs_cem = max(
            0,
            math.ceil(
                (mixture_training_seconds - cem_training_seconds)
                / (cem_query_seconds - mixture_query_seconds)
            ),
        )
    else:
        break_even_queries_vs_cem = None

    minimum_mode_share = min(
        mixture_values["left_contribution_share"],
        mixture_values["right_contribution_share"],
    )
    passed = {
        "all_expert_replay_exact": maximum_replay_error <= 1e-10,
        "both_modes_contribute": minimum_mode_share
        >= float(validation["minimum_mode_contribution_share"]),
        "mixture_raw_variance_improves": mixture_values["single_path_variance"]
        < union_values["single_path_variance"],
        "mixture_work_improves": geometric_work_vrf
        >= float(validation["minimum_work_vrf"]),
        "mixture_beats_natural_work": work_vrf_vs_natural
        >= float(validation["minimum_work_vrf"]),
        "mixture_beats_best_strong_baseline": geometric_work_vrf_vs_best
        >= float(validation["minimum_work_vrf"]),
        "paired_best_direction_consistent": improving_vs_best_seeds
        >= int(validation["minimum_improving_seeds"]),
        "cem_baseline_converged": left_cem.converged and right_cem.converged,
        "paired_direction_consistent": improving_seeds
        >= int(validation["minimum_improving_seeds"]),
        "reported_bias_within_gate": abs(mixture_values["difference_z_vs_natural"])
        <= float(validation["maximum_absolute_difference_z"]),
    }
    return {
        "protocol_id": config["protocol_id"],
        "protocol_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "protocol_frozen": bool(config.get("frozen", False)),
        "smoke": smoke,
        "calibration": config["calibration"],
        "learned_weights": [float(value) for value in learned_weights],
        "training": {
            "seconds": training_seconds,
            "records": training_records,
            "union_parameter_count": sum(
                parameter.numel() for parameter in controls["union"].parameters()
            ),
            "mixture_parameter_count": sum(
                parameter.numel()
                for mode in ("left", "right")
                for parameter in controls[mode].parameters()
            ),
        },
        "validation": {
            "runs": runs,
            "summary": summary,
            "paired_log_work_ratios": paired_log_ratios,
            "geometric_work_vrf_vs_single": geometric_work_vrf,
            "work_vrf_vs_natural": work_vrf_vs_natural,
            "paired_log_work_ratios_vs_best": paired_log_vs_best,
            "geometric_work_vrf_vs_best": geometric_work_vrf_vs_best,
            "improving_vs_best_seeds": improving_vs_best_seeds,
            "improving_seeds": improving_seeds,
            "maximum_selected_replay_error": maximum_replay_error,
        },
        "break_even": {
            "target_relative_error": float(validation["target_relative_error"]),
            "single_query_seconds": union_query_seconds,
            "natural_query_seconds": natural_query_seconds,
            "cem_query_seconds": cem_query_seconds,
            "mixture_query_seconds": mixture_query_seconds,
            "single_training_seconds": union_training_seconds,
            "natural_training_seconds": 0.0,
            "cem_training_seconds": cem_training_seconds,
            "mixture_training_seconds": mixture_training_seconds,
            "queries_vs_single": break_even_queries_vs_single,
            "queries_vs_natural": break_even_queries_vs_natural,
            "queries_vs_cem": break_even_queries_vs_cem,
        },
        "cem": {
            "left_control": list(left_cem.control),
            "right_control": list(right_cem.control),
            "left_converged": left_cem.converged,
            "right_converged": right_cem.converged,
        },
        "gates": passed,
        "passed": all(passed.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/g4_rbergomi_mixture_development.yaml")
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
