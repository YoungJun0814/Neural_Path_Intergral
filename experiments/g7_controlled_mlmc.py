"""G7 falsification experiment for controlled adjacent-grid rBergomi MLMC."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch
import yaml

from src.evaluation.multilevel import (
    break_even_query_count,
    optimal_mlmc_sample_counts,
    single_level_online_work,
)
from src.path_integral import DownsideExcursionTask, TimePiecewiseTwoDriverControl
from src.path_integral.rbergomi_coupling import simulate_coupled_rbergomi_adjacent
from src.path_integral.rbergomi_multilevel import simulate_coupled_rbergomi_mixture
from src.physics_engine import RBergomiSimulator
from src.training.rbergomi_multilevel_cem import (
    CorrectionCEMResult,
    fit_rbergomi_correction_cem,
)
from src.training.rbergomi_piecewise_cem import PiecewiseValues


def _load(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("expected a G7 schema-version-1 config")
    if payload.get("frozen") is not True:
        raise ValueError("G7 protocol must be frozen")
    levels = [int(value) for value in payload["hierarchy"]["fine_steps"]]
    if len(levels) < 2 or any(level < 2 or level % 2 for level in levels):
        raise ValueError("G7 levels must be even integers")
    if any(right != 2 * left for left, right in zip(levels[:-1], levels[1:], strict=True)):
        raise ValueError("G7 hierarchy must use adjacent dyadic levels")
    return payload, hashlib.sha256(raw).hexdigest()


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


def _piecewise_values(values: Any) -> PiecewiseValues:
    resolved = tuple(tuple(float(entry) for entry in row) for row in values)
    if not resolved or any(len(row) != 2 for row in resolved):
        raise ValueError("piecewise values must contain two coordinates per segment")
    return cast(PiecewiseValues, resolved)


def _piecewise(values: Any, maturity: float) -> TimePiecewiseTwoDriverControl:
    return TimePiecewiseTwoDriverControl(_piecewise_values(values), maturity=maturity)


def _level_statistics(
    contribution: torch.Tensor,
    *,
    disagreement: torch.Tensor,
    elapsed: float,
    likelihood: torch.Tensor,
    replay_error: float,
    level: int,
    fine_steps: int,
    method: str,
) -> dict[str, float | int | str]:
    paths = contribution.numel()
    variance = float(contribution.var(unbiased=True))
    likelihood_se = float(likelihood.std(unbiased=True)) / math.sqrt(paths)
    likelihood_mean = float(likelihood.mean())
    likelihood_z = (
        abs(likelihood_mean - 1.0) / likelihood_se
        if likelihood_se > 0.0
        else (0.0 if likelihood_mean == 1.0 else math.inf)
    )
    return {
        "method": method,
        "level": level,
        "fine_steps": fine_steps,
        "estimate": float(contribution.mean()),
        "standard_error": math.sqrt(variance / paths),
        "single_path_variance": variance,
        "second_moment": float(contribution.square().mean()),
        "disagreement_fraction": float(disagreement.double().mean()),
        "cost_per_path": elapsed / paths,
        "likelihood_mean": likelihood_mean,
        "likelihood_normalization_z": likelihood_z,
        "maximum_replay_error": replay_error,
    }


def _evaluate_coupled_level(
    simulator: RBergomiSimulator,
    task: DownsideExcursionTask,
    *,
    spot: float,
    maturity: float,
    fine_steps: int,
    level: int,
    paths: int,
    seed: int,
    method: str,
    controls: list[TimePiecewiseTwoDriverControl] | None,
    weights: torch.Tensor | None,
) -> dict[str, float | int | str]:
    if level <= 0:
        raise ValueError("coupled corrections are defined only above level zero")
    torch.manual_seed(seed)
    start = time.perf_counter()
    with torch.no_grad():
        if controls is None:
            coupled = simulate_coupled_rbergomi_adjacent(
                simulator,
                S0=spot,
                T=maturity,
                fine_steps=fine_steps,
                num_paths=paths,
                record_augmented=False,
            )
            log_likelihood = coupled.log_likelihood
            replay_error = 0.0
        else:
            if weights is None:
                raise ValueError("mixture evaluation requires weights")
            mixture = simulate_coupled_rbergomi_mixture(
                simulator,
                controls,
                weights,
                spot=spot,
                maturity=maturity,
                fine_steps=fine_steps,
                num_paths=paths,
                label_generator=torch.Generator().manual_seed(seed + 10_000_000),
            )
            coupled = mixture.paths
            log_likelihood = mixture.mixture_log_likelihood
            replay_error = mixture.maximum_selected_replay_error
    elapsed = time.perf_counter() - start
    fine_event = task.hard_event(coupled.fine.spot, coupled.fine.step_dt).double()
    coarse_event = task.hard_event(coupled.coarse.spot, coupled.coarse.step_dt).double()
    disagreement = fine_event != coarse_event
    payoff = fine_event - coarse_event
    likelihood = torch.exp(log_likelihood)
    return _level_statistics(
        payoff * likelihood,
        disagreement=disagreement,
        elapsed=elapsed,
        likelihood=likelihood,
        replay_error=replay_error,
        level=level,
        fine_steps=fine_steps,
        method=method,
    )


def _evaluate_base_level(
    simulator: RBergomiSimulator,
    task: DownsideExcursionTask,
    *,
    spot: float,
    maturity: float,
    fine_steps: int,
    paths: int,
    seed: int,
    method: str,
    control: TimePiecewiseTwoDriverControl | None,
) -> dict[str, float | int | str]:
    """Evaluate H_0 without constructing an unused coarser path."""
    torch.manual_seed(seed)
    start = time.perf_counter()
    with torch.no_grad():
        sample = simulator.simulate_controlled_two_driver(
            S0=spot,
            T=maturity,
            dt=maturity / fine_steps,
            num_paths=paths,
            control_fn=control,
            record_augmented=False,
            dtype=torch.float64,
        )
    elapsed = time.perf_counter() - start
    event = task.hard_event(sample.spot, sample.step_dt).double()
    likelihood = torch.exp(sample.log_likelihood)
    return _level_statistics(
        event * likelihood,
        disagreement=torch.zeros_like(event, dtype=torch.bool),
        elapsed=elapsed,
        likelihood=likelihood,
        replay_error=0.0,
        level=0,
        fine_steps=fine_steps,
        method=method,
    )


def _evaluate_single_level(
    simulator: RBergomiSimulator,
    task: DownsideExcursionTask,
    control: TimePiecewiseTwoDriverControl,
    *,
    spot: float,
    maturity: float,
    fine_steps: int,
    paths: int,
    seed: int,
) -> dict[str, float | str]:
    torch.manual_seed(seed)
    start = time.perf_counter()
    with torch.no_grad():
        sample = simulator.simulate_controlled_two_driver(
            S0=spot,
            T=maturity,
            dt=maturity / fine_steps,
            num_paths=paths,
            control_fn=control,
            record_augmented=False,
            dtype=torch.float64,
        )
    elapsed = time.perf_counter() - start
    event = task.hard_event(sample.spot, sample.step_dt).double()
    likelihood = torch.exp(sample.log_likelihood)
    contribution = event * likelihood
    variance = float(contribution.var(unbiased=True))
    return {
        "method": "single_level_piecewise_cem",
        "estimate": float(contribution.mean()),
        "standard_error": math.sqrt(variance / paths),
        "single_path_variance": variance,
        "second_moment": float(contribution.square().mean()),
        "event_fraction": float(event.mean()),
        "cost_per_path": elapsed / paths,
    }


def _method_summary(levels: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "estimate": sum(float(level["estimate"]) for level in levels),
        "standard_error": math.sqrt(sum(float(level["standard_error"]) ** 2 for level in levels)),
        "online_work_coefficient": sum(
            math.sqrt(float(level["single_path_variance"]) * float(level["cost_per_path"]))
            for level in levels
        )
        ** 2,
    }


def _aggregate_difference_z(runs: list[dict[str, Any]], left: str, right: str) -> float:
    count = len(runs)
    left_mean = statistics.mean(float(run[left]["estimate"]) for run in runs)
    right_mean = statistics.mean(float(run[right]["estimate"]) for run in runs)
    variance = sum(
        float(run[left]["standard_error"]) ** 2 + float(run[right]["standard_error"]) ** 2
        for run in runs
    ) / (count * count)
    return (left_mean - right_mean) / math.sqrt(variance) if variance > 0.0 else 0.0


def _log_slope(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or any(value <= 0.0 for value in y):
        return None
    log_x = [math.log(value) for value in x]
    log_y = [math.log(value) for value in y]
    mean_x = statistics.mean(log_x)
    mean_y = statistics.mean(log_y)
    denominator = sum((value - mean_x) ** 2 for value in log_x)
    if denominator == 0.0:
        return None
    return (
        sum((left - mean_x) * (right - mean_y) for left, right in zip(log_x, log_y, strict=True))
        / denominator
    )


def run(config_path: Path, *, smoke: bool) -> dict[str, Any]:
    config, config_sha256 = _load(config_path)
    task = _task(config)
    simulator = _simulator(config)
    model = config["model"]
    spot = float(model["spot"])
    maturity = float(model["maturity"])
    levels = [int(value) for value in config["hierarchy"]["fine_steps"]]
    anchor_values = _piecewise_values(config["event_cem_anchor"]["values"])
    anchor = _piecewise(anchor_values, maturity)
    zero = _piecewise(tuple((0.0, 0.0) for _ in anchor_values), maturity)
    correction = config["correction_cem"]
    training_paths = 3_000 if smoke else int(correction["paths_per_iteration"])
    training_iterations = 2 if smoke else int(correction["max_iterations"])
    min_disagreements = 24 if smoke else int(correction["min_disagreement_paths"])
    training_results: dict[int, CorrectionCEMResult] = {}
    training_start = time.perf_counter()
    for level, (fine_steps, seed) in enumerate(
        zip(levels[1:], config["seeds"]["correction_training"], strict=True), start=1
    ):
        training_results[level] = fit_rbergomi_correction_cem(
            simulator,
            task,
            spot=spot,
            maturity=maturity,
            fine_steps=fine_steps,
            initial_control=anchor_values,
            num_paths=training_paths,
            seed=int(seed),
            max_iterations=training_iterations,
            smoothing=float(correction["smoothing"]),
            min_disagreement_paths=min_disagreements,
            control_bound=float(correction["control_bound"]),
            parameter_tolerance=float(correction["parameter_tolerance"]),
            stable_repetitions=int(correction["stable_repetitions"]),
        )
    training_seconds = time.perf_counter() - training_start

    natural_weight = float(config["mixture"]["natural_weight"])
    mixture_weights = torch.tensor([natural_weight, 1.0 - natural_weight], dtype=torch.float64)
    validation_paths = 3_000 if smoke else int(config["validation"]["paths_per_seed"])
    validation_seeds = [int(seed) for seed in config["seeds"]["validation"]]
    if smoke:
        validation_seeds = validation_seeds[:2]
    validation_runs: list[dict[str, Any]] = []
    validation_start = time.perf_counter()
    for seed_index, seed in enumerate(validation_seeds):
        natural_levels: list[dict[str, Any]] = [
            _evaluate_base_level(
                simulator,
                task,
                spot=spot,
                maturity=maturity,
                fine_steps=levels[0],
                paths=validation_paths,
                seed=seed,
                method="natural_mlmc",
                control=None,
            )
        ]
        event_base = _evaluate_base_level(
            simulator,
            task,
            spot=spot,
            maturity=maturity,
            fine_steps=levels[0],
            paths=validation_paths,
            seed=seed + 20_000,
            method="event_cem_mlmc",
            control=anchor,
        )
        event_levels: list[dict[str, Any]] = [event_base]
        correction_levels: list[dict[str, Any]] = [{**event_base, "method": "correction_cem_mlmc"}]
        for level, fine_steps in enumerate(levels[1:], start=1):
            natural_levels.append(
                _evaluate_coupled_level(
                    simulator,
                    task,
                    spot=spot,
                    maturity=maturity,
                    fine_steps=fine_steps,
                    level=level,
                    paths=validation_paths,
                    seed=seed + level * 101,
                    method="natural_mlmc",
                    controls=None,
                    weights=None,
                )
            )
            event_levels.append(
                _evaluate_coupled_level(
                    simulator,
                    task,
                    spot=spot,
                    maturity=maturity,
                    fine_steps=fine_steps,
                    level=level,
                    paths=validation_paths,
                    seed=seed + 20_000 + level * 101,
                    method="event_cem_mlmc",
                    controls=[zero, anchor],
                    weights=mixture_weights,
                )
            )
            fitted = _piecewise(training_results[level].control, maturity)
            correction_levels.append(
                _evaluate_coupled_level(
                    simulator,
                    task,
                    spot=spot,
                    maturity=maturity,
                    fine_steps=fine_steps,
                    level=level,
                    paths=validation_paths,
                    seed=seed + 40_000 + level * 101,
                    method="correction_cem_mlmc",
                    controls=[zero, fitted],
                    weights=mixture_weights,
                )
            )
        single = _evaluate_single_level(
            simulator,
            task,
            anchor,
            spot=spot,
            maturity=maturity,
            fine_steps=levels[-1],
            paths=validation_paths,
            seed=seed + 60_000,
        )
        natural_summary = _method_summary(natural_levels)
        event_summary = _method_summary(event_levels)
        correction_summary = _method_summary(correction_levels)
        baseline_coefficient = float(single["single_path_variance"]) * float(
            single["cost_per_path"]
        )
        validation_runs.append(
            {
                "seed": seed,
                "natural_levels": natural_levels,
                "event_levels": event_levels,
                "correction_levels": correction_levels,
                "natural_mlmc": natural_summary,
                "event_cem_mlmc": event_summary,
                "correction_cem_mlmc": correction_summary,
                "single_level": single,
                "event_work_ratio": baseline_coefficient / event_summary["online_work_coefficient"],
                "correction_work_ratio": baseline_coefficient
                / correction_summary["online_work_coefficient"],
                "seed_index": seed_index,
            }
        )
    validation_seconds = time.perf_counter() - validation_start

    work_ratios = [float(run["correction_work_ratio"]) for run in validation_runs]
    geometric_work_ratio = math.exp(statistics.mean(math.log(value) for value in work_ratios))
    improved_seeds = sum(value > 1.0 for value in work_ratios)
    nonzero_levels = range(1, len(levels))
    level_variance_ratios: dict[int, float] = {}
    for level in nonzero_levels:
        event_variance = statistics.geometric_mean(
            float(run["event_levels"][level]["single_path_variance"]) for run in validation_runs
        )
        correction_variance = statistics.geometric_mean(
            float(run["correction_levels"][level]["single_path_variance"])
            for run in validation_runs
        )
        level_variance_ratios[level] = event_variance / correction_variance
    improved_levels = sum(value > 1.0 for value in level_variance_ratios.values())

    reference_probability = statistics.mean(
        float(run["natural_mlmc"]["estimate"]) for run in validation_runs
    )
    relative_error = float(config["validation"]["target_relative_error"])
    variance_budget = (relative_error * reference_probability) ** 2
    baseline_works: list[float] = []
    candidate_works: list[float] = []
    allocations: list[dict[str, Any]] = []
    for run in validation_runs:
        single = run["single_level"]
        baseline_count, baseline_work = single_level_online_work(
            float(single["single_path_variance"]),
            float(single["cost_per_path"]),
            variance_budget=variance_budget,
        )
        correction_levels = run["correction_levels"]
        allocation = optimal_mlmc_sample_counts(
            [float(level["single_path_variance"]) for level in correction_levels],
            [float(level["cost_per_path"]) for level in correction_levels],
            variance_budget=variance_budget,
        )
        baseline_works.append(baseline_work)
        candidate_works.append(allocation.predicted_online_work)
        allocations.append(
            {
                "seed": run["seed"],
                "baseline_sample_count": baseline_count,
                "baseline_online_seconds": baseline_work,
                "mlmc": asdict(allocation),
            }
        )
    mean_baseline_work = statistics.mean(baseline_works)
    mean_candidate_work = statistics.mean(candidate_works)
    break_even = break_even_query_count(training_seconds, mean_baseline_work, mean_candidate_work)

    likelihood_z = max(
        float(level["likelihood_normalization_z"])
        for run in validation_runs
        for key in ("event_levels", "correction_levels")
        for level in run[key]
    )
    replay_error = max(
        float(level["maximum_replay_error"])
        for run in validation_runs
        for key in ("event_levels", "correction_levels")
        for level in run[key]
    )
    consistency_z = _aggregate_difference_z(validation_runs, "correction_cem_mlmc", "natural_mlmc")
    thresholds = config["validation"]
    gates = {
        "consistency": abs(consistency_z) <= float(thresholds["maximum_absolute_difference_z"]),
        "correction_variance": improved_levels
        >= int(thresholds["minimum_improved_correction_levels"]),
        "geometric_work_ratio": geometric_work_ratio
        > float(thresholds["minimum_geometric_work_ratio"]),
        "improving_seeds": improved_seeds >= int(thresholds["minimum_improving_seeds"]),
        "break_even": break_even <= float(thresholds["maximum_break_even_queries"]),
        "likelihood_normalization": likelihood_z
        <= float(thresholds["maximum_likelihood_normalization_z"]),
        "replay": replay_error <= float(thresholds["maximum_replay_error"]),
    }
    mean_natural_disagreement = [
        statistics.mean(
            float(run["natural_levels"][level]["disagreement_fraction"]) for run in validation_runs
        )
        for level in nonzero_levels
    ]
    mean_natural_variance = [
        statistics.mean(
            float(run["natural_levels"][level]["single_path_variance"]) for run in validation_runs
        )
        for level in nonzero_levels
    ]
    h_values = [maturity / levels[level] for level in nonzero_levels]
    return {
        "protocol_id": config["protocol_id"],
        "protocol_sha256": config_sha256,
        "smoke": smoke,
        "theory_contract": {
            "target": "finite finest-grid hard event expectation",
            "correction": "(H_fine-H_coarse) times one fine-space likelihood",
            "self_normalized": False,
            "continuous_rate_claimed": False,
        },
        "training": {
            "seconds": training_seconds,
            "paths_per_iteration": training_paths,
            "results": {str(level): asdict(result) for level, result in training_results.items()},
        },
        "validation": {
            "seconds": validation_seconds,
            "paths_per_seed": validation_paths,
            "runs": validation_runs,
            "aggregate_consistency_z": consistency_z,
            "level_event_over_correction_variance_ratio": level_variance_ratios,
            "improved_correction_levels": improved_levels,
            "geometric_work_ratio": geometric_work_ratio,
            "improving_work_seeds": improved_seeds,
            "maximum_likelihood_normalization_z": likelihood_z,
            "maximum_replay_error": replay_error,
            "reference_probability": reference_probability,
            "variance_budget": variance_budget,
            "allocations": allocations,
            "mean_baseline_online_seconds": mean_baseline_work,
            "mean_candidate_online_seconds": mean_candidate_work,
            "break_even_queries": break_even if math.isfinite(break_even) else None,
            "break_even_is_finite": math.isfinite(break_even),
        },
        "rate_diagnostics": {
            "h": h_values,
            "natural_disagreement_probability": mean_natural_disagreement,
            "natural_correction_variance": mean_natural_variance,
            "disagreement_log_slope": _log_slope(h_values, mean_natural_disagreement),
            "variance_log_slope": _log_slope(h_values, mean_natural_variance),
            "interpretation": "empirical finite-level diagnostic, not a theorem",
        },
        "gates": gates,
        "passed": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/g7_controlled_mlmc.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    result = run(args.config, smoke=args.smoke)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({"passed": result["passed"], "gates": result["gates"]}, indent=2))


if __name__ == "__main__":
    main()
