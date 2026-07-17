"""G8 falsification experiment for conditional Volterra bridge branching."""

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
from src.evaluation.volterra_branching import (
    evaluate_variable_branched_correction,
)
from src.path_integral import DownsideExcursionTask, TimePiecewiseTwoDriverControl
from src.path_integral.rbergomi_branching import (
    RBergomiCoarseTrunks,
    refine_rbergomi_coarse_trunks,
    sample_rbergomi_coarse_trunks,
)
from src.path_integral.rbergomi_coupling import simulate_coupled_rbergomi_adjacent
from src.physics_engine import RBergomiSimulator
from src.training.rbergomi_piecewise_cem import PiecewiseValues
from src.training.volterra_branching import (
    BoundaryVarianceFit,
    boundary_variance_scores,
    coarse_variance_features,
    fit_coarse_boundary_variance_model,
    score_threshold_branch_counts,
)


def _load(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("expected a G8 schema-version-1 config")
    if payload.get("frozen") is not True:
        raise ValueError("G8 protocol must be frozen")
    levels = [int(value) for value in payload["hierarchy"]["correction_fine_steps"]]
    base = int(payload["hierarchy"]["base_steps"])
    if not levels or levels[0] != 2 * base:
        raise ValueError("the first correction level must refine the base by two")
    if any(right != 2 * left for left, right in zip(levels[:-1], levels[1:], strict=True)):
        raise ValueError("G8 correction hierarchy must be dyadic")
    return payload, hashlib.sha256(raw).hexdigest()


def _piecewise_values(values: Any) -> PiecewiseValues:
    resolved = tuple(tuple(float(entry) for entry in row) for row in values)
    if not resolved or any(len(row) != 2 for row in resolved):
        raise ValueError("piecewise values must contain two coordinates per segment")
    return cast(PiecewiseValues, resolved)


def _control(values: Any, maturity: float) -> TimePiecewiseTwoDriverControl:
    return TimePiecewiseTwoDriverControl(_piecewise_values(values), maturity=maturity)


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
    values = config["model"]
    return RBergomiSimulator(
        H=float(values["H"]),
        eta=float(values["eta"]),
        xi=float(values["xi"]),
        rho=float(values["rho"]),
        device="cpu",
    )


def _branch_contributions(
    refined: Any,
    task: DownsideExcursionTask,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    trunks = refined.trunks
    parents, branches = refined.log_likelihood.shape
    fine_event = (
        task.hard_event(refined.fine_spot.reshape(-1, trunks.fine_steps + 1), trunks.fine_dt)
        .to(torch.float64)
        .reshape(parents, branches)
    )
    coarse_event = task.hard_event(trunks.spot, 2.0 * trunks.fine_dt).to(torch.float64)
    likelihood = torch.exp(refined.log_likelihood)
    return (fine_event - coarse_event[:, None]) * likelihood, fine_event, coarse_event


def _likelihood_z(likelihood_parent_mean: torch.Tensor) -> tuple[float, float]:
    mean = float(likelihood_parent_mean.mean())
    standard_error = float(likelihood_parent_mean.std(unbiased=True)) / math.sqrt(
        likelihood_parent_mean.numel()
    )
    z_score = (
        abs(mean - 1.0) / standard_error
        if standard_error > 0.0
        else (0.0 if mean == 1.0 else math.inf)
    )
    return mean, z_score


def _constant_control_energy(control_energy: torch.Tensor | float) -> float:
    if isinstance(control_energy, float):
        return control_energy
    flattened = control_energy.to(torch.float64).reshape(-1)
    resolved = float(flattened.mean())
    if float(torch.max(torch.abs(flattened - resolved))) > 1e-10:
        raise ValueError("G8 log-likelihood moment audit requires deterministic control energy")
    return resolved


def _log_likelihood_moment_z(
    log_likelihood_probe: torch.Tensor,
    control_energy: torch.Tensor | float,
) -> tuple[float, float, float, float, float]:
    """Audit the stable Gaussian moments of a deterministic Girsanov log weight.

    Raw likelihood means are retained in the report, but their studentized z
    statistic is unstable for the G8 controls (likelihood CV around 20--27).
    For deterministic controls, log L is exactly N(-E/2, E), so these two
    moment diagnostics test the implemented density without exponentiating it.
    """
    values = log_likelihood_probe.to(torch.float64).reshape(-1)
    energy = _constant_control_energy(control_energy)
    mean = float(values.mean())
    variance = float(values.var(unbiased=True))
    if energy == 0.0:
        mean_z = 0.0 if mean == 0.0 else math.inf
        variance_z = 0.0 if variance == 0.0 else math.inf
    else:
        mean_z = abs(mean + 0.5 * energy) / math.sqrt(energy / values.numel())
        variance_z = abs(variance - energy) / (energy * math.sqrt(2.0 / (values.numel() - 1)))
    return mean, variance, mean_z, variance_z, max(mean_z, variance_z)


def _summary(
    contribution: torch.Tensor,
    *,
    elapsed: float,
    likelihood_parent_mean: torch.Tensor,
    log_likelihood_probe: torch.Tensor,
    control_energy: torch.Tensor | float,
    method: str,
    fine_steps: int,
) -> dict[str, float | int | str]:
    variance = float(contribution.var(unbiased=True))
    likelihood_mean, likelihood_z = _likelihood_z(likelihood_parent_mean)
    log_mean, log_variance, log_mean_z, log_variance_z, log_moment_z = _log_likelihood_moment_z(
        log_likelihood_probe, control_energy
    )
    return {
        "method": method,
        "fine_steps": fine_steps,
        "estimate": float(contribution.mean()),
        "standard_error": math.sqrt(variance / contribution.numel()),
        "single_path_variance": variance,
        "second_moment": float(contribution.square().mean()),
        "cost_per_parent": elapsed / contribution.numel(),
        "likelihood_mean": likelihood_mean,
        "likelihood_normalization_z": likelihood_z,
        "log_likelihood_mean": log_mean,
        "log_likelihood_variance": log_variance,
        "log_likelihood_mean_z": log_mean_z,
        "log_likelihood_variance_z": log_variance_z,
        "log_likelihood_moment_z": log_moment_z,
    }


def _fit_allocator(
    trunks: RBergomiCoarseTrunks,
    task: DownsideExcursionTask,
    branch_values: torch.Tensor,
    config: dict[str, Any],
    *,
    seed: int,
    smoke: bool,
) -> tuple[BoundaryVarianceFit, dict[str, Any]]:
    development = config["development"]
    fit_paths = min(
        1_500 if smoke else int(development["fit_paths"]),
        branch_values.shape[0] // 2,
    )
    features = coarse_variance_features(
        trunks, task, feature_points=int(development["feature_points"])
    )
    conditional_variance = branch_values.var(dim=1, unbiased=True)
    fit = fit_coarse_boundary_variance_model(
        features[:fit_paths],
        conditional_variance[:fit_paths],
        seed=seed,
        high_variance_fraction=float(development["high_variance_fraction"]),
        hidden_dimension=int(development["hidden_dimension"]),
        epochs=20 if smoke else int(development["epochs"]),
        batch_size=int(development["batch_size"]),
        learning_rate=float(development["learning_rate"]),
        weight_decay=float(development["weight_decay"]),
    )
    training_scores = boundary_variance_scores(fit, features[:fit_paths])
    holdout_scores = boundary_variance_scores(fit, features[fit_paths:])
    holdout_values = branch_values[fit_paths:]
    fine_steps = trunks.fine_steps
    baseline_variance = float(holdout_values[:, 0].var(unbiased=True))
    baseline_work = baseline_variance * (fine_steps / 2.0 + fine_steps)
    maximum_branches = branch_values.shape[1]
    candidates: list[dict[str, float | int]] = []
    best: dict[str, float | int] | None = None
    branch_candidates = [
        int(value)
        for value in development["high_branch_candidates"]
        if int(value) <= maximum_branches
    ]
    for fraction_value in development["selection_fractions"]:
        fraction = float(fraction_value)
        threshold = float(torch.quantile(training_scores, 1.0 - fraction))
        selected = holdout_scores >= threshold
        for branches in branch_candidates:
            contribution = holdout_values[:, 0].clone()
            contribution[selected] = holdout_values[selected, :branches].mean(dim=1)
            variance = float(contribution.var(unbiased=True))
            mean_branches = 1.0 + (branches - 1.0) * float(selected.double().mean())
            work = variance * (fine_steps / 2.0 + fine_steps * mean_branches)
            candidate: dict[str, float | int] = {
                "selection_fraction": fraction,
                "score_threshold": threshold,
                "high_branches": branches,
                "holdout_selected_fraction": float(selected.double().mean()),
                "holdout_mean_branches": mean_branches,
                "holdout_variance": variance,
                "holdout_work_proxy": work,
                "holdout_work_ratio": baseline_work / work if work > 0.0 else math.inf,
            }
            candidates.append(candidate)
            if best is None or float(candidate["holdout_work_proxy"]) < float(
                best["holdout_work_proxy"]
            ):
                best = candidate
    if best is None:
        raise RuntimeError("allocator calibration produced no candidate")

    fixed_ratios: dict[int, float] = {}
    for branches in (1, 2, 4, maximum_branches):
        if branches > maximum_branches:
            continue
        contribution = holdout_values[:, :branches].mean(dim=1)
        work = float(contribution.var(unbiased=True)) * (fine_steps / 2.0 + fine_steps * branches)
        fixed_ratios[branches] = baseline_work / work if work > 0.0 else math.inf
    total_variance = float(holdout_values.reshape(-1).var(unbiased=True))
    conditional_component = float(holdout_values.var(dim=1, unbiased=True).mean())
    irreducible_fraction = (
        max(total_variance - conditional_component, 0.0) / total_variance
        if total_variance > 0.0
        else 0.0
    )
    top_labels = conditional_variance[fit_paths:] >= torch.quantile(
        conditional_variance[fit_paths:],
        1.0 - float(development["high_variance_fraction"]),
    )
    predicted_top = holdout_scores >= torch.quantile(
        holdout_scores, 1.0 - float(development["high_variance_fraction"])
    )
    top_recall = float((top_labels & predicted_top).sum() / top_labels.sum())
    return fit, {
        "fit_paths": fit_paths,
        "holdout_paths": branch_values.shape[0] - fit_paths,
        "selected": best,
        "fixed_work_ratios": fixed_ratios,
        "irreducible_variance_fraction": irreducible_fraction,
        "top_variance_recall": top_recall,
        "training_history": [asdict(record) for record in fit.history],
    }


def _checkpoint_payload(
    fit: BoundaryVarianceFit, calibration: dict[str, Any], feature_points: int
) -> dict[str, Any]:
    selected = calibration["selected"]
    return {
        "state_dict": fit.model.state_dict(),
        "feature_mean": fit.feature_mean,
        "feature_scale": fit.feature_scale,
        "score_threshold": float(selected["score_threshold"]),
        "high_branches": int(selected["high_branches"]),
        "feature_points": feature_points,
    }


def _evaluate_base_or_single(
    simulator: RBergomiSimulator,
    task: DownsideExcursionTask,
    control: TimePiecewiseTwoDriverControl,
    *,
    spot: float,
    maturity: float,
    steps: int,
    paths: int,
    seed: int,
    method: str,
) -> dict[str, float | int | str]:
    torch.manual_seed(seed)
    start = time.perf_counter()
    sample = simulator.simulate_controlled_two_driver(
        S0=spot,
        T=maturity,
        dt=maturity / steps,
        num_paths=paths,
        control_fn=control,
        record_augmented=False,
        dtype=torch.float64,
    )
    elapsed = time.perf_counter() - start
    event = task.hard_event(sample.spot, sample.step_dt).to(torch.float64)
    likelihood = torch.exp(sample.log_likelihood)
    return _summary(
        event * likelihood,
        elapsed=elapsed,
        likelihood_parent_mean=likelihood,
        log_likelihood_probe=sample.log_likelihood,
        control_energy=sample.control_energy,
        method=method,
        fine_steps=steps,
    )


def _evaluate_adjacent(
    simulator: RBergomiSimulator,
    task: DownsideExcursionTask,
    control: TimePiecewiseTwoDriverControl,
    *,
    spot: float,
    maturity: float,
    fine_steps: int,
    paths: int,
    seed: int,
) -> dict[str, float | int | str]:
    torch.manual_seed(seed)
    start = time.perf_counter()
    sample = simulate_coupled_rbergomi_adjacent(
        simulator,
        S0=spot,
        T=maturity,
        fine_steps=fine_steps,
        num_paths=paths,
        control_fn=control,
    )
    elapsed = time.perf_counter() - start
    fine = task.hard_event(sample.fine.spot, sample.fine.step_dt).to(torch.float64)
    coarse = task.hard_event(sample.coarse.spot, sample.coarse.step_dt).to(torch.float64)
    likelihood = torch.exp(sample.log_likelihood)
    return _summary(
        (fine - coarse) * likelihood,
        elapsed=elapsed,
        likelihood_parent_mean=likelihood,
        log_likelihood_probe=sample.log_likelihood,
        control_energy=sample.control_energy,
        method="standard_adjacent_correction",
        fine_steps=fine_steps,
    )


def _evaluate_adaptive(
    simulator: RBergomiSimulator,
    task: DownsideExcursionTask,
    control: TimePiecewiseTwoDriverControl,
    fit: BoundaryVarianceFit,
    calibration: dict[str, Any],
    *,
    feature_points: int,
    spot: float,
    maturity: float,
    fine_steps: int,
    paths: int,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    start = time.perf_counter()
    trunks = sample_rbergomi_coarse_trunks(
        simulator,
        S0=spot,
        T=maturity,
        fine_steps=fine_steps,
        num_parents=paths,
        control=control,
    )
    features = coarse_variance_features(trunks, task, feature_points=feature_points)
    scores = boundary_variance_scores(fit, features)
    selected = calibration["selected"]
    counts = score_threshold_branch_counts(
        scores,
        threshold=float(selected["score_threshold"]),
        high_branches=int(selected["high_branches"]),
    )
    result = evaluate_variable_branched_correction(simulator, trunks, task, branch_counts=counts)
    elapsed = time.perf_counter() - start
    summary: dict[str, Any] = _summary(
        result.contributions,
        elapsed=elapsed,
        likelihood_parent_mean=result.likelihood_parent_means,
        log_likelihood_probe=result.first_branch_log_likelihood,
        control_energy=result.control_energy,
        method="adaptive_volterra_branching",
        fine_steps=fine_steps,
    )
    summary.update(
        {
            "mean_branches": result.mean_branches,
            "selected_fraction": result.selected_fraction,
            "branch_disagreement_fraction": result.branch_disagreement_fraction,
            "maximum_constraint_error": result.maximum_constraint_error,
            "high_branches": int(selected["high_branches"]),
            "score_threshold": float(selected["score_threshold"]),
        }
    )
    return summary


def _method_summary(levels: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "estimate": sum(float(level["estimate"]) for level in levels),
        "standard_error": math.sqrt(sum(float(level["standard_error"]) ** 2 for level in levels)),
        "online_work_coefficient": sum(
            math.sqrt(float(level["single_path_variance"]) * float(level["cost_per_parent"]))
            for level in levels
        )
        ** 2,
    }


def _aggregate_correction_difference_z(runs: list[dict[str, Any]]) -> float:
    count = len(runs)
    adaptive_mean = statistics.mean(
        sum(float(level["estimate"]) for level in run["adaptive_levels"][1:]) for run in runs
    )
    standard_mean = statistics.mean(
        sum(float(level["estimate"]) for level in run["standard_levels"][1:]) for run in runs
    )
    variance = sum(
        sum(float(level["standard_error"]) ** 2 for level in run["adaptive_levels"][1:])
        + sum(float(level["standard_error"]) ** 2 for level in run["standard_levels"][1:])
        for run in runs
    ) / (count * count)
    return (adaptive_mean - standard_mean) / math.sqrt(variance) if variance > 0.0 else 0.0


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


def run(
    config_path: Path,
    *,
    checkpoint_dir: Path,
    smoke: bool,
) -> dict[str, Any]:
    config, config_hash = _load(config_path)
    simulator = _simulator(config)
    task = _task(config)
    model = config["model"]
    spot = float(model["spot"])
    maturity = float(model["maturity"])
    base_steps = int(config["hierarchy"]["base_steps"])
    levels = [int(value) for value in config["hierarchy"]["correction_fine_steps"]]
    anchor = _control(config["event_cem_anchor"]["values"], maturity)
    controls = {
        level: _control(config["correction_controls"][str(level)], maturity) for level in levels
    }
    development = config["development"]
    development_paths = 3_000 if smoke else int(development["paths_per_level"])
    maximum_branches = 4 if smoke else int(development["maximum_branches"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    fits: dict[int, BoundaryVarianceFit] = {}
    calibrations: dict[int, dict[str, Any]] = {}
    checkpoint_records: dict[int, dict[str, str]] = {}
    development_start = time.perf_counter()
    for level, development_seed, allocator_seed in zip(
        levels,
        config["seeds"]["development"],
        config["seeds"]["allocator"],
        strict=True,
    ):
        torch.manual_seed(int(development_seed))
        trunks = sample_rbergomi_coarse_trunks(
            simulator,
            S0=spot,
            T=maturity,
            fine_steps=level,
            num_parents=development_paths,
            control=controls[level],
        )
        refined = refine_rbergomi_coarse_trunks(simulator, trunks, branches=maximum_branches)
        branch_values, _fine, _coarse = _branch_contributions(refined, task)
        fit, calibration = _fit_allocator(
            trunks,
            task,
            branch_values,
            config,
            seed=int(allocator_seed),
            smoke=smoke,
        )
        fits[level] = fit
        calibrations[level] = calibration
        payload = _checkpoint_payload(fit, calibration, int(development["feature_points"]))
        checkpoint_path = checkpoint_dir / f"g8_boundary_allocator_{level}.pt"
        torch.save(payload, checkpoint_path)
        checkpoint_records[level] = {
            "path": checkpoint_path.as_posix(),
            "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        }
    development_seconds = time.perf_counter() - development_start

    validation_paths = 3_000 if smoke else int(config["validation"]["paths_per_seed"])
    validation_seeds = [int(seed) for seed in config["seeds"]["validation"]]
    if smoke:
        validation_seeds = validation_seeds[:2]
    runs: list[dict[str, Any]] = []
    validation_start = time.perf_counter()
    for seed in validation_seeds:
        base = _evaluate_base_or_single(
            simulator,
            task,
            anchor,
            spot=spot,
            maturity=maturity,
            steps=base_steps,
            paths=validation_paths,
            seed=seed,
            method="base_piecewise_cem",
        )
        standard_levels: list[dict[str, Any]] = [base]
        adaptive_levels: list[dict[str, Any]] = [{**base, "method": "base_piecewise_cem_shared"}]
        correction_work_ratios: dict[int, float] = {}
        for level_index, level in enumerate(levels):
            standard = _evaluate_adjacent(
                simulator,
                task,
                controls[level],
                spot=spot,
                maturity=maturity,
                fine_steps=level,
                paths=validation_paths,
                seed=seed + 10_000 + level_index * 101,
            )
            adaptive = _evaluate_adaptive(
                simulator,
                task,
                controls[level],
                fits[level],
                calibrations[level],
                feature_points=int(development["feature_points"]),
                spot=spot,
                maturity=maturity,
                fine_steps=level,
                paths=validation_paths,
                seed=seed + 20_000 + level_index * 101,
            )
            standard_levels.append(standard)
            adaptive_levels.append(adaptive)
            standard_work = float(standard["single_path_variance"]) * float(
                standard["cost_per_parent"]
            )
            adaptive_work = float(adaptive["single_path_variance"]) * float(
                adaptive["cost_per_parent"]
            )
            correction_work_ratios[level] = standard_work / adaptive_work
        single = _evaluate_base_or_single(
            simulator,
            task,
            anchor,
            spot=spot,
            maturity=maturity,
            steps=levels[-1],
            paths=validation_paths,
            seed=seed + 30_000,
            method="single_level_piecewise_cem",
        )
        standard_summary = _method_summary(standard_levels)
        adaptive_summary = _method_summary(adaptive_levels)
        single_work_coefficient = float(single["single_path_variance"]) * float(
            single["cost_per_parent"]
        )
        total_work_ratio = single_work_coefficient / adaptive_summary["online_work_coefficient"]
        runs.append(
            {
                "seed": seed,
                "standard_levels": standard_levels,
                "adaptive_levels": adaptive_levels,
                "standard_mlmc": standard_summary,
                "adaptive_mlmc": adaptive_summary,
                "single_level": single,
                "correction_work_ratios": correction_work_ratios,
                "total_work_ratio": total_work_ratio,
            }
        )
    validation_seconds = time.perf_counter() - validation_start

    fixed_best_ratios = {
        level: max(
            float(value)
            for key, value in calibrations[level]["fixed_work_ratios"].items()
            if int(key) > 1
        )
        for level in levels
    }
    fixed_improved_levels = sum(value > 1.10 for value in fixed_best_ratios.values())
    correction_work_values = [
        float(run["correction_work_ratios"][level]) for run in runs for level in levels
    ]
    correction_geometric_ratio = math.exp(
        statistics.mean(math.log(value) for value in correction_work_values)
    )
    total_work_values = [float(run["total_work_ratio"]) for run in runs]
    total_geometric_ratio = math.exp(
        statistics.mean(math.log(value) for value in total_work_values)
    )
    improving_seeds = sum(value > 1.0 for value in total_work_values)
    consistency_z = _aggregate_correction_difference_z(runs)
    maximum_likelihood_z = max(
        float(level["likelihood_normalization_z"])
        for run in runs
        for key in ("standard_levels", "adaptive_levels")
        for level in run[key]
    )
    maximum_log_likelihood_moment_z = max(
        float(level["log_likelihood_moment_z"])
        for run in runs
        for key in ("standard_levels", "adaptive_levels")
        for level in run[key]
    )
    maximum_constraint_error = max(
        float(level.get("maximum_constraint_error", 0.0))
        for run in runs
        for level in run["adaptive_levels"]
    )
    reference_probability = statistics.mean(float(run["standard_mlmc"]["estimate"]) for run in runs)
    variance_budget = (
        float(config["validation"]["target_relative_error"]) * reference_probability
    ) ** 2
    baseline_works: list[float] = []
    candidate_works: list[float] = []
    allocations: list[dict[str, Any]] = []
    for run in runs:
        single = run["single_level"]
        baseline_count, baseline_work = single_level_online_work(
            float(single["single_path_variance"]),
            float(single["cost_per_parent"]),
            variance_budget=variance_budget,
        )
        adaptive_levels = run["adaptive_levels"]
        allocation = optimal_mlmc_sample_counts(
            [float(level["single_path_variance"]) for level in adaptive_levels],
            [float(level["cost_per_parent"]) for level in adaptive_levels],
            variance_budget=variance_budget,
        )
        baseline_works.append(baseline_work)
        candidate_works.append(allocation.predicted_online_work)
        allocations.append(
            {
                "seed": run["seed"],
                "baseline_sample_count": baseline_count,
                "baseline_online_seconds": baseline_work,
                "adaptive_mlmc": asdict(allocation),
            }
        )
    mean_baseline_work = statistics.mean(baseline_works)
    mean_candidate_work = statistics.mean(candidate_works)
    break_even = break_even_query_count(
        development_seconds, mean_baseline_work, mean_candidate_work
    )
    mean_adaptive_variance = [
        statistics.mean(
            float(run["adaptive_levels"][index + 1]["single_path_variance"]) for run in runs
        )
        for index in range(len(levels))
    ]
    h_values = [maturity / level for level in levels]
    variance_slope = _log_slope(h_values, mean_adaptive_variance)
    thresholds = config["validation"]
    likelihood_gate = str(thresholds.get("likelihood_gate", "raw_mean"))
    if likelihood_gate not in {"raw_mean", "gaussian_log_moments"}:
        raise ValueError("likelihood_gate must be raw_mean or gaussian_log_moments")
    likelihood_gate_value = (
        maximum_likelihood_z if likelihood_gate == "raw_mean" else maximum_log_likelihood_moment_z
    )
    likelihood_gate_threshold = float(
        thresholds[
            "maximum_likelihood_normalization_z"
            if likelihood_gate == "raw_mean"
            else "maximum_log_likelihood_moment_z"
        ]
    )
    gates = {
        "fixed_branch_feasibility": fixed_improved_levels
        >= int(thresholds["minimum_fixed_improved_levels"]),
        "correction_consistency": abs(consistency_z)
        <= float(thresholds["maximum_absolute_difference_z"]),
        "adaptive_correction_work": correction_geometric_ratio
        > float(thresholds["minimum_correction_work_ratio"]),
        "total_work": total_geometric_ratio > float(thresholds["minimum_total_work_ratio"]),
        "improving_seeds": improving_seeds >= int(thresholds["minimum_improving_seeds"]),
        "break_even": break_even <= float(thresholds["maximum_break_even_queries"]),
        "likelihood_normalization": likelihood_gate_value <= likelihood_gate_threshold,
        "conditional_constraint": maximum_constraint_error
        <= float(thresholds["maximum_constraint_error"]),
        "positive_variance_decay": variance_slope is not None and variance_slope > 0.0,
    }
    return {
        "protocol_id": config["protocol_id"],
        "protocol_sha256": config_hash,
        "smoke": smoke,
        "theory_contract": {
            "conditioning": "coarse Brownian and BLP singular-local innovations",
            "allocation_measurability": "coarse path only",
            "likelihood": "one exact fine-space likelihood per branch",
            "likelihood_gate": likelihood_gate,
            "self_normalized": False,
            "continuous_rate_claimed": False,
        },
        "development": {
            "seconds": development_seconds,
            "paths_per_level": development_paths,
            "maximum_branches": maximum_branches,
            "calibrations": calibrations,
            "fixed_best_work_ratios": fixed_best_ratios,
            "fixed_improved_levels": fixed_improved_levels,
            "checkpoints": checkpoint_records,
        },
        "validation": {
            "seconds": validation_seconds,
            "paths_per_seed": validation_paths,
            "runs": runs,
            "aggregate_correction_difference_z": consistency_z,
            "correction_geometric_work_ratio": correction_geometric_ratio,
            "total_geometric_work_ratio": total_geometric_ratio,
            "improving_total_work_seeds": improving_seeds,
            "maximum_likelihood_normalization_z": maximum_likelihood_z,
            "maximum_log_likelihood_moment_z": maximum_log_likelihood_moment_z,
            "maximum_conditional_constraint_error": maximum_constraint_error,
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
            "mean_adaptive_correction_variance": mean_adaptive_variance,
            "variance_log_slope": variance_slope,
            "interpretation": "empirical finite-level diagnostic, not a theorem",
        },
        "gates": gates,
        "passed": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/g8_volterra_bridge_branching.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("results/checkpoints/g8_volterra_branching"),
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    result = run(args.config, checkpoint_dir=args.checkpoint_dir, smoke=args.smoke)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({"passed": result["passed"], "gates": result["gates"]}, indent=2))


if __name__ == "__main__":
    main()
