"""G9 development experiment for monotone Gaussian Volterra smoothing."""

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

from src.evaluation.smoothed_multilevel import (
    PairedLevelDiagnostics,
    paired_level_diagnostics,
    paired_mlmc_diagnostics,
)
from src.path_integral.controllers import TimePiecewiseTwoDriverControl
from src.path_integral.path_functionals import DownsideExcursionTask
from src.path_integral.rbergomi_fft import (
    simulate_coupled_rbergomi_adjacent_fft,
    simulate_rbergomi_fft,
)
from src.path_integral.rbergomi_smoothing import (
    evaluate_smoothed_adjacent_rbergomi_sample,
    evaluate_smoothed_rbergomi_sample,
)
from src.physics_engine import RBergomiSimulator


def _load(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("expected a G9 development schema-version-1 config")
    if payload.get("frozen") is not False:
        raise ValueError("development config must explicitly declare frozen: false")
    levels = [int(value) for value in payload["hierarchy"]["fine_steps"]]
    if len(levels) < 3 or any(value < 2 or value % 2 != 0 for value in levels):
        raise ValueError("at least three positive even hierarchy levels are required")
    if any(right != 2 * left for left, right in zip(levels[:-1], levels[1:], strict=True)):
        raise ValueError("hierarchy must be dyadic")
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


def _task(config: dict[str, Any]) -> DownsideExcursionTask:
    task = config["task"]
    return DownsideExcursionTask(
        hit_barrier=float(task["hit_barrier"]),
        stress_level=float(task["stress_level"]),
        minimum_occupation=float(task["minimum_occupation"]),
        hit_scale=float(task["hit_scale"]),
        occupation_scale=float(task["occupation_scale"]),
    )


def _control(config: dict[str, Any]) -> TimePiecewiseTwoDriverControl:
    values = tuple(
        tuple(float(entry) for entry in row) for row in config["event_cem_anchor"]["values"]
    )
    return TimePiecewiseTwoDriverControl(
        cast(tuple[tuple[float, float], ...], values),
        maturity=float(config["model"]["maturity"]),
    )


def _likelihood_normalization_z(likelihood: torch.Tensor) -> float:
    standard_error = float(likelihood.std(unbiased=True)) / math.sqrt(likelihood.numel())
    difference = float(likelihood.mean()) - 1.0
    return abs(difference) / standard_error if standard_error > 0.0 else 0.0


def _evaluate_level(
    simulator: RBergomiSimulator,
    task: DownsideExcursionTask,
    control: TimePiecewiseTwoDriverControl,
    *,
    spot: float,
    maturity: float,
    fine_steps: int,
    level: int,
    paths: int,
    chunk_size: int,
    seed: int,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> tuple[
    PairedLevelDiagnostics,
    dict[str, float | int],
    torch.Tensor,
    torch.Tensor,
]:
    torch.manual_seed(seed)
    raw_values: list[torch.Tensor] = []
    smoothed_values: list[torch.Tensor] = []
    likelihood_values: list[torch.Tensor] = []
    simulation_seconds = 0.0
    raw_postprocess_seconds = 0.0
    smoothing_postprocess_seconds = 0.0
    maximum_likelihood_error = 0.0
    maximum_path_error = 0.0
    maximum_residual_projection = 0.0
    completed = 0
    while completed < paths:
        current = min(chunk_size, paths - completed)
        start = time.perf_counter()
        if level == 0:
            sample = simulate_rbergomi_fft(
                simulator,
                S0=spot,
                T=maturity,
                dt=maturity / fine_steps,
                num_paths=current,
                control_fn=control,
            )
        else:
            sample = simulate_coupled_rbergomi_adjacent_fft(
                simulator,
                S0=spot,
                T=maturity,
                fine_steps=fine_steps,
                num_paths=current,
                control_fn=control,
            )
        simulation_seconds += time.perf_counter() - start

        start = time.perf_counter()
        likelihood = torch.exp(sample.log_likelihood)
        if level == 0:
            raw = task.hard_event(sample.spot, sample.step_dt).to(torch.float64) * likelihood
        else:
            fine_event = task.hard_event(sample.fine.spot, sample.fine.step_dt)
            coarse_event = task.hard_event(sample.coarse.spot, sample.coarse.step_dt)
            raw = (fine_event.to(torch.float64) - coarse_event.to(torch.float64)) * likelihood
        raw_postprocess_seconds += time.perf_counter() - start

        start = time.perf_counter()
        if level == 0:
            smoothed = evaluate_smoothed_rbergomi_sample(
                sample,
                task=task,
                rho=simulator.rho,
                declared_deterministic_control=True,
            )
            smoothed_value = smoothed.level.smoothed_contribution
            raw_replay = smoothed.level.raw_contribution
            path_error = smoothed.maximum_path_reconstruction_error
        else:
            smoothed = evaluate_smoothed_adjacent_rbergomi_sample(
                sample,
                task=task,
                rho=simulator.rho,
                declared_deterministic_control=True,
            )
            smoothed_value = smoothed.smoothed_correction
            raw_replay = smoothed.raw_correction
            path_error = max(
                smoothed.maximum_fine_path_reconstruction_error,
                smoothed.maximum_coarse_path_reconstruction_error,
            )
        smoothing_postprocess_seconds += time.perf_counter() - start
        if not torch.equal(raw, raw_replay):
            raise AssertionError("raw contribution changed inside the smoothing evaluator")
        maximum_likelihood_error = max(
            maximum_likelihood_error,
            smoothed.maximum_likelihood_reconstruction_error,
        )
        maximum_path_error = max(maximum_path_error, path_error)
        maximum_residual_projection = max(
            maximum_residual_projection, smoothed.maximum_residual_projection
        )
        raw_values.append(raw.detach().cpu())
        smoothed_values.append(smoothed_value.detach().cpu())
        likelihood_values.append(likelihood.detach().cpu())
        completed += current

    raw_all = torch.cat(raw_values)
    smoothed_all = torch.cat(smoothed_values)
    likelihood_all = torch.cat(likelihood_values)
    raw_cost = (simulation_seconds + raw_postprocess_seconds) / paths
    # This is conservative: the audited smoothing evaluator also recomputes
    # the raw event solely for its exactness assertion.
    smoothed_cost = (simulation_seconds + smoothing_postprocess_seconds) / paths
    report = paired_level_diagnostics(
        raw_all,
        smoothed_all,
        raw_cost_per_path=raw_cost,
        smoothed_cost_per_path=smoothed_cost,
        confidence_level=confidence_level,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    audit: dict[str, float | int] = {
        "level": level,
        "fine_steps": fine_steps,
        "likelihood_normalization_z": _likelihood_normalization_z(likelihood_all),
        "maximum_likelihood_reconstruction_error": maximum_likelihood_error,
        "maximum_path_reconstruction_error": maximum_path_error,
        "maximum_residual_projection": maximum_residual_projection,
        "simulation_seconds": simulation_seconds,
        "raw_postprocess_seconds": raw_postprocess_seconds,
        "smoothing_postprocess_seconds": smoothing_postprocess_seconds,
    }
    return report, audit, raw_all, smoothed_all


def _log_slope(h: list[float], variance: list[float]) -> float:
    x = [math.log(value) for value in h]
    y = [math.log(value) for value in variance]
    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)
    denominator = sum((value - mean_x) ** 2 for value in x)
    return (
        sum((left - mean_x) * (right - mean_y) for left, right in zip(x, y, strict=True))
        / denominator
    )


def _one_sided_slope_lower_95(slopes: list[float]) -> float | None:
    if len(slopes) < 3:
        return None
    critical = {2: 2.920, 3: 2.353, 4: 2.132}.get(len(slopes) - 1, 1.645)
    return statistics.mean(slopes) - critical * statistics.stdev(slopes) / math.sqrt(len(slopes))


def run(config_path: Path, *, smoke: bool) -> dict[str, Any]:
    config, digest = _load(config_path)
    simulator = _simulator(config)
    task = _task(config)
    control = _control(config)
    levels = [int(value) for value in config["hierarchy"]["fine_steps"]]
    evaluation = config["evaluation"]
    paths = 2_000 if smoke else int(evaluation["paths_per_seed"])
    chunk_size = min(paths, 500 if smoke else int(evaluation["chunk_size"]))
    bootstrap_replicates = 100 if smoke else int(evaluation["bootstrap_replicates"])
    confidence_level = float(evaluation["confidence_level"])
    seeds = [int(value) for value in config["seeds"]["validation"]]
    if smoke:
        seeds = seeds[:2]
    torch.set_num_threads(int(evaluation["thread_count"]))
    model = config["model"]
    runs: list[dict[str, Any]] = []
    pooled_raw: list[list[torch.Tensor]] = [[] for _ in levels]
    pooled_smoothed: list[list[torch.Tensor]] = [[] for _ in levels]
    for seed_index, seed in enumerate(seeds):
        reports: list[PairedLevelDiagnostics] = []
        audits: list[dict[str, float | int]] = []
        for level, fine_steps in enumerate(levels):
            report, audit, raw_values, smoothed_values = _evaluate_level(
                simulator,
                task,
                control,
                spot=float(model["spot"]),
                maturity=float(model["maturity"]),
                fine_steps=fine_steps,
                level=level,
                paths=paths,
                chunk_size=chunk_size,
                seed=seed + 101 * level,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=int(config["seeds"]["bootstrap_base"]) + 1_000 * seed_index + level,
                confidence_level=confidence_level,
            )
            reports.append(report)
            audits.append(audit)
            pooled_raw[level].append(raw_values)
            pooled_smoothed[level].append(smoothed_values)
        combined = paired_mlmc_diagnostics(reports)
        corrections = reports[1:]
        raw_correction_coefficient = (
            sum(math.sqrt(report.raw_variance * report.raw_cost_per_path) for report in corrections)
            ** 2
        )
        smoothed_correction_coefficient = (
            sum(
                math.sqrt(report.smoothed_variance * report.smoothed_cost_per_path)
                for report in corrections
            )
            ** 2
        )
        correction_ratio = raw_correction_coefficient / smoothed_correction_coefficient
        slope = _log_slope(
            [float(model["maturity"]) / levels[index] for index in range(1, len(levels))],
            [report.smoothed_variance for report in corrections],
        )
        runs.append(
            {
                "seed": seed,
                "levels": [asdict(report) for report in reports],
                "audits": audits,
                "mlmc": asdict(combined),
                "correction_raw_over_smoothed_work_ratio": correction_ratio,
                "smoothed_correction_variance_slope": slope,
            }
        )

    pooled_reports = [
        paired_level_diagnostics(
            torch.cat(pooled_raw[level]),
            torch.cat(pooled_smoothed[level]),
            raw_cost_per_path=statistics.mean(
                float(run["levels"][level]["raw_cost_per_path"]) for run in runs
            ),
            smoothed_cost_per_path=statistics.mean(
                float(run["levels"][level]["smoothed_cost_per_path"]) for run in runs
            ),
            confidence_level=confidence_level,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=int(config["seeds"]["bootstrap_base"]) + 100_000 + level,
        )
        for level in range(len(levels))
    ]
    work_ratios = [float(run["correction_raw_over_smoothed_work_ratio"]) for run in runs]
    slopes = [float(run["smoothed_correction_variance_slope"]) for run in runs]
    geometric_ratio = math.exp(statistics.mean(math.log(value) for value in work_ratios))
    slope_lower = _one_sided_slope_lower_95(slopes)
    all_levels = [level for run in runs for level in run["levels"]]
    all_audits = [audit for run in runs for audit in run["audits"]]
    gates_config = config["gates"]
    gates = {
        "mean_consistency": max(
            abs(float(level["paired_mean_difference_z"])) for level in all_levels
        )
        <= float(gates_config["maximum_absolute_mean_difference_z"]),
        "rao_blackwell_upper_bound": max(
            report.variance_ratio_ci_upper for report in pooled_reports
        )
        <= float(gates_config["maximum_smoothed_over_raw_variance_ci_upper"]),
        "correction_work_ratio": geometric_ratio
        > float(gates_config["minimum_geometric_correction_work_ratio"]),
        "improving_seeds": statistics.mean(value > 1.0 for value in work_ratios)
        >= float(gates_config["minimum_improving_seed_fraction"]),
        "variance_slope": statistics.mean(slopes)
        > float(gates_config["minimum_smoothed_variance_slope"]),
        "variance_slope_lower_95": slope_lower is not None
        and slope_lower > float(gates_config["minimum_smoothed_variance_slope_lower_95"]),
        "deepest_kurtosis": (
            abs(pooled_reports[-1].smoothed_excess_kurtosis)
            < abs(pooled_reports[-1].raw_excess_kurtosis)
            if bool(gates_config["require_deepest_kurtosis_reduction"])
            else True
        ),
        "likelihood_normalization": max(
            float(audit["likelihood_normalization_z"]) for audit in all_audits
        )
        <= float(gates_config["maximum_likelihood_normalization_z"]),
        "exactness": max(
            max(
                float(audit["maximum_likelihood_reconstruction_error"]),
                float(audit["maximum_path_reconstruction_error"]),
                float(audit["maximum_residual_projection"]),
            )
            for audit in all_audits
        )
        <= float(gates_config["maximum_exactness_error"]),
    }
    return {
        "protocol_id": config["protocol_id"],
        "protocol_sha256": digest,
        "smoke": smoke,
        "theory_contract": {
            "target": "finite-grid hit-and-occupation event",
            "conditioning_axis": "positive direction in independent W2 price driver",
            "proposal_control": "deterministic time-only",
            "self_normalized": False,
            "continuous_time_claimed": False,
        },
        "evaluation": {
            "paths_per_seed": paths,
            "chunk_size": chunk_size,
            "bootstrap_replicates": bootstrap_replicates,
            "runs": runs,
            "pooled_levels": [asdict(report) for report in pooled_reports],
            "geometric_correction_work_ratio": geometric_ratio,
            "improving_seed_fraction": statistics.mean(value > 1.0 for value in work_ratios),
            "mean_smoothed_correction_variance_slope": statistics.mean(slopes),
            "slope_lower_95_one_sided": slope_lower,
            "deepest_raw_excess_kurtosis": [
                run["levels"][-1]["raw_excess_kurtosis"] for run in runs
            ],
            "deepest_smoothed_excess_kurtosis": [
                run["levels"][-1]["smoothed_excess_kurtosis"] for run in runs
            ],
        },
        "gates": gates,
        "passed": all(gates.values()),
        "interpretation": (
            "development evidence only; frozen multi-regime validation is required before a paper claim"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/g9_mgvs_development.yaml"))
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
