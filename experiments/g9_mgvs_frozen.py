"""Frozen 12-core/6-stress end-to-end validation for MGVS."""

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

from experiments.g9_mgvs_development import _evaluate_level
from src.evaluation.smoothed_multilevel import (
    PairedLevelDiagnostics,
    paired_level_diagnostics,
    paired_mlmc_diagnostics,
)
from src.path_integral.controllers import TimePiecewiseTwoDriverControl
from src.path_integral.gaussian_smoothing import positive_exponential_direction
from src.path_integral.path_functionals import DownsideExcursionTask
from src.path_integral.rbergomi_fft import simulate_rbergomi_fft
from src.path_integral.rbergomi_smoothing import evaluate_smoothed_rbergomi_sample
from src.physics_engine import RBergomiSimulator


def _canonical_json_sha256(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(canonical).hexdigest()


def _load(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("expected a G9 frozen schema-version-1 config")
    if payload.get("frozen") is not True:
        raise ValueError("confirmatory protocol must be frozen")
    levels = [int(value) for value in payload["hierarchy"]["fine_steps"]]
    if len(levels) < 4 or any(
        right != 2 * left for left, right in zip(levels[:-1], levels[1:], strict=True)
    ):
        raise ValueError("frozen hierarchy must contain at least four dyadic levels")
    return payload, hashlib.sha256(raw).hexdigest()


def _verified_json(source: dict[str, Any]) -> dict[str, Any]:
    path = Path(source["path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    if _canonical_json_sha256(payload) != str(source["canonical_json_sha256"]):
        raise ValueError(f"source canonical JSON hash mismatch: {path}")
    if payload.get("smoke") is not False or payload.get("passed") is not True:
        raise ValueError(f"source is not a passed non-smoke calibration: {path}")
    return cast(dict[str, Any], payload)


def _task(values: dict[str, Any]) -> DownsideExcursionTask:
    return DownsideExcursionTask(
        hit_barrier=float(values["hit_barrier"]),
        stress_level=float(values["stress_level"]),
        minimum_occupation=float(values["minimum_occupation"]),
        hit_scale=float(values["hit_scale"]),
        occupation_scale=float(values["occupation_scale"]),
    )


def _simulator(values: dict[str, Any]) -> RBergomiSimulator:
    return RBergomiSimulator(
        H=float(values["H"]),
        eta=float(values["eta"]),
        xi=float(values["xi"]),
        rho=float(values["rho"]),
        device="cpu",
    )


def _control(regime: dict[str, Any]) -> TimePiecewiseTwoDriverControl:
    values = tuple(tuple(float(entry) for entry in row) for row in regime["control"])
    return TimePiecewiseTwoDriverControl(
        cast(tuple[tuple[float, float], ...], values),
        maturity=float(regime["model"]["maturity"]),
    )


def _reference(
    regime: dict[str, Any],
    *,
    paths: int,
    chunk_size: int,
    steps: int,
    seed: int,
    direction_decay: float,
) -> dict[str, float]:
    simulator = _simulator(regime["model"])
    task = _task(regime["task"])
    control = _control(regime)
    torch.manual_seed(seed)
    values: list[torch.Tensor] = []
    completed = 0
    start = time.perf_counter()
    while completed < paths:
        current = min(chunk_size, paths - completed)
        sample = simulate_rbergomi_fft(
            simulator,
            S0=float(regime["model"]["spot"]),
            T=float(regime["model"]["maturity"]),
            dt=float(regime["model"]["maturity"]) / steps,
            num_paths=current,
            control_fn=control,
        )
        smoothed = evaluate_smoothed_rbergomi_sample(
            sample,
            task=task,
            rho=simulator.rho,
            direction=positive_exponential_direction(
                steps,
                decay=direction_decay,
                device="cpu",
                dtype=torch.float64,
            ),
            declared_deterministic_control=True,
        )
        values.append(smoothed.level.smoothed_contribution.cpu())
        completed += current
    elapsed = time.perf_counter() - start
    contribution = torch.cat(values)
    variance = float(contribution.var(unbiased=True))
    return {
        "estimate": float(contribution.mean()),
        "standard_error": math.sqrt(variance / paths),
        "single_path_variance": variance,
        "paths": paths,
        "seconds": elapsed,
        "seed": seed,
    }


def _single_level_baseline(
    regime: dict[str, Any],
    *,
    paths: int,
    steps: int,
    seed: int,
    direction_decay: float,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    simulator = _simulator(regime["model"])
    task = _task(regime["task"])
    control = _control(regime)
    torch.manual_seed(seed)
    start = time.perf_counter()
    sample = simulate_rbergomi_fft(
        simulator,
        S0=float(regime["model"]["spot"]),
        T=float(regime["model"]["maturity"]),
        dt=float(regime["model"]["maturity"]) / steps,
        num_paths=paths,
        control_fn=control,
    )
    simulation_seconds = time.perf_counter() - start
    start = time.perf_counter()
    raw = task.hard_event(sample.spot, sample.step_dt).to(torch.float64) * torch.exp(
        sample.log_likelihood
    )
    raw_postprocess_seconds = time.perf_counter() - start
    start = time.perf_counter()
    smoothed_result = evaluate_smoothed_rbergomi_sample(
        sample,
        task=task,
        rho=simulator.rho,
        direction=positive_exponential_direction(
            steps,
            decay=direction_decay,
            device="cpu",
            dtype=torch.float64,
        ),
        declared_deterministic_control=True,
    )
    smoothing_postprocess_seconds = time.perf_counter() - start
    smoothed = smoothed_result.level.smoothed_contribution
    if not torch.equal(raw, smoothed_result.level.raw_contribution):
        raise AssertionError("single-level raw contribution replay failed")
    raw_variance = float(raw.var(unbiased=True))
    smoothed_variance = float(smoothed.var(unbiased=True))
    raw_cost = (simulation_seconds + raw_postprocess_seconds) / paths
    smoothed_cost = (simulation_seconds + smoothing_postprocess_seconds) / paths
    difference = smoothed - raw
    paired_se = math.sqrt(float(difference.var(unbiased=True)) / paths)
    likelihood = torch.exp(sample.log_likelihood)
    likelihood_se = float(likelihood.std(unbiased=True)) / math.sqrt(paths)
    likelihood_z = (
        abs(float(likelihood.mean()) - 1.0) / likelihood_se if likelihood_se > 0.0 else 0.0
    )
    summary = {
        "raw_estimate": float(raw.mean()),
        "smoothed_estimate": float(smoothed.mean()),
        "raw_standard_error": math.sqrt(raw_variance / paths),
        "smoothed_standard_error": math.sqrt(smoothed_variance / paths),
        "raw_single_path_variance": raw_variance,
        "smoothed_single_path_variance": smoothed_variance,
        "raw_cost_per_path": raw_cost,
        "smoothed_cost_per_path": smoothed_cost,
        "raw_work_coefficient": raw_variance * raw_cost,
        "smoothed_work_coefficient": smoothed_variance * smoothed_cost,
        "raw_over_smoothed_work_ratio": (
            raw_variance * raw_cost / (smoothed_variance * smoothed_cost)
        ),
        "paired_mean_difference_z": (
            float(difference.mean()) / paired_se if paired_se > 0.0 else 0.0
        ),
        "likelihood_normalization_z": likelihood_z,
        "maximum_exactness_error": max(
            smoothed_result.maximum_likelihood_reconstruction_error,
            smoothed_result.maximum_path_reconstruction_error,
            smoothed_result.maximum_residual_projection,
        ),
    }
    return summary, raw.detach().cpu(), smoothed.detach().cpu()


def _geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("geometric mean requires positive finite values")
    return math.exp(statistics.mean(math.log(value) for value in values))


def _log_ratio_lower_95(values: list[float]) -> float:
    logs = [math.log(value) for value in values]
    if len(logs) < 2:
        raise ValueError("at least two ratios are required")
    critical = {9: 1.833, 11: 1.796}.get(len(logs) - 1, 1.645)
    lower_log = statistics.mean(logs) - critical * statistics.stdev(logs) / math.sqrt(len(logs))
    return math.exp(lower_log)


def _evaluate_regime(
    regime: dict[str, Any],
    *,
    group: str,
    direction_selection: dict[str, Any],
    levels: list[int],
    reference: dict[str, float],
    validation_seeds: list[int],
    paths: int,
    chunk_size: int,
    seed_bootstrap_replicates: int,
    pooled_bootstrap_replicates: int,
    bootstrap_base: int,
    confidence_level: float,
    gates_config: dict[str, Any],
) -> dict[str, Any]:
    simulator = _simulator(regime["model"])
    task = _task(regime["task"])
    control = _control(regime)
    runs: list[dict[str, Any]] = []
    pooled_raw: list[list[torch.Tensor]] = [[] for _ in levels]
    pooled_smoothed: list[list[torch.Tensor]] = [[] for _ in levels]
    pooled_single_raw: list[torch.Tensor] = []
    pooled_single_smoothed: list[torch.Tensor] = []
    for seed_index, seed in enumerate(validation_seeds):
        reports: list[PairedLevelDiagnostics] = []
        audits: list[dict[str, float | int]] = []
        for level, fine_steps in enumerate(levels):
            decay = float(
                direction_selection["selections"][f"level_{level}_{fine_steps}"]["selected_decay"]
            )
            report, audit, raw, smoothed = _evaluate_level(
                simulator,
                task,
                control,
                spot=float(regime["model"]["spot"]),
                maturity=float(regime["model"]["maturity"]),
                fine_steps=fine_steps,
                level=level,
                paths=paths,
                chunk_size=chunk_size,
                seed=seed + 101 * level,
                bootstrap_replicates=seed_bootstrap_replicates,
                bootstrap_seed=bootstrap_base + 10_000 * seed_index + level,
                confidence_level=confidence_level,
                direction=positive_exponential_direction(
                    fine_steps,
                    decay=decay,
                    device="cpu",
                    dtype=torch.float64,
                ),
            )
            reports.append(report)
            audits.append(audit)
            pooled_raw[level].append(raw)
            pooled_smoothed[level].append(smoothed)
        mlmc = paired_mlmc_diagnostics(reports)
        single, single_raw, single_smoothed = _single_level_baseline(
            regime,
            paths=paths,
            steps=levels[-1],
            seed=seed + 900_000,
            direction_decay=float(
                direction_selection["selections"][f"single_{levels[-1]}"]["selected_decay"]
            ),
        )
        pooled_single_raw.append(single_raw)
        pooled_single_smoothed.append(single_smoothed)
        primary_ratio = float(single["raw_over_smoothed_work_ratio"])
        raw_mlmc_ratio = mlmc.raw_work_coefficient / mlmc.smoothed_work_coefficient
        difference = float(single["smoothed_estimate"]) - float(reference["estimate"])
        comparison_se = math.sqrt(
            float(single["smoothed_standard_error"]) ** 2 + float(reference["standard_error"]) ** 2
        )
        reference_z = difference / comparison_se if comparison_se > 0.0 else 0.0
        runs.append(
            {
                "seed": seed,
                "levels": [asdict(report) for report in reports],
                "audits": audits,
                "mlmc": asdict(mlmc),
                "single_level": single,
                "raw_over_smoothed_single_work_ratio": primary_ratio,
                "raw_over_smoothed_mlmc_work_ratio": raw_mlmc_ratio,
                "reference_difference_z": reference_z,
                "reference_ci_covered": abs(reference_z) <= 1.96,
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
            bootstrap_replicates=pooled_bootstrap_replicates,
            bootstrap_seed=bootstrap_base + 800_000 + level,
        )
        for level in range(len(levels))
    ]
    pooled_mlmc = paired_mlmc_diagnostics(pooled_reports)
    pooled_single = paired_level_diagnostics(
        torch.cat(pooled_single_raw),
        torch.cat(pooled_single_smoothed),
        raw_cost_per_path=statistics.mean(
            float(run["single_level"]["raw_cost_per_path"]) for run in runs
        ),
        smoothed_cost_per_path=statistics.mean(
            float(run["single_level"]["smoothed_cost_per_path"]) for run in runs
        ),
        confidence_level=confidence_level,
        bootstrap_replicates=pooled_bootstrap_replicates,
        bootstrap_seed=bootstrap_base + 890_000,
    )
    total_ratios = [float(run["raw_over_smoothed_single_work_ratio"]) for run in runs]
    raw_ratios = [float(run["raw_over_smoothed_mlmc_work_ratio"]) for run in runs]
    pooled_reference_difference = statistics.mean(
        float(run["single_level"]["smoothed_estimate"]) for run in runs
    ) - float(reference["estimate"])
    pooled_reference_se = math.sqrt(
        statistics.mean(float(run["single_level"]["smoothed_standard_error"]) ** 2 for run in runs)
        / len(runs)
        + float(reference["standard_error"]) ** 2
    )
    maximum_likelihood_z = max(
        max(
            max(float(audit["likelihood_normalization_z"]) for audit in run["audits"]),
            float(run["single_level"]["likelihood_normalization_z"]),
        )
        for run in runs
    )
    maximum_exactness = max(
        max(
            max(
                float(audit["maximum_likelihood_reconstruction_error"]),
                float(audit["maximum_path_reconstruction_error"]),
                float(audit["maximum_residual_projection"]),
            )
            for audit in run["audits"]
        )
        for run in runs
    )
    maximum_exactness = max(
        maximum_exactness,
        max(float(run["single_level"]["maximum_exactness_error"]) for run in runs),
    )
    regime_gates = {
        "mean_consistency": max(
            max(abs(report.paired_mean_difference_z) for report in pooled_reports),
            abs(pooled_single.paired_mean_difference_z),
        )
        <= float(gates_config["maximum_pooled_mean_difference_z"]),
        "rao_blackwell": max(
            max(report.variance_ratio_ci_upper for report in pooled_reports),
            pooled_single.variance_ratio_ci_upper,
        )
        <= float(gates_config["maximum_smoothed_over_raw_variance_ci_upper"]),
        "correction_work": _geometric_mean(raw_ratios)
        > float(gates_config["minimum_correction_work_ratio"]),
        "total_work": _geometric_mean(total_ratios)
        > float(gates_config["minimum_total_work_ratio"]),
        "improving_seeds": sum(value > 1.0 for value in total_ratios)
        >= int(gates_config["minimum_improving_seeds_per_regime"]),
        "coverage": statistics.mean(bool(run["reference_ci_covered"]) for run in runs)
        >= float(gates_config["minimum_coverage_fraction"]),
        "reference_consistency": abs(pooled_reference_difference / pooled_reference_se)
        <= float(gates_config["maximum_reference_difference_z"]),
        "likelihood_normalization": maximum_likelihood_z
        <= float(gates_config["maximum_likelihood_normalization_z"]),
        "exactness": maximum_exactness <= float(gates_config["maximum_exactness_error"]),
    }
    return {
        "regime_id": regime["regime_id"],
        "group": group,
        "model": regime["model"],
        "task": regime["task"],
        "reference": reference,
        "runs": runs,
        "pooled_levels": [asdict(report) for report in pooled_reports],
        "pooled_mlmc": asdict(pooled_mlmc),
        "pooled_single_level": asdict(pooled_single),
        "geometric_raw_over_smoothed_single_work_ratio": _geometric_mean(total_ratios),
        "geometric_raw_over_smoothed_mlmc_work_ratio": _geometric_mean(raw_ratios),
        "total_work_ratio_lower_95_one_sided": _log_ratio_lower_95(total_ratios),
        "improving_total_work_seeds": sum(value > 1.0 for value in total_ratios),
        "coverage_fraction": statistics.mean(bool(run["reference_ci_covered"]) for run in runs),
        "pooled_reference_difference_z": pooled_reference_difference / pooled_reference_se,
        "maximum_likelihood_normalization_z": maximum_likelihood_z,
        "maximum_exactness_error": maximum_exactness,
        "gates": regime_gates,
        "passed": all(regime_gates.values()),
    }


def run(config_path: Path, *, smoke: bool) -> dict[str, Any]:
    config, digest = _load(config_path)
    regimes: list[tuple[str, dict[str, Any]]] = []
    calibration_hashes: dict[str, str] = {}
    calibration_seeds: set[int] = set()
    for source in config["calibration_sources"]:
        payload = _verified_json(source)
        calibration_hashes[str(source["path"])] = str(source["canonical_json_sha256"])
        for regime in payload["regimes"]:
            regimes.append((str(source["group"]), regime))
            calibration_seeds.add(int(regime["training_seed"]))
            calibration_seeds.add(int(regime["validation_seed"]))
    direction_payload = _verified_json(config["direction_source"])
    direction_by_regime = {str(value["regime_id"]): value for value in direction_payload["regimes"]}
    if set(direction_by_regime) != {str(regime["regime_id"]) for _group, regime in regimes}:
        raise ValueError("direction calibration regimes do not match CEM calibration regimes")
    if smoke:
        regimes = regimes[:1]
    validation_seeds = [int(value) for value in config["seeds"]["validation"]]
    if smoke:
        validation_seeds = validation_seeds[:2]
    reference_seeds = {
        int(config["reference"]["seed_base"]) + index for index in range(len(regimes))
    }
    if calibration_seeds & (set(validation_seeds) | reference_seeds):
        raise ValueError("calibration, reference, and validation seeds must be disjoint")
    levels = [int(value) for value in config["hierarchy"]["fine_steps"]]
    validation = config["validation"]
    paths = 500 if smoke else int(validation["paths_per_seed"])
    chunk_size = min(paths, int(validation["chunk_size"]))
    seed_bootstrap = 100 if smoke else int(validation["seed_level_bootstrap_replicates"])
    pooled_bootstrap = 100 if smoke else int(validation["pooled_bootstrap_replicates"])
    reference_paths = 2_000 if smoke else int(config["reference"]["paths_per_regime"])
    torch.set_num_threads(int(validation["thread_count"]))
    outputs: list[dict[str, Any]] = []
    for regime_index, (group, regime) in enumerate(regimes):
        reference = _reference(
            regime,
            paths=reference_paths,
            chunk_size=min(reference_paths, int(config["reference"]["chunk_size"])),
            steps=levels[-1],
            seed=int(config["reference"]["seed_base"]) + regime_index,
            direction_decay=float(
                direction_by_regime[str(regime["regime_id"])]["selections"][f"single_{levels[-1]}"][
                    "selected_decay"
                ]
            ),
        )
        outputs.append(
            _evaluate_regime(
                regime,
                group=group,
                direction_selection=direction_by_regime[str(regime["regime_id"])],
                levels=levels,
                reference=reference,
                validation_seeds=validation_seeds,
                paths=paths,
                chunk_size=chunk_size,
                seed_bootstrap_replicates=seed_bootstrap,
                pooled_bootstrap_replicates=pooled_bootstrap,
                bootstrap_base=int(config["seeds"]["bootstrap_base"]) + 1_000_000 * regime_index,
                confidence_level=float(validation["confidence_level"]),
                gates_config=config["gates"],
            )
        )

    core = [regime for regime in outputs if regime["group"] == "core"]
    stress = [regime for regime in outputs if regime["group"] == "stress"]
    total_ratios = [
        float(run["raw_over_smoothed_single_work_ratio"])
        for regime in core
        for run in regime["runs"]
    ]
    raw_ratios = [
        float(run["raw_over_smoothed_mlmc_work_ratio"]) for regime in core for run in regime["runs"]
    ]
    core_improved = sum(
        float(regime["geometric_raw_over_smoothed_single_work_ratio"]) > 1.0 for regime in core
    )
    # Every core regime reuses the same independent validation-seed labels.  The
    # suite-wide confidence interval must therefore cluster by seed rather than
    # treating the regime-by-seed cells as 120 independent replicates.
    suite_seed_ratios = [
        _geometric_mean(
            [
                float(regime["runs"][seed_index]["raw_over_smoothed_single_work_ratio"])
                for regime in core
            ]
        )
        for seed_index in range(len(validation_seeds))
    ]
    core_regime_ratios = [
        float(regime["geometric_raw_over_smoothed_single_work_ratio"]) for regime in core
    ]
    fixed_suite_lower_95 = _log_ratio_lower_95(suite_seed_ratios)
    regime_sensitivity_lower_95 = _log_ratio_lower_95(core_regime_ratios)
    gates_config = config["gates"]
    aggregate_gates = {
        "geometric_correction_work": _geometric_mean(raw_ratios)
        > float(gates_config["minimum_correction_work_ratio"]),
        "geometric_total_work": _geometric_mean(total_ratios)
        > float(gates_config["minimum_total_work_ratio"]),
        "total_work_lower_95": fixed_suite_lower_95
        > float(gates_config["minimum_total_work_ratio_lower_95"]),
        "core_regime_fraction": core_improved / len(core)
        >= float(gates_config["minimum_core_regime_improvement_fraction"]),
        "core_exactness": all(bool(regime["gates"]["exactness"]) for regime in core),
        "stress_exactness": all(bool(regime["gates"]["exactness"]) for regime in stress),
    }

    legacy: dict[str, Any] = {}
    for name, source in config["legacy_baselines"].items():
        path = Path(source["path"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        if _canonical_json_sha256(payload) != str(source["canonical_json_sha256"]):
            raise ValueError(f"legacy baseline canonical JSON hash mismatch: {path}")
        legacy[name] = {
            "path": str(path),
            "passed": payload["passed"],
            "gates": payload["gates"],
            "validation": {
                key: payload["validation"].get(key)
                for key in (
                    "geometric_work_ratio",
                    "total_geometric_work_ratio",
                    "correction_geometric_work_ratio",
                    "mean_candidate_online_seconds",
                )
            },
        }
    return {
        "protocol_id": config["protocol_id"],
        "protocol_sha256": digest,
        "smoke": smoke,
        "calibration_hashes": calibration_hashes,
        "direction_hash": str(config["direction_source"]["canonical_json_sha256"]),
        "theory_contract": {
            "target": "finite-grid hit-and-occupation probability",
            "control": "calibrated then frozen deterministic time-only CEM",
            "direction": "calibration-selected deterministic positive exponential direction",
            "self_normalized": False,
            "validation_seed_reuse": False,
            "continuous_time_claimed": False,
            "work_metric": "online variance-times-cost; direction calibration excluded",
            "aggregate_inference": "fixed core suite clustered by validation seed",
        },
        "reference_paths_per_regime": reference_paths,
        "validation_paths_per_seed": paths,
        "validation_seeds": validation_seeds,
        "regimes": outputs,
        "aggregate": {
            "core_regimes": len(core),
            "stress_regimes": len(stress),
            "geometric_raw_over_smoothed_mlmc_work_ratio": _geometric_mean(raw_ratios),
            "geometric_raw_over_smoothed_single_work_ratio": _geometric_mean(total_ratios),
            "total_work_ratio_lower_95_one_sided": fixed_suite_lower_95,
            "regime_heterogeneity_sensitivity_lower_95_one_sided": (
                regime_sensitivity_lower_95
            ),
            "superseded_unclustered_lower_95_one_sided": _log_ratio_lower_95(total_ratios),
            "improved_core_regime_fraction": core_improved / len(core),
            "passed_core_regimes": sum(bool(regime["passed"]) for regime in core),
            "passed_stress_regimes": sum(bool(regime["passed"]) for regime in stress),
        },
        "legacy_baselines": legacy,
        "gates": aggregate_gates,
        "passed": all(aggregate_gates.values()),
        "interpretation": (
            "fixed-suite inference clusters by validation seed; regime-heterogeneity sensitivity "
            "and individual failures remain visible"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/g9_mgvs_frozen.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    result = run(arguments.config, smoke=arguments.smoke)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "aggregate": result["aggregate"],
                "gates": result["gates"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
