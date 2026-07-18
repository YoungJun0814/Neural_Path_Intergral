"""Development benchmark for exact adjacent DCS-MGI corrections."""

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

from experiments.g10_control_span_development import (
    _canonical_json_sha256,
    _controls,
    _geometric_mean,
    _normalization_z,
    _simulator,
    _task,
)
from src.path_integral import (
    evaluate_control_span_marginalized_adjacent_mixture,
    simulate_coupled_rbergomi_mixture,
)


def _load(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("expected a G10 correction schema-version-1 config")
    if payload.get("development_only") is not True:
        raise ValueError("correction selection must remain development-only")
    return payload, hashlib.sha256(raw).hexdigest()


def _verified(source: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(Path(source["path"]).read_text(encoding="utf-8"))
    if _canonical_json_sha256(payload) != str(source["canonical_json_sha256"]):
        raise ValueError(f"source canonical JSON hash mismatch: {source['path']}")
    if payload.get("smoke") is not False or payload.get("passed") is not True:
        raise ValueError(f"source must be a passed non-smoke artifact: {source['path']}")
    return payload


def _log_slope(step_sizes: list[float], variances: list[float]) -> float:
    if any(value <= 0.0 or not math.isfinite(value) for value in variances):
        raise ValueError("variance slope requires positive finite variances")
    x = [math.log(value) for value in step_sizes]
    y = [math.log(value) for value in variances]
    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)
    denominator = sum((value - mean_x) ** 2 for value in x)
    return sum(
        (left - mean_x) * (right - mean_y)
        for left, right in zip(x, y, strict=True)
    ) / denominator


def _regime(
    regime: dict[str, Any],
    *,
    alpha: float,
    levels: list[int],
    paths: int,
    seeds: list[int],
    label_seed_base: int,
    gates: dict[str, Any],
) -> dict[str, Any]:
    simulator = _simulator(regime["model"])
    task = _task(regime["task"])
    controls = _controls(regime)
    weights = torch.tensor([alpha, 1.0 - alpha], dtype=torch.float64)
    pooled_raw: list[list[torch.Tensor]] = [[] for _ in levels]
    pooled_marginalized: list[list[torch.Tensor]] = [[] for _ in levels]
    pooled_outer: list[list[torch.Tensor]] = [[] for _ in levels]
    runs: list[dict[str, Any]] = []
    maximum_exactness = 0.0
    maximum_bound_violation = 0.0
    for seed_index, seed in enumerate(seeds):
        level_reports: list[dict[str, float | int]] = []
        for level_index, fine_steps in enumerate(levels):
            torch.manual_seed(seed + 101 * level_index)
            start = time.perf_counter()
            sample = simulate_coupled_rbergomi_mixture(
                simulator,
                controls,
                weights,
                spot=float(regime["model"]["spot"]),
                maturity=float(regime["model"]["maturity"]),
                fine_steps=fine_steps,
                num_paths=paths,
                label_generator=torch.Generator().manual_seed(
                    label_seed_base + 10_000 * seed_index + level_index
                ),
                engine="fft",
            )
            simulation_seconds = time.perf_counter() - start
            start = time.perf_counter()
            fine_event = task.hard_event(sample.paths.fine.spot, sample.paths.fine.step_dt)
            coarse_event = task.hard_event(
                sample.paths.coarse.spot,
                sample.paths.coarse.step_dt,
            )
            raw = (
                fine_event.to(torch.float64) - coarse_event.to(torch.float64)
            ) * torch.exp(sample.mixture_log_likelihood)
            raw_postprocess_seconds = time.perf_counter() - start
            start = time.perf_counter()
            estimate = evaluate_control_span_marginalized_adjacent_mixture(
                sample,
                task=task,
                rho=simulator.rho,
            )
            marginalized_postprocess_seconds = time.perf_counter() - start
            if not torch.equal(raw, estimate.raw_correction):
                raise AssertionError("raw adjacent defensive-mixture replay failed")
            raw_variance = float(raw.var(unbiased=True))
            marginalized_variance = float(estimate.marginalized_correction.var(unbiased=True))
            raw_cost = (simulation_seconds + raw_postprocess_seconds) / paths
            marginalized_cost = (simulation_seconds + marginalized_postprocess_seconds) / paths
            work_ratio = raw_variance * raw_cost / (marginalized_variance * marginalized_cost)
            exactness = max(
                estimate.maximum_component_log_density_error,
                estimate.maximum_mixture_log_density_error,
                estimate.maximum_coordinate_mismatch,
                estimate.maximum_fine_path_reconstruction_error,
                estimate.maximum_coarse_path_reconstruction_error,
                estimate.maximum_residual_projection,
                estimate.span.maximum_span_residual,
            )
            maximum_exactness = max(maximum_exactness, exactness)
            maximum_bound_violation = max(
                maximum_bound_violation,
                estimate.maximum_defensive_bound_violation,
            )
            pooled_raw[level_index].append(raw.detach().cpu())
            pooled_marginalized[level_index].append(
                estimate.marginalized_correction.detach().cpu()
            )
            pooled_outer[level_index].append(estimate.outer_likelihood.detach().cpu())
            level_reports.append(
                {
                    "fine_steps": fine_steps,
                    "simulation_seconds": simulation_seconds,
                    "raw_postprocess_seconds": raw_postprocess_seconds,
                    "marginalized_postprocess_seconds": marginalized_postprocess_seconds,
                    "raw_variance": raw_variance,
                    "marginalized_variance": marginalized_variance,
                    "raw_over_marginalized_work_ratio": work_ratio,
                    "maximum_exactness_error": exactness,
                }
            )
        runs.append({"seed": seed, "levels": level_reports})

    pooled_levels: list[dict[str, float | int]] = []
    for level_index, fine_steps in enumerate(levels):
        raw = torch.cat(pooled_raw[level_index])
        marginalized = torch.cat(pooled_marginalized[level_index])
        outer = torch.cat(pooled_outer[level_index])
        difference = marginalized - raw
        paired_se = math.sqrt(float(difference.var(unbiased=True)) / difference.numel())
        raw_cost = statistics.mean(
            (
                float(run["levels"][level_index]["simulation_seconds"])
                + float(run["levels"][level_index]["raw_postprocess_seconds"])
            )
            / paths
            for run in runs
        )
        marginalized_cost = statistics.mean(
            (
                float(run["levels"][level_index]["simulation_seconds"])
                + float(run["levels"][level_index]["marginalized_postprocess_seconds"])
            )
            / paths
            for run in runs
        )
        raw_variance = float(raw.var(unbiased=True))
        marginalized_variance = float(marginalized.var(unbiased=True))
        pooled_levels.append(
            {
                "fine_steps": fine_steps,
                "paths": int(raw.numel()),
                "raw_mean": float(raw.mean()),
                "marginalized_mean": float(marginalized.mean()),
                "paired_mean_difference_z": (
                    float(difference.mean()) / paired_se if paired_se > 0.0 else 0.0
                ),
                "raw_variance": raw_variance,
                "marginalized_variance": marginalized_variance,
                "marginalized_over_raw_variance": marginalized_variance / raw_variance,
                "raw_cost_per_path": raw_cost,
                "marginalized_cost_per_path": marginalized_cost,
                "raw_over_marginalized_work_ratio": (
                    raw_variance * raw_cost / (marginalized_variance * marginalized_cost)
                ),
                "outer_likelihood_normalization_z": _normalization_z(outer),
                "maximum_outer_likelihood": float(torch.max(outer)),
            }
        )
    step_sizes = [float(regime["model"]["maturity"]) / value for value in levels]
    raw_slope = _log_slope(
        step_sizes,
        [float(level["raw_variance"]) for level in pooled_levels],
    )
    marginalized_slope = _log_slope(
        step_sizes,
        [float(level["marginalized_variance"]) for level in pooled_levels],
    )
    geometric_work = _geometric_mean(
        [
            float(level["raw_over_marginalized_work_ratio"])
            for run in runs
            for level in run["levels"]
        ]
    )
    regime_gates = {
        "exactness": maximum_exactness <= float(gates["maximum_exactness_error"]),
        "defensive_bound": maximum_bound_violation
        <= float(gates["maximum_defensive_bound_violation"]),
        "mean_consistency": max(
            abs(float(level["paired_mean_difference_z"])) for level in pooled_levels
        )
        <= float(gates["maximum_paired_mean_difference_z"]),
        "outer_likelihood_normalization": max(
            float(level["outer_likelihood_normalization_z"]) for level in pooled_levels
        )
        <= float(gates["maximum_outer_likelihood_normalization_z"]),
        "correction_work": geometric_work
        > float(gates["minimum_geometric_correction_work_ratio"]),
        "positive_marginalized_slope": marginalized_slope
        > float(gates["minimum_smoothed_variance_slope"]),
    }
    return {
        "regime_id": regime["regime_id"],
        "natural_weight": alpha,
        "runs": runs,
        "pooled_levels": pooled_levels,
        "raw_variance_slope": raw_slope,
        "marginalized_variance_slope": marginalized_slope,
        "geometric_raw_over_marginalized_correction_work_ratio": geometric_work,
        "maximum_exactness_error": maximum_exactness,
        "maximum_defensive_bound_violation": maximum_bound_violation,
        "gates": regime_gates,
        "passed": all(regime_gates.values()),
    }


def run(config_path: Path, *, smoke: bool) -> dict[str, Any]:
    config, digest = _load(config_path)
    selection = _verified(config["selection_source"])
    calibration = _verified(config["calibration_source"])
    selected = {
        str(regime["regime_id"]): float(regime["selected_natural_weight"])
        for regime in selection["regimes"]
    }
    calibrated = {str(regime["regime_id"]): regime for regime in calibration["regimes"]}
    regime_ids = list(selected)
    if smoke:
        regime_ids = regime_ids[:1]
    levels = [int(value) for value in config["hierarchy"]["fine_steps"]]
    if smoke:
        levels = levels[:2]
    evaluation = config["evaluation"]
    paths = 500 if smoke else int(evaluation["paths_per_seed"])
    seeds = [int(value) for value in evaluation["validation_seeds"]]
    if smoke:
        seeds = seeds[:2]
    torch.set_num_threads(int(evaluation["thread_count"]))
    outputs = [
        _regime(
            calibrated[regime_id],
            alpha=selected[regime_id],
            levels=levels,
            paths=paths,
            seeds=[seed + 100_000 * regime_index for seed in seeds],
            label_seed_base=int(evaluation["label_seed_base"]) + 100_000 * regime_index,
            gates=config["gates"],
        )
        for regime_index, regime_id in enumerate(regime_ids)
    ]
    ratios = [
        float(output["geometric_raw_over_marginalized_correction_work_ratio"])
        for output in outputs
    ]
    aggregate_ratio = _geometric_mean(ratios)
    gates = {
        "all_exactness": all(bool(output["gates"]["exactness"]) for output in outputs),
        "all_mean_consistency": all(
            bool(output["gates"]["mean_consistency"]) for output in outputs
        ),
        "geometric_correction_work": aggregate_ratio
        > float(config["gates"]["minimum_geometric_correction_work_ratio"]),
    }
    return {
        "protocol_id": config["protocol_id"],
        "protocol_sha256": digest,
        "development_only": True,
        "smoke": smoke,
        "source_hashes": {
            str(config["selection_source"]["path"]): str(
                config["selection_source"]["canonical_json_sha256"]
            ),
            str(config["calibration_source"]["path"]): str(
                config["calibration_source"]["canonical_json_sha256"]
            ),
        },
        "levels": levels,
        "paths_per_seed": paths,
        "seeds": seeds,
        "regimes": outputs,
        "aggregate": {
            "geometric_correction_work_ratio": aggregate_ratio,
            "passed_regimes": sum(bool(output["passed"]) for output in outputs),
            "total_regimes": len(outputs),
        },
        "gates": gates,
        "passed": all(gates.values()),
        "restriction": "development correction evidence; no confirmatory claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/g10_control_span_correction_development.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    result = run(arguments.config, smoke=arguments.smoke)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({"passed": result["passed"], **result["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
