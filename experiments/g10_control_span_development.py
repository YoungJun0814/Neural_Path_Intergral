"""Development-only benchmark for rank-one defensive control-span smoothing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, cast

import torch
import yaml

from src.path_integral import (
    DownsideExcursionTask,
    TimePiecewiseTwoDriverControl,
    evaluate_control_span_marginalized_mixture,
    simulate_rbergomi_mixture,
)
from src.physics_engine import RBergomiSimulator


def _canonical_json_sha256(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("expected a G10 development schema-version-1 config")
    if payload.get("development_only") is not True:
        raise ValueError("G10 selection config must be explicitly development-only")
    return payload, hashlib.sha256(raw).hexdigest()


def _source(source: dict[str, Any]) -> dict[str, Any]:
    path = Path(source["path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    if _canonical_json_sha256(payload) != str(source["canonical_json_sha256"]):
        raise ValueError("calibration source canonical hash mismatch")
    if payload.get("smoke") is not False or payload.get("passed") is not True:
        raise ValueError("calibration source must be a passed non-smoke artifact")
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


def _controls(
    regime: dict[str, Any],
) -> tuple[TimePiecewiseTwoDriverControl, TimePiecewiseTwoDriverControl]:
    maturity = float(regime["model"]["maturity"])
    cem_values = tuple(tuple(float(value) for value in row) for row in regime["control"])
    cem = TimePiecewiseTwoDriverControl(
        cast(tuple[tuple[float, float], ...], cem_values),
        maturity=maturity,
    )
    natural = TimePiecewiseTwoDriverControl(
        tuple((0.0, 0.0) for _ in cem_values),
        maturity=maturity,
    )
    return natural, cem


def _geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("geometric mean requires positive finite values")
    return math.exp(statistics.mean(math.log(value) for value in values))


def _normalization_z(values: torch.Tensor) -> float:
    standard_error = float(values.std(unbiased=True)) / math.sqrt(values.numel())
    return abs(float(values.mean()) - 1.0) / standard_error if standard_error > 0.0 else 0.0


def _candidate(
    regime: dict[str, Any],
    *,
    alpha: float,
    alpha_index: int,
    steps: int,
    paths: int,
    seeds: list[int],
    label_seed_base: int,
    gates: dict[str, Any],
) -> dict[str, Any]:
    simulator = _simulator(regime["model"])
    task = _task(regime["task"])
    controls = _controls(regime)
    weights = torch.tensor([alpha, 1.0 - alpha], dtype=torch.float64)
    raw_chunks: list[torch.Tensor] = []
    marginalized_chunks: list[torch.Tensor] = []
    outer_chunks: list[torch.Tensor] = []
    runs: list[dict[str, float | int]] = []
    maximum_exactness = 0.0
    maximum_bound_violation = 0.0
    for seed_index, seed in enumerate(seeds):
        torch.manual_seed(seed + 10_000 * alpha_index)
        start = time.perf_counter()
        sample = simulate_rbergomi_mixture(
            simulator,
            controls,
            weights,
            spot=float(regime["model"]["spot"]),
            maturity=float(regime["model"]["maturity"]),
            dt=float(regime["model"]["maturity"]) / steps,
            num_paths=paths,
            label_generator=torch.Generator().manual_seed(
                label_seed_base + 10_000 * alpha_index + seed_index
            ),
            engine="fft",
        )
        simulation_seconds = time.perf_counter() - start
        start = time.perf_counter()
        raw = task.hard_event(sample.paths.spot, sample.paths.step_dt).to(torch.float64) * torch.exp(
            sample.mixture_log_likelihood
        )
        raw_postprocess_seconds = time.perf_counter() - start
        start = time.perf_counter()
        estimate = evaluate_control_span_marginalized_mixture(
            sample,
            task=task,
            rho=simulator.rho,
        )
        marginalized_postprocess_seconds = time.perf_counter() - start
        if not torch.equal(raw, estimate.raw_mixture_contribution):
            raise AssertionError("raw defensive-mixture replay failed")
        raw_variance = float(raw.var(unbiased=True))
        marginalized_variance = float(estimate.marginalized_contribution.var(unbiased=True))
        raw_cost = (simulation_seconds + raw_postprocess_seconds) / paths
        marginalized_cost = (simulation_seconds + marginalized_postprocess_seconds) / paths
        work_ratio = raw_variance * raw_cost / (marginalized_variance * marginalized_cost)
        exactness = max(
            estimate.maximum_component_log_density_error,
            estimate.maximum_mixture_log_density_error,
            estimate.maximum_path_reconstruction_error,
            estimate.maximum_residual_projection,
            estimate.span.maximum_span_residual,
        )
        maximum_exactness = max(maximum_exactness, exactness)
        maximum_bound_violation = max(
            maximum_bound_violation,
            estimate.maximum_defensive_bound_violation,
        )
        runs.append(
            {
                "seed": seed,
                "simulation_seconds": simulation_seconds,
                "raw_postprocess_seconds": raw_postprocess_seconds,
                "marginalized_postprocess_seconds": marginalized_postprocess_seconds,
                "raw_variance": raw_variance,
                "marginalized_variance": marginalized_variance,
                "raw_over_marginalized_work_ratio": work_ratio,
                "maximum_outer_likelihood": float(torch.max(estimate.outer_likelihood)),
                "maximum_exactness_error": exactness,
            }
        )
        raw_chunks.append(raw.detach().cpu())
        marginalized_chunks.append(estimate.marginalized_contribution.detach().cpu())
        outer_chunks.append(estimate.outer_likelihood.detach().cpu())

    raw_all = torch.cat(raw_chunks)
    marginalized_all = torch.cat(marginalized_chunks)
    outer_all = torch.cat(outer_chunks)
    difference = marginalized_all - raw_all
    paired_se = math.sqrt(float(difference.var(unbiased=True)) / difference.numel())
    paired_z = float(difference.mean()) / paired_se if paired_se > 0.0 else 0.0
    geometric_work = _geometric_mean(
        [float(run["raw_over_marginalized_work_ratio"]) for run in runs]
    )
    outer_z = _normalization_z(outer_all)
    candidate_gates = {
        "exactness": maximum_exactness <= float(gates["maximum_exactness_error"]),
        "defensive_bound": maximum_bound_violation
        <= float(gates["maximum_defensive_bound_violation"]),
        "mean_consistency": abs(paired_z)
        <= float(gates["maximum_paired_mean_difference_z"]),
        "outer_likelihood_normalization": outer_z
        <= float(gates["maximum_outer_likelihood_normalization_z"]),
        "development_work": geometric_work > float(gates["minimum_geometric_work_ratio"]),
    }
    return {
        "natural_weight": alpha,
        "paths": int(raw_all.numel()),
        "raw_estimate": float(raw_all.mean()),
        "marginalized_estimate": float(marginalized_all.mean()),
        "paired_mean_difference_z": paired_z,
        "raw_variance": float(raw_all.var(unbiased=True)),
        "marginalized_variance": float(marginalized_all.var(unbiased=True)),
        "marginalized_over_raw_variance": float(
            marginalized_all.var(unbiased=True) / raw_all.var(unbiased=True)
        ),
        "geometric_raw_over_marginalized_work_ratio": geometric_work,
        "outer_likelihood_normalization_z": outer_z,
        "maximum_outer_likelihood": float(torch.max(outer_all)),
        "declared_outer_likelihood_bound": 1.0 / alpha,
        "maximum_exactness_error": maximum_exactness,
        "maximum_defensive_bound_violation": maximum_bound_violation,
        "runs": runs,
        "gates": candidate_gates,
        "admissible": all(
            value for key, value in candidate_gates.items() if key != "development_work"
        ),
        "passed": all(candidate_gates.values()),
    }


def run(config_path: Path, *, smoke: bool) -> dict[str, Any]:
    config, digest = _load(config_path)
    calibration = _source(config["calibration_source"])
    available = {str(regime["regime_id"]): regime for regime in calibration["regimes"]}
    regime_ids = [str(value) for value in config["regime_ids"]]
    if any(regime_id not in available for regime_id in regime_ids):
        raise ValueError("development regime is absent from the calibration source")
    if smoke:
        regime_ids = regime_ids[:1]
    candidates = [float(value) for value in config["mixture"]["natural_weight_candidates"]]
    if smoke:
        candidates = candidates[:2]
    if any(not 0.0 < value < 1.0 for value in candidates) or len(set(candidates)) != len(
        candidates
    ):
        raise ValueError("natural weights must be unique and lie strictly inside (0,1)")
    evaluation = config["evaluation"]
    steps = 64 if smoke else int(evaluation["fine_steps"])
    paths = 500 if smoke else int(evaluation["paths_per_seed"])
    seeds = [int(value) for value in evaluation["validation_seeds"]]
    if smoke:
        seeds = seeds[:2]
    torch.set_num_threads(int(evaluation["thread_count"]))
    outputs: list[dict[str, Any]] = []
    for regime_index, regime_id in enumerate(regime_ids):
        regime = available[regime_id]
        reports = [
            _candidate(
                regime,
                alpha=alpha,
                alpha_index=alpha_index,
                steps=steps,
                paths=paths,
                seeds=[seed + 100_000 * regime_index for seed in seeds],
                label_seed_base=int(evaluation["label_seed_base"]) + 100_000 * regime_index,
                gates=config["gates"],
            )
            for alpha_index, alpha in enumerate(candidates)
        ]
        admissible = [report for report in reports if report["admissible"]]
        selected = (
            max(
                admissible,
                key=lambda report: (
                    float(report["geometric_raw_over_marginalized_work_ratio"]),
                    -float(report["natural_weight"]),
                ),
            )
            if admissible
            else None
        )
        outputs.append(
            {
                "regime_id": regime_id,
                "candidates": reports,
                "selected_natural_weight": (
                    float(selected["natural_weight"]) if selected is not None else None
                ),
                "selected_work_ratio": (
                    float(selected["geometric_raw_over_marginalized_work_ratio"])
                    if selected is not None
                    else None
                ),
                "passed": bool(selected is not None and selected["passed"]),
            }
        )
    selected_ratios = [
        float(output["selected_work_ratio"])
        for output in outputs
        if output["selected_work_ratio"] is not None
    ]
    aggregate_ratio = _geometric_mean(selected_ratios) if selected_ratios else 0.0
    improved_fraction = (
        statistics.mean(ratio > 1.0 for ratio in selected_ratios)
        if len(selected_ratios) == len(outputs)
        else 0.0
    )
    gates = {
        "all_regimes_admissible": len(selected_ratios) == len(outputs),
        "geometric_development_work": aggregate_ratio
        > float(config["gates"]["minimum_geometric_work_ratio"]),
        "improved_regime_fraction": improved_fraction
        >= float(config["gates"]["minimum_improved_regime_fraction"]),
    }
    return {
        "protocol_id": config["protocol_id"],
        "protocol_sha256": digest,
        "development_only": True,
        "smoke": smoke,
        "calibration_hash": str(config["calibration_source"]["canonical_json_sha256"]),
        "steps": steps,
        "paths_per_seed": paths,
        "seeds": seeds,
        "regimes": outputs,
        "aggregate": {
            "geometric_selected_work_ratio": aggregate_ratio,
            "improved_regime_fraction": improved_fraction,
            "passed_regimes": sum(bool(output["passed"]) for output in outputs),
            "total_regimes": len(outputs),
        },
        "gates": gates,
        "passed": all(gates.values()),
        "restriction": "development evidence only; selected weights may not use future validation",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/g10_control_span_development.yaml"),
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
