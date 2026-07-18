"""Development stop gate for unbiased stratified rank-two marginalization."""

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
    _simulator,
    _task,
)
from src.path_integral import (
    evaluate_control_span_marginalized_mixture,
    evaluate_rank_two_control_span_marginalized_mixture,
    simulate_rbergomi_mixture,
)


def _load(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("expected a rank-two development schema-version-1 config")
    if payload.get("development_only") is not True:
        raise ValueError("rank-two selection must remain development-only")
    return payload, hashlib.sha256(raw).hexdigest()


def _source(source: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(Path(source["path"]).read_text(encoding="utf-8"))
    if _canonical_json_sha256(payload) != str(source["canonical_json_sha256"]):
        raise ValueError("rank-two selection source hash mismatch")
    if payload.get("smoke") is not False or payload.get("passed") is not True:
        raise ValueError("rank-two selection source must be passed and non-smoke")
    return payload


def _evaluate_regime(
    regime: dict[str, Any],
    *,
    inner_counts: list[int],
    decay_candidates: tuple[float, ...],
    steps: int,
    paths: int,
    seeds: list[int],
    label_seed_base: int,
    inner_seed_base: int,
    gates: dict[str, Any],
) -> dict[str, Any]:
    simulator = _simulator(regime["model"])
    task = _task(regime["task"])
    controls = _controls(regime)
    alpha = float(regime["selected_natural_weight"])
    weights = torch.tensor([alpha, 1.0 - alpha], dtype=torch.float64)
    raw_chunks: list[torch.Tensor] = []
    rank_one_chunks: list[torch.Tensor] = []
    rank_two_chunks: dict[int, list[torch.Tensor]] = {count: [] for count in inner_counts}
    runs: list[dict[str, Any]] = []
    maximum_exactness: dict[int, float] = {count: 0.0 for count in inner_counts}
    maximum_bound: dict[int, float] = {count: 0.0 for count in inner_counts}
    for seed_index, seed in enumerate(seeds):
        torch.manual_seed(seed)
        start = time.perf_counter()
        sample = simulate_rbergomi_mixture(
            simulator,
            controls,
            weights,
            spot=float(regime["model"]["spot"]),
            maturity=float(regime["model"]["maturity"]),
            dt=float(regime["model"]["maturity"]) / steps,
            num_paths=paths,
            label_generator=torch.Generator().manual_seed(label_seed_base + seed_index),
            engine="fft",
        )
        simulation_seconds = time.perf_counter() - start
        start = time.perf_counter()
        raw = task.hard_event(sample.paths.spot, sample.paths.step_dt).to(torch.float64) * torch.exp(
            sample.mixture_log_likelihood
        )
        raw_postprocess_seconds = time.perf_counter() - start
        start = time.perf_counter()
        rank_one = evaluate_control_span_marginalized_mixture(
            sample,
            task=task,
            rho=simulator.rho,
        )
        rank_one_postprocess_seconds = time.perf_counter() - start
        raw_variance = float(raw.var(unbiased=True))
        rank_one_variance = float(rank_one.marginalized_contribution.var(unbiased=True))
        raw_cost = (simulation_seconds + raw_postprocess_seconds) / paths
        rank_one_cost = (simulation_seconds + rank_one_postprocess_seconds) / paths
        reports: dict[str, Any] = {}
        for count_index, count in enumerate(inner_counts):
            start = time.perf_counter()
            rank_two = evaluate_rank_two_control_span_marginalized_mixture(
                sample,
                task=task,
                rho=simulator.rho,
                inner_samples=count,
                inner_generator=torch.Generator().manual_seed(
                    inner_seed_base + 10_000 * count_index + seed_index
                ),
                decay_candidates=decay_candidates,
                inner_rule="stratified",
            )
            rank_two_postprocess_seconds = time.perf_counter() - start
            rank_two_variance = float(rank_two.marginalized_contribution.var(unbiased=True))
            rank_two_cost = (simulation_seconds + rank_two_postprocess_seconds) / paths
            exactness = max(
                rank_two.maximum_component_log_density_error,
                rank_two.maximum_mixture_log_density_error,
                rank_two.maximum_path_reconstruction_error,
                rank_two.maximum_residual_projection,
            )
            maximum_exactness[count] = max(maximum_exactness[count], exactness)
            maximum_bound[count] = max(
                maximum_bound[count],
                rank_two.maximum_defensive_bound_violation,
            )
            reports[str(count)] = {
                "inner_samples": count,
                "rank_two_postprocess_seconds": rank_two_postprocess_seconds,
                "rank_two_variance": rank_two_variance,
                "raw_over_rank_two_work_ratio": (
                    raw_variance * raw_cost / (rank_two_variance * rank_two_cost)
                ),
                "rank_one_over_rank_two_work_ratio": (
                    rank_one_variance
                    * rank_one_cost
                    / (rank_two_variance * rank_two_cost)
                ),
                "event_direction_decay": rank_two.subspace.event_direction_decay,
                "minimum_gram_eigenvalue": rank_two.subspace.minimum_gram_eigenvalue,
                "maximum_exactness_error": exactness,
            }
            rank_two_chunks[count].append(rank_two.marginalized_contribution.detach().cpu())
        runs.append(
            {
                "seed": seed,
                "simulation_seconds": simulation_seconds,
                "raw_postprocess_seconds": raw_postprocess_seconds,
                "rank_one_postprocess_seconds": rank_one_postprocess_seconds,
                "raw_variance": raw_variance,
                "rank_one_variance": rank_one_variance,
                "raw_over_rank_one_work_ratio": (
                    raw_variance * raw_cost / (rank_one_variance * rank_one_cost)
                ),
                "rank_two": reports,
            }
        )
        raw_chunks.append(raw.detach().cpu())
        rank_one_chunks.append(rank_one.marginalized_contribution.detach().cpu())

    raw_all = torch.cat(raw_chunks)
    rank_one_all = torch.cat(rank_one_chunks)
    candidates: list[dict[str, Any]] = []
    for count in inner_counts:
        rank_two_all = torch.cat(rank_two_chunks[count])
        difference = rank_two_all - raw_all
        paired_se = math.sqrt(float(difference.var(unbiased=True)) / difference.numel())
        paired_z = float(difference.mean()) / paired_se if paired_se > 0.0 else 0.0
        raw_ratios = [
            float(run["rank_two"][str(count)]["raw_over_rank_two_work_ratio"])
            for run in runs
        ]
        rank_one_ratios = [
            float(run["rank_two"][str(count)]["rank_one_over_rank_two_work_ratio"])
            for run in runs
        ]
        candidate_gates = {
            "exactness": maximum_exactness[count]
            <= float(gates["maximum_exactness_error"]),
            "defensive_bound": maximum_bound[count]
            <= float(gates["maximum_defensive_bound_violation"]),
            "mean_consistency": abs(paired_z)
            <= float(gates["maximum_paired_mean_difference_z"]),
            "raw_work": _geometric_mean(raw_ratios)
            > float(gates["minimum_raw_over_rank_two_work_ratio"]),
            "rank_one_work": _geometric_mean(rank_one_ratios)
            > float(gates["minimum_rank_one_over_rank_two_work_ratio"]),
        }
        candidates.append(
            {
                "inner_samples": count,
                "paths": int(rank_two_all.numel()),
                "estimate": float(rank_two_all.mean()),
                "variance": float(rank_two_all.var(unbiased=True)),
                "paired_mean_difference_z": paired_z,
                "geometric_raw_over_rank_two_work_ratio": _geometric_mean(raw_ratios),
                "geometric_rank_one_over_rank_two_work_ratio": _geometric_mean(
                    rank_one_ratios
                ),
                "maximum_exactness_error": maximum_exactness[count],
                "maximum_defensive_bound_violation": maximum_bound[count],
                "gates": candidate_gates,
                "admissible": all(
                    value
                    for key, value in candidate_gates.items()
                    if key not in ("raw_work", "rank_one_work")
                ),
                "passed": all(candidate_gates.values()),
            }
        )
    admissible = [candidate for candidate in candidates if candidate["admissible"]]
    selected = (
        max(
            admissible,
            key=lambda candidate: (
                float(candidate["geometric_raw_over_rank_two_work_ratio"]),
                -int(candidate["inner_samples"]),
            ),
        )
        if admissible
        else None
    )
    return {
        "regime_id": regime["regime_id"],
        "natural_weight": alpha,
        "raw_estimate": float(raw_all.mean()),
        "rank_one_estimate": float(rank_one_all.mean()),
        "rank_one_variance": float(rank_one_all.var(unbiased=True)),
        "candidates": candidates,
        "runs": runs,
        "selected_inner_samples": int(selected["inner_samples"]) if selected else None,
        "selected_raw_work_ratio": (
            float(selected["geometric_raw_over_rank_two_work_ratio"]) if selected else None
        ),
        "selected_rank_one_work_ratio": (
            float(selected["geometric_rank_one_over_rank_two_work_ratio"])
            if selected
            else None
        ),
        "passed": bool(selected is not None and selected["passed"]),
    }


def run(config_path: Path, *, smoke: bool) -> dict[str, Any]:
    config, digest = _load(config_path)
    source = _source(config["selection_source"])
    available = {str(regime["regime_id"]): regime for regime in source["regimes"]}
    regime_ids = [str(value) for value in config["regime_ids"]]
    if smoke:
        regime_ids = regime_ids[:1]
    inner_counts = [int(value) for value in config["inner_samples"]]
    if smoke:
        inner_counts = inner_counts[:2]
    if any(value <= 0 for value in inner_counts) or len(set(inner_counts)) != len(inner_counts):
        raise ValueError("inner sample counts must be unique and positive")
    decay_candidates = tuple(
        float(value) for value in config["event_direction_decay_candidates"]
    )
    evaluation = config["evaluation"]
    steps = 64 if smoke else int(evaluation["fine_steps"])
    paths = 500 if smoke else int(evaluation["paths_per_seed"])
    seeds = [int(value) for value in evaluation["seeds"]]
    if smoke:
        seeds = seeds[:2]
    torch.set_num_threads(int(evaluation["thread_count"]))
    outputs = [
        _evaluate_regime(
            available[regime_id],
            inner_counts=inner_counts,
            decay_candidates=decay_candidates,
            steps=steps,
            paths=paths,
            seeds=[seed + 100_000 * regime_index for seed in seeds],
            label_seed_base=int(evaluation["label_seed_base"]) + 100_000 * regime_index,
            inner_seed_base=int(evaluation["inner_seed_base"]) + 100_000 * regime_index,
            gates=config["gates"],
        )
        for regime_index, regime_id in enumerate(regime_ids)
    ]
    raw_ratios = [
        float(output["selected_raw_work_ratio"])
        for output in outputs
        if output["selected_raw_work_ratio"] is not None
    ]
    rank_one_ratios = [
        float(output["selected_rank_one_work_ratio"])
        for output in outputs
        if output["selected_rank_one_work_ratio"] is not None
    ]
    improved_fraction = (
        statistics.mean(value > 1.0 for value in rank_one_ratios)
        if len(rank_one_ratios) == len(outputs)
        else 0.0
    )
    gates = {
        "all_regimes_admissible": len(raw_ratios) == len(outputs),
        "raw_work": bool(raw_ratios)
        and _geometric_mean(raw_ratios)
        > float(config["gates"]["minimum_raw_over_rank_two_work_ratio"]),
        "rank_one_work": bool(rank_one_ratios)
        and _geometric_mean(rank_one_ratios)
        > float(config["gates"]["minimum_rank_one_over_rank_two_work_ratio"]),
        "improved_regime_fraction": improved_fraction
        >= float(config["gates"]["minimum_improved_regime_fraction"]),
    }
    return {
        "protocol_id": config["protocol_id"],
        "protocol_sha256": digest,
        "development_only": True,
        "smoke": smoke,
        "selection_hash": str(config["selection_source"]["canonical_json_sha256"]),
        "steps": steps,
        "paths_per_seed": paths,
        "seeds": seeds,
        "inner_rule": "randomized stratified normal",
        "regimes": outputs,
        "aggregate": {
            "geometric_raw_over_rank_two_work_ratio": (
                _geometric_mean(raw_ratios) if raw_ratios else None
            ),
            "geometric_rank_one_over_rank_two_work_ratio": (
                _geometric_mean(rank_one_ratios) if rank_one_ratios else None
            ),
            "improved_regime_fraction": improved_fraction,
            "passed_regimes": sum(bool(output["passed"]) for output in outputs),
            "total_regimes": len(outputs),
        },
        "gates": gates,
        "passed": all(gates.values()),
        "restriction": "development-only rank-two stop gate; no confirmatory claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/g10_rank_two_development.yaml"),
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
