"""Untouched multi-regime validation for defensive control-span marginalization."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch
import yaml

from experiments.g10_control_span_correction_development import _regime as _correction_regime
from experiments.g10_control_span_development import (
    _candidate as _single_candidate,
)
from experiments.g10_control_span_development import (
    _canonical_json_sha256,
    _geometric_mean,
)


def _load(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("expected a G10 frozen schema-version-1 config")
    if payload.get("frozen") is not True:
        raise ValueError("G10 validation protocol must be frozen")
    levels = [int(value) for value in payload["hierarchy"]["correction_fine_steps"]]
    if len(levels) < 4 or any(
        right != 2 * left for left, right in zip(levels[:-1], levels[1:], strict=True)
    ):
        raise ValueError("correction hierarchy must contain at least four dyadic levels")
    return payload, hashlib.sha256(raw).hexdigest()


def _verified(source: dict[str, Any], *, require_passed: bool) -> dict[str, Any]:
    payload = json.loads(Path(source["path"]).read_text(encoding="utf-8"))
    if _canonical_json_sha256(payload) != str(source["canonical_json_sha256"]):
        raise ValueError(f"source canonical JSON hash mismatch: {source['path']}")
    if payload.get("smoke") is not False:
        raise ValueError(f"source must be a non-smoke artifact: {source['path']}")
    if require_passed and payload.get("passed") is not True:
        raise ValueError(f"selection source must have passed: {source['path']}")
    return payload


def _log_ratio_lower_95(values: list[float]) -> float:
    logs = [math.log(value) for value in values]
    if len(logs) < 2:
        raise ValueError("at least two ratios are required")
    critical = {9: 1.833, 11: 1.796}.get(len(logs) - 1, 1.645)
    return math.exp(
        statistics.mean(logs)
        - critical * statistics.stdev(logs) / math.sqrt(len(logs))
    )


def run(config_path: Path, *, smoke: bool) -> dict[str, Any]:
    config, digest = _load(config_path)
    selection = _verified(config["selection_source"], require_passed=True)
    reference_payload = _verified(config["reference_source"], require_passed=False)
    references = {
        str(regime["regime_id"]): regime["reference"]
        for regime in reference_payload["regimes"]
    }
    selected_regimes = selection["regimes"]
    if set(references) != {str(regime["regime_id"]) for regime in selected_regimes}:
        raise ValueError("selection and reference regime sets differ")
    if smoke:
        selected_regimes = selected_regimes[:1]
    validation = config["validation"]
    seeds = [int(value) for value in validation["seeds"]]
    if smoke:
        seeds = seeds[:2]
    paths = 500 if smoke else int(validation["paths_per_seed"])
    single_steps = 64 if smoke else int(config["hierarchy"]["single_steps"])
    levels = [int(value) for value in config["hierarchy"]["correction_fine_steps"]]
    if smoke:
        levels = levels[:2]
    torch.set_num_threads(int(validation["thread_count"]))
    outputs: list[dict[str, Any]] = []
    for regime_index, regime in enumerate(selected_regimes):
        alpha = float(regime["selected_natural_weight"])
        regime_seeds = [seed + 100_000 * regime_index for seed in seeds]
        single = _single_candidate(
            regime,
            alpha=alpha,
            alpha_index=0,
            steps=single_steps,
            paths=paths,
            seeds=regime_seeds,
            label_seed_base=int(validation["single_label_seed_base"])
            + 100_000 * regime_index,
            gates=config["gates"],
        )
        correction = _correction_regime(
            regime,
            alpha=alpha,
            levels=levels,
            paths=paths,
            seeds=regime_seeds,
            label_seed_base=int(validation["correction_label_seed_base"])
            + 100_000 * regime_index,
            gates=config["gates"],
        )
        reference = references[str(regime["regime_id"])]
        difference = float(single["marginalized_estimate"]) - float(reference["estimate"])
        comparison_se = math.sqrt(
            float(single["marginalized_variance"]) / int(single["paths"])
            + float(reference["standard_error"]) ** 2
        )
        reference_z = difference / comparison_se if comparison_se > 0.0 else 0.0
        single_ratios = [
            float(run["raw_over_marginalized_work_ratio"]) for run in single["runs"]
        ]
        regime_gates = {
            "single_exactness": bool(single["gates"]["exactness"]),
            "single_mean_consistency": bool(single["gates"]["mean_consistency"]),
            "single_outer_likelihood": bool(
                single["gates"]["outer_likelihood_normalization"]
            ),
            "single_total_work": float(
                single["geometric_raw_over_marginalized_work_ratio"]
            )
            > float(config["gates"]["minimum_geometric_work_ratio"]),
            "single_total_work_lower_95": _log_ratio_lower_95(single_ratios)
            > float(config["gates"]["minimum_total_work_ratio_lower_95"]),
            "reference_consistency": abs(reference_z)
            <= float(config["gates"]["maximum_reference_difference_z"]),
            "correction_exactness": bool(correction["gates"]["exactness"]),
            "correction_mean_consistency": bool(
                correction["gates"]["mean_consistency"]
            ),
            "correction_outer_likelihood": bool(
                correction["gates"]["outer_likelihood_normalization"]
            ),
            "correction_work": bool(correction["gates"]["correction_work"]),
        }
        outputs.append(
            {
                "regime_id": regime["regime_id"],
                "group": regime["group"],
                "model": regime["model"],
                "task": regime["task"],
                "natural_weight": alpha,
                "reference": reference,
                "reference_difference_z": reference_z,
                "single": single,
                "correction": correction,
                "single_work_ratio_lower_95_one_sided": _log_ratio_lower_95(
                    single_ratios
                ),
                "gates": regime_gates,
                "passed": all(regime_gates.values()),
            }
        )

    core = [regime for regime in outputs if regime["group"] == "core"]
    stress = [regime for regime in outputs if regime["group"] == "stress"]
    suite_seed_ratios = [
        _geometric_mean(
            [
                float(regime["single"]["runs"][seed_index]["raw_over_marginalized_work_ratio"])
                for regime in core
            ]
        )
        for seed_index in range(len(seeds))
    ]
    core_regime_ratios = [
        float(regime["single"]["geometric_raw_over_marginalized_work_ratio"])
        for regime in core
    ]
    correction_ratios = [
        float(regime["correction"]["geometric_raw_over_marginalized_correction_work_ratio"])
        for regime in core
    ]
    improved_core = statistics.mean(value > 1.0 for value in core_regime_ratios)
    likelihood_core = statistics.mean(
        bool(regime["gates"]["single_outer_likelihood"])
        and bool(regime["gates"]["correction_outer_likelihood"])
        for regime in core
    )
    aggregate_single = _geometric_mean(suite_seed_ratios)
    aggregate_correction = _geometric_mean(correction_ratios)
    aggregate_gates = {
        "geometric_total_work": aggregate_single
        > float(config["gates"]["minimum_geometric_work_ratio"]),
        "total_work_lower_95": _log_ratio_lower_95(suite_seed_ratios)
        > float(config["gates"]["minimum_total_work_ratio_lower_95"]),
        "improved_core_regime_fraction": improved_core
        >= float(config["gates"]["minimum_improved_core_regime_fraction"]),
        "core_likelihood_pass_fraction": likelihood_core
        >= float(config["gates"]["minimum_core_likelihood_pass_fraction"]),
        "geometric_correction_work": aggregate_correction
        > float(config["gates"]["minimum_geometric_correction_work_ratio"]),
        "core_exactness": all(
            bool(regime["gates"]["single_exactness"])
            and bool(regime["gates"]["correction_exactness"])
            for regime in core
        ),
        "stress_exactness": all(
            bool(regime["gates"]["single_exactness"])
            and bool(regime["gates"]["correction_exactness"])
            for regime in stress
        ),
        "reference_consistency": all(
            bool(regime["gates"]["reference_consistency"]) for regime in outputs
        ),
    }
    return {
        "protocol_id": config["protocol_id"],
        "protocol_sha256": digest,
        "smoke": smoke,
        "source_hashes": {
            str(config["selection_source"]["path"]): str(
                config["selection_source"]["canonical_json_sha256"]
            ),
            str(config["reference_source"]["path"]): str(
                config["reference_source"]["canonical_json_sha256"]
            ),
        },
        "theory_contract": {
            "target": "finite-grid hit-and-occupation probability",
            "proposal": "natural/CEM defensive mixture with frozen natural weight",
            "integrated_subspace": "complete rank-one deterministic price-control span",
            "outer_likelihood_bound": "one over natural-component weight",
            "self_normalized": False,
            "continuous_time_claimed": False,
            "work_metric": "paired online variance-times-cost; calibration excluded",
            "aggregate_inference": "fixed core suite clustered by validation seed",
        },
        "single_steps": single_steps,
        "correction_levels": levels,
        "paths_per_seed": paths,
        "validation_seeds": seeds,
        "regimes": outputs,
        "aggregate": {
            "core_regimes": len(core),
            "stress_regimes": len(stress),
            "geometric_raw_over_marginalized_single_work_ratio": aggregate_single,
            "fixed_suite_total_work_ratio_lower_95_one_sided": _log_ratio_lower_95(
                suite_seed_ratios
            ),
            "regime_heterogeneity_lower_95_one_sided": _log_ratio_lower_95(
                core_regime_ratios
            )
            if len(core_regime_ratios) >= 2
            else None,
            "improved_core_regime_fraction": improved_core,
            "core_likelihood_pass_fraction": likelihood_core,
            "geometric_raw_over_marginalized_correction_work_ratio": aggregate_correction,
            "passed_core_regimes": sum(bool(regime["passed"]) for regime in core),
            "passed_stress_regimes": sum(bool(regime["passed"]) for regime in stress),
        },
        "gates": aggregate_gates,
        "passed": all(aggregate_gates.values()),
        "interpretation": "untouched aggregate gate; all regime failures retained",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/g10_control_span_frozen.yaml"),
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
