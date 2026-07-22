"""Deterministic training-inclusive efficiency audit for V5 G6 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from src.path_integral.provenance import source_provenance

METHODS = ("crude_mc", "pure_cem_slis", "defensive_cem_mixture")


def _canonical_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize an empty efficiency sample")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("audit input must be a JSON object")
    return payload


def audit(
    hybrid_path: Path,
    baseline_path: Path,
    *,
    minimum_final_samples: int = 128,
) -> dict[str, Any]:
    if minimum_final_samples < 2:
        raise ValueError("minimum_final_samples must be at least two")
    hybrid = _load_object(hybrid_path)
    baseline = _load_object(baseline_path)
    if hybrid.get("schema") != "npi.g11.v5-confirmatory.v1" or not hybrid.get(
        "qualification_passed"
    ):
        raise ValueError("hybrid input is not a passed achieved-RMSE qualification")
    if baseline.get("schema") != "npi.g11.v5-baseline-qualification.v1" or not baseline.get(
        "baseline_qualified"
    ):
        raise ValueError("baseline input is not qualified")

    baseline_by_key = {
        (record["model_id"], record["task"], int(record["cluster"])): record
        for record in baseline["records"]
    }
    if len(baseline_by_key) != len(baseline["records"]):
        raise ValueError("baseline artifact contains duplicate cell/cluster keys")

    records: list[dict[str, Any]] = []
    for hybrid_record in hybrid["records"]:
        cluster_text = str(hybrid_record["cell_id"]).rsplit(":cluster-", 1)
        if len(cluster_text) != 2:
            raise ValueError("malformed hybrid cluster id")
        cluster = int(cluster_text[1])
        key = (hybrid_record["model_id"], hybrid_record["task_name"], cluster)
        if key not in baseline_by_key:
            raise ValueError(f"missing matched baseline record for {key}")
        baseline_record = baseline_by_key[key]
        final = hybrid_record["result"]
        if not final["complete"] or final["resource_censored"]:
            raise ValueError("efficiency audit requires complete uncensored hybrid records")
        probability = float(hybrid_record["reference"]["probability"])
        relative_rmse = float(final["requested_relative_sampling_rmse"])
        target_variance = (probability * relative_rmse) ** 2
        hybrid_work = math.fsum(
            float(entry["work_units"]) for entry in final["work"]["entries"]
        )
        projections: dict[str, Any] = {}
        for method in METHODS:
            summary = baseline_record[method]
            training_work = (
                0.0 if method == "crude_mc" else float(baseline_record["cem"]["training_work_units"])
            )
            evaluation_work = float(summary["work_units"]) - training_work
            cost_per_sample = evaluation_work / int(summary["samples"])
            required_samples = max(
                minimum_final_samples,
                math.ceil(float(summary["variance"]) / target_variance),
            )
            projected_work = training_work + required_samples * cost_per_sample
            projections[method] = {
                "variance_source": float(summary["variance"]),
                "training_work": training_work,
                "cost_per_sample": cost_per_sample,
                "required_samples": required_samples,
                "projected_total_work": projected_work,
                "baseline_to_hybrid_work_ratio": projected_work / hybrid_work,
            }
        records.append(
            {
                "model_id": key[0],
                "task_name": key[1],
                "cluster": cluster,
                "reference_probability": probability,
                "requested_relative_sampling_rmse": relative_rmse,
                "hybrid_selected_candidate": final["selected_candidate"],
                "hybrid_total_work": hybrid_work,
                "hybrid_selection_work": math.fsum(
                    float(entry["work_units"])
                    for entry in final["work"]["entries"]
                    if entry["role"] == "selection"
                ),
                "baseline_projections": projections,
            }
        )

    expected = len(hybrid["records"])
    if len(records) != expected or expected == 0:
        raise ValueError("efficiency matrix is incomplete")
    aggregates: list[dict[str, Any]] = []
    cells = sorted({(record["model_id"], record["task_name"]) for record in records})
    for model_id, task_name in cells:
        group = [
            record
            for record in records
            if record["model_id"] == model_id and record["task_name"] == task_name
        ]
        method_summaries: dict[str, Any] = {}
        for method in METHODS:
            ratios = [
                float(record["baseline_projections"][method]["baseline_to_hybrid_work_ratio"])
                for record in group
            ]
            method_summaries[method] = {
                "median_baseline_to_hybrid_work_ratio": _quantile(ratios, 0.5),
                "p10_baseline_to_hybrid_work_ratio": _quantile(ratios, 0.1),
                "p90_baseline_to_hybrid_work_ratio": _quantile(ratios, 0.9),
                "hybrid_cheaper_fraction": sum(ratio > 1.0 for ratio in ratios) / len(ratios),
            }
        selection_fractions = [
            record["hybrid_selection_work"] / record["hybrid_total_work"] for record in group
        ]
        aggregates.append(
            {
                "model_id": model_id,
                "task_name": task_name,
                "clusters": len(group),
                "reference_probability": group[0]["reference_probability"],
                "selection_work_fraction_median": _quantile(selection_fractions, 0.5),
                "methods": method_summaries,
            }
        )

    gates = {
        "complete_matched_matrix": len(records) == len(baseline["records"]),
        "all_cells_rare_at_most_5_percent": all(
            aggregate["reference_probability"] <= 0.05 for aggregate in aggregates
        ),
        "hybrid_median_work_beats_crude_all_cells": all(
            aggregate["methods"]["crude_mc"]["median_baseline_to_hybrid_work_ratio"] >= 1.0
            for aggregate in aggregates
        ),
        "hybrid_median_work_beats_pure_cem_all_cells": all(
            aggregate["methods"]["pure_cem_slis"]["median_baseline_to_hybrid_work_ratio"]
            >= 1.0
            for aggregate in aggregates
        ),
        "hybrid_median_work_beats_defensive_cem_all_cells": all(
            aggregate["methods"]["defensive_cem_mixture"][
                "median_baseline_to_hybrid_work_ratio"
            ]
            >= 1.0
            for aggregate in aggregates
        ),
    }
    return {
        "schema": "npi.g11.v5-g6-efficiency-audit.v1",
        "hybrid_input": {
            "path": str(hybrid_path),
            "canonical_json_sha256": _canonical_hash(hybrid),
        },
        "baseline_input": {
            "path": str(baseline_path),
            "canonical_json_sha256": _canonical_hash(baseline),
        },
        "minimum_final_samples": minimum_final_samples,
        "projection_scope": (
            "Held-out baseline variances projected to the hybrid sampling-variance target; "
            "CEM training work is included and no performance claim is made from this audit alone."
        ),
        "records": records,
        "aggregates": aggregates,
        "gates": gates,
        "g6_efficiency_passed": all(gates.values()),
        **source_provenance(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-final-samples", type=int, default=128)
    arguments = parser.parse_args()
    result = audit(
        arguments.hybrid,
        arguments.baseline,
        minimum_final_samples=arguments.minimum_final_samples,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
