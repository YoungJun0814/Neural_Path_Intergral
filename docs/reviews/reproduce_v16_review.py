"""Read-only numerical/semantic checks supporting the 2026-09-27 review.

No experiments are retrained and no historical result or source file is changed.
The optional output is a new review artifact, not a replacement benchmark.
Run from the repository root with ``python -m docs.reviews.reproduce_v16_review``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
from pathlib import Path
from unittest.mock import patch

import yaml

from experiments import g11_v16_final_policy_audit as audit_module

ROOT = Path(__file__).resolve().parents[2]
AUDIT_CONFIG = "configs/g11_v15/v16_final_policy_audit_v1.yaml"


def read(path: str) -> dict:
    source = ROOT / path
    raw = source.read_text(encoding="utf-8")
    return json.loads(raw) if source.suffix == ".json" else yaml.safe_load(raw)


def summarize_confirmation(path: str) -> list[dict]:
    artifact = read(path)
    config = read(artifact["config_binding"]["path"])
    baseline = read(artifact["artifact_bindings"]["baseline"]["path"])
    cells = {cell["cell_id"]: cell for cell in baseline["cells"]}
    summaries = []
    for record in artifact["variants"]:
        baseline_cell = cells[record["reference_cell"]]
        methods = baseline_cell["methods"]
        qualified = set(record["qualified_comparators"])
        original_qualified = set(baseline_cell["accuracy_qualified_comparators"])
        units = len(record["clusters"]) * config["evaluation"]["samples_per_cluster"]
        horizon_ratios = {}
        for horizon in (1, 10, 100, 1000):
            candidate_cost = record["training_work"] + horizon * record["evaluation_work"]
            candidate_wnv = record["sample_variance"] / units * candidate_cost
            ratios = {}
            for method in methods:
                if method["method"] not in qualified:
                    continue
                cost = method["training_cost"]["algorithmic_work_units"] + (
                    horizon * method["evaluation_cost"]["algorithmic_work_units"]
                )
                ratios[method["method"]] = (
                    method["sample_variance"] / method["inferential_units"] * cost / candidate_wnv
                )
            strongest = min(ratios, key=ratios.get)
            horizon_ratios[str(horizon)] = {
                "strongest_comparator": strongest,
                "comparator_over_candidate": ratios[strongest],
            }
        standard_error = math.sqrt(record["sample_variance"] / units)
        accuracy_z = abs(record["estimate"] - record["external_reference_estimate"]) / math.hypot(
            max(standard_error, record["between_cluster_standard_error"]),
            record["external_reference_standard_error"],
        )
        summaries.append(
            {
                "cell": record["reference_cell"],
                "initializer": record["initializer"],
                "role": record.get("claim_role", "canonical_dominance"),
                "estimate": record["estimate"],
                "reference_estimate": record["external_reference_estimate"],
                "reference_rse": record["external_reference_standard_error"]
                / record["external_reference_estimate"],
                "relative_difference_from_reference": record["estimate"]
                / record["external_reference_estimate"]
                - 1,
                "robust_rse": record["robust_relative_standard_error"],
                "accuracy_z_recomputed": accuracy_z,
                "accuracy_z_matches": math.isclose(
                    accuracy_z, record["external_accuracy_z"], rel_tol=1e-12
                ),
                "evaluation_clusters": len(record["clusters"]),
                "evaluation_samples": units,
                "training_fits_per_confirmation_variant": 1,
                "horizon_ratios": horizon_ratios,
                "primary_ratio_matches": math.isclose(
                    horizon_ratios[str(config["evaluation"]["primary_query_count"])][
                        "comparator_over_candidate"
                    ],
                    record["strongest_comparator_over_candidate_work_ratio"],
                    rel_tol=1e-12,
                ),
                "added_to_qualified_by_candidate_runner": sorted(qualified - original_qualified),
                "removed_from_qualified_by_candidate_runner": sorted(
                    original_qualified - qualified
                ),
                "baseline_rse": {
                    m["method"]: (
                        m["robust_relative_standard_error"]
                        if math.isfinite(m["robust_relative_standard_error"])
                        else None
                    )
                    for m in methods
                },
                "baseline_nonfinite_rse_methods": [
                    m["method"]
                    for m in methods
                    if not math.isfinite(m["robust_relative_standard_error"])
                ],
                "candidate_training_work": record["training_work"],
                "candidate_evaluation_work": record["evaluation_work"],
            }
        )
    return summaries


def semantic_audit_probes(config: dict) -> dict:
    """Isolate semantic validation AFTER integrity checks; no disk tampering.

    The mocked loader represents newly bound, internally inconsistent artifacts.
    This does not demonstrate a bypass of the SHA-256 integrity check.
    """
    loaded = {
        binding["path"]: audit_module._load_bound(binding)
        for binding in config["artifacts"].values()
    }
    ood_path = config["artifacts"]["ood_confirmation"]["path"]
    results = {}
    for probe in (
        "impossible_accuracy_and_rse_with_stale_pass",
        "nan_dominance_ratio_with_stale_pass",
    ):
        altered = copy.deepcopy(loaded)
        for record in altered[ood_path]["variants"]:
            if probe == "impossible_accuracy_and_rse_with_stale_pass":
                record["external_accuracy_z"] = 1_000_000.0
                record["robust_relative_standard_error"] = 1_000_000.0
                record["likelihood_normalization_z"] = 1_000_000.0
            elif record.get("claim_role") == "dominance_candidate":
                record["strongest_comparator_over_candidate_work_ratio"] = float("nan")
        with patch.object(
            audit_module, "_load_bound", side_effect=lambda binding, data=altered: data[binding["path"]]
        ):
            result = audit_module.build_audit(config)
        results[probe] = {
            "accepted_by_semantic_layer": result["passed"],
            "internal_failures": result["internal_failures"],
        }
    return results


def build_report() -> dict:
    config = read(AUDIT_CONFIG)
    actual = audit_module.build_audit(config)
    paths = [
        AUDIT_CONFIG,
        *(binding["path"] for binding in config["artifacts"].values()),
        "experiments/g11_v16_final_policy_audit.py",
        "experiments/g11_v16_deep_tail_transport_development.py",
        "src/path_integral/defensive_proposal_selection.py",
    ]
    operator_path = "results/g11_v16_operator_confirmation_v2_2026-08-11.json"
    operator = read(operator_path)
    paths.append(operator_path)
    operator_summaries = []
    for replication in operator["replications"]:
        holdouts = replication["holdouts"]
        operator_summaries.append(
            {
                "training_seed": replication["training_seed"],
                "holdout_tasks": len(holdouts),
                "median_function_evaluation_speedup": replication[
                    "median_function_evaluation_speedup"
                ],
                "median_recorded_cold_over_correction_wall_ratio": statistics.median(
                    h["cold_seconds"] / h["correction_seconds"] for h in holdouts
                ),
                "total_recorded_cold_over_correction_wall_ratio": sum(
                    h["cold_seconds"] for h in holdouts
                )
                / sum(h["correction_seconds"] for h in holdouts),
                "correction_faster_tasks": sum(
                    h["cold_seconds"] > h["correction_seconds"] for h in holdouts
                ),
            }
        )
    return {
        "schema": "npi.review.v16-reorientation.2026-09-27.v1",
        "scope": "existing-artifact recalculation, source review, and in-memory semantic probes; not fresh scientific confirmation",
        "input_sha256": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths
        },
        "historical_audit_reproduced_pass": actual["passed"],
        "semantic_probes": semantic_audit_probes(config),
        "operator_summaries": operator_summaries,
        "operator_timing_caveat": "Single-run recorded correction timings only; cold records are reused across training replications. Not controlled repeated wall-time benchmarks; teacher/training costs excluded here.",
        "confirmation_summaries": [
            *summarize_confirmation(config["artifacts"]["canonical_confirmation"]["path"]),
            *summarize_confirmation(config["artifacts"]["ood_confirmation"]["path"]),
        ],
        "note_on_horizons": "Same existing frozen proposals and qualified comparator sets. Query count changes only the amortization arithmetic, not the task distribution. Ratios are work-proxy point estimates, not wall-time confidence bounds.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = build_report()
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        output = args.output.resolve()
        if output.parent != Path(__file__).resolve().parent:
            raise ValueError("write review outputs only beside this diagnostic script")
        if output.exists():
            raise FileExistsError("preserve existing review evidence; use a new output path")
        output.write_text(payload, encoding="utf-8")
        print(output)


if __name__ == "__main__":
    main()
