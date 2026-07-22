"""Independent arithmetic and contract audit for V5 confirmatory artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import yaml


def _close(left: float, right: float, tolerance: float = 1e-11) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def _linear_quantile(sorted_values: list[float], probability: float) -> float | None:
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    return sorted_values[lower] + (position - lower) * (
        sorted_values[upper] - sorted_values[lower]
    )


def _aggregate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((record["model_id"], record["task_name"]), []).append(record)
    critical = NormalDist().inv_cdf(0.975)
    summaries: list[dict[str, Any]] = []
    for (model_id, task_name), group in sorted(grouped.items()):
        completed = [
            record
            for record in group
            if record["result"]["complete"] and not record["result"]["resource_censored"]
        ]
        reference = group[0]["reference"]
        probability = float(reference["probability"])
        reference_se = float(reference["standard_error"])
        squared_errors = [
            (float(record["result"]["estimate"]) - probability) ** 2 for record in completed
        ]
        rmse = (
            math.sqrt(math.fsum(squared_errors) / len(squared_errors))
            if squared_errors
            else None
        )
        coverage: list[bool] = []
        bounded_coverage: list[bool] = []
        attainment: list[bool] = []
        work: list[float] = []
        selections: dict[str, int] = {}
        for record in completed:
            final = record["result"]
            combined_radius = critical * math.sqrt(
                float(final["empirical_sampling_variance"]) + reference_se**2
            )
            coverage.append(abs(float(final["estimate"]) - probability) <= combined_radius)
            bounded = final["bounded_confidence_interval"]
            bounded_coverage.append(float(bounded[0]) <= probability <= float(bounded[1]))
            attainment.append(bool(final["empirical_target_attained"]))
            work.append(
                math.fsum(float(item["work_units"]) for item in final["work"]["entries"])
            )
            selected = str(final["selected_candidate"])
            selections[selected] = selections.get(selected, 0) + 1
        work.sort()
        requested = float(group[0]["result"]["requested_relative_sampling_rmse"])
        summaries.append(
            {
                "model_id": model_id,
                "task_name": task_name,
                "clusters_planned": len(group),
                "clusters_complete": len(completed),
                "resource_censored": len(group) - len(completed),
                "reference": reference,
                "requested_relative_sampling_rmse": requested,
                "empirical_rmse_against_reference": rmse,
                "empirical_relative_rmse_against_reference": (
                    rmse / probability if rmse is not None else None
                ),
                "empirical_target_attainment_fraction": (
                    sum(attainment) / len(attainment) if attainment else None
                ),
                "combined_asymptotic_95_coverage": (
                    sum(coverage) / len(coverage) if coverage else None
                ),
                "bounded_interval_coverage": (
                    sum(bounded_coverage) / len(bounded_coverage)
                    if bounded_coverage
                    else None
                ),
                "work_units_median": _linear_quantile(work, 0.5),
                "work_units_p90": _linear_quantile(work, 0.9),
                "selected_candidate_counts": selections,
            }
        )
    return summaries


def _nested_close(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_nested_close(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _nested_close(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    if isinstance(left, float) or isinstance(right, float):
        try:
            return _close(float(left), float(right))
        except (TypeError, ValueError):
            return False
    return left == right


def _derive_seed(key: dict[str, Any]) -> int:
    canonical = json.dumps(
        key, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    digest = hashlib.sha256(b"NPI-G11-SEED-V1\x00" + canonical).digest()
    seed = int.from_bytes(digest[:8], byteorder="big") & ((1 << 63) - 1)
    return seed or 1


def _qualification_gates(
    records: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    smoke: bool,
) -> dict[str, bool]:
    model_count = min(1, len(config.get("models", []))) if smoke else len(
        config.get("models", [])
    )
    clusters = 1 if smoke else int(config.get("sampling", {}).get("clusters", 0))
    expected_records = (
        model_count * len(config.get("tasks", {})) * clusters
    )
    expected_cells = model_count * len(config.get("tasks", {}))
    complete_matrix = len(records) == expected_records and expected_records > 0
    complete_aggregates = len(aggregates) == expected_cells and expected_cells > 0
    thresholds = config.get(
        "qualification_gates",
        {
            "minimum_empirical_target_attainment": 0.0,
            "maximum_relative_rmse_ratio": 10.0,
            "minimum_combined_asymptotic_coverage": 0.0,
        }
        if smoke
        else {},
    )
    required_thresholds = {
        "minimum_empirical_target_attainment",
        "maximum_relative_rmse_ratio",
        "minimum_combined_asymptotic_coverage",
    }
    if not required_thresholds.issubset(thresholds):
        return {"audit_config_has_qualification_thresholds": False}
    all_complete = complete_matrix and all(record["result"]["complete"] for record in records)
    return {
        "complete_cluster_matrix": complete_matrix and complete_aggregates,
        "no_resource_censoring": complete_matrix
        and all(not record["result"]["resource_censored"] for record in records),
        "all_runs_complete": all_complete,
        "all_design_targets_attained": all_complete
        and all(record["result"]["design_target_attained"] for record in records),
        "minimum_empirical_target_attainment": complete_aggregates
        and all(
            isinstance(item["empirical_target_attainment_fraction"], (int, float))
            and item["empirical_target_attainment_fraction"]
            >= float(thresholds["minimum_empirical_target_attainment"])
            for item in aggregates
        ),
        "across_cluster_relative_rmse": complete_aggregates
        and all(
            isinstance(item["empirical_relative_rmse_against_reference"], (int, float))
            and item["empirical_relative_rmse_against_reference"]
            <= item["requested_relative_sampling_rmse"]
            * float(thresholds["maximum_relative_rmse_ratio"])
            for item in aggregates
        ),
        "minimum_combined_asymptotic_coverage": complete_aggregates
        and all(
            isinstance(item["combined_asymptotic_95_coverage"], (int, float))
            and item["combined_asymptotic_95_coverage"]
            >= float(thresholds["minimum_combined_asymptotic_coverage"])
            for item in aggregates
        ),
        "selection_frozen_before_final": complete_matrix
        and all(
            record["selection"]["stopped"]
            and record["selection"]["frozen_decision"] is not None
            for record in records
        ),
        "no_final_samples_reused_from_selection": complete_matrix
        and all(record["seed_role_audit"]["selection_final_disjoint"] for record in records),
        "all_preparation_hashes_unique": complete_matrix
        and len({record["preparation"]["preparation_hash"] for record in records})
        == len(records),
    }


def audit(result_path: Path, config_path: Path) -> dict[str, Any]:
    result_bytes = result_path.read_bytes()
    result = json.loads(result_bytes)
    smoke = bool(result.get("smoke"))
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    failures: list[str] = []
    if result.get("schema") != "npi.g11.v5-confirmatory.v1":
        failures.append("unsupported result schema")
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    if result.get("config_sha256") != config_hash:
        failures.append("config hash mismatch")
    if result.get("protocol_id") != config.get("protocol_id"):
        failures.append("protocol id mismatch")
    records = result.get("records")
    if not isinstance(records, list) or not records:
        failures.append("result has no records")
        records = []

    ledger = result.get("seed_ledger", {})
    ledger_records = ledger.get("records", []) if isinstance(ledger, dict) else []
    if not isinstance(ledger, dict) or ledger.get(
        "schema"
    ) != "npi.g11.seed-ledger.v1" or not isinstance(ledger_records, list):
        failures.append("invalid or missing seed ledger")
        ledger = {"schema": "invalid", "records": []}
        ledger_records = []
    canonical_ledger = json.dumps(
        ledger, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    if hashlib.sha256(canonical_ledger).hexdigest() != result.get("seed_ledger_sha256"):
        failures.append("seed ledger hash mismatch")
    seen_seed_keys: set[str] = set()
    seen_seed_values: set[int] = set()
    for item in ledger_records:
        if not isinstance(item, dict) or set(item) != {"key", "seed"}:
            failures.append("malformed seed ledger record")
            continue
        key = item["key"]
        seed = item["seed"]
        if not isinstance(key, dict) or not isinstance(seed, int):
            failures.append("malformed seed ledger fields")
            continue
        canonical_key = json.dumps(key, sort_keys=True, separators=(",", ":"))
        if canonical_key in seen_seed_keys or seed in seen_seed_values:
            failures.append("duplicate seed key or value")
        seen_seed_keys.add(canonical_key)
        seen_seed_values.add(seed)
        if seed != _derive_seed(key):
            failures.append("seed does not match its canonical key")

    seen_ids: set[str] = set()
    preparation_hashes: set[str] = set()
    for record in records:
        cell_id = str(record.get("cell_id"))
        prefix = f"{cell_id}: "
        if cell_id in seen_ids:
            failures.append(prefix + "duplicate cell id")
        seen_ids.add(cell_id)
        model_id = str(record.get("model_id"))
        task_name = str(record.get("task_name"))
        expected_reference = None
        if isinstance(config.get("references"), dict):
            expected_reference = config["references"].get(model_id, {}).get(task_name)
        if smoke and expected_reference is None:
            task_spec = config.get("tasks", {}).get(task_name, {})
            if {"reference_probability", "reference_standard_error"}.issubset(task_spec):
                expected_reference = {
                    "probability": float(task_spec["reference_probability"]),
                    "standard_error": float(task_spec["reference_standard_error"]),
                }
        if not isinstance(expected_reference, dict) or not _nested_close(
            record.get("reference"), expected_reference
        ):
            failures.append(prefix + "model/task reference mismatch")
        cell_parts = cell_id.rsplit(":", 1)
        if len(cell_parts) != 2 or not cell_parts[1].startswith("cluster-"):
            failures.append(prefix + "malformed cell id")
            cell_cluster = ""
        else:
            cell_cluster = cell_parts[1]
        regime = f"{model_id}:{cell_cluster}"
        cell_seed_records = [
            item
            for item in ledger_records
            if item.get("key", {}).get("regime") == regime
            and item.get("key", {}).get("task") == task_name
        ]
        selection_seeds = {
            int(item["seed"])
            for item in cell_seed_records
            if item.get("key", {}).get("role") == "selection"
        }
        final_seeds = {
            int(item["seed"])
            for item in cell_seed_records
            if item.get("key", {}).get("role") == "final"
        }
        unexpected_roles = sorted(
            {
                str(item.get("key", {}).get("role"))
                for item in cell_seed_records
                if item.get("key", {}).get("role") not in {"selection", "final"}
            }
        )
        seed_audit = record.get("seed_role_audit", {})
        if int(seed_audit.get("selection_seed_count", -1)) != len(selection_seeds):
            failures.append(prefix + "selection seed count mismatch")
        if int(seed_audit.get("final_seed_count", -1)) != len(final_seeds):
            failures.append(prefix + "final seed count mismatch")
        recomputed_disjoint = selection_seeds.isdisjoint(final_seeds) and not unexpected_roles
        if bool(seed_audit.get("selection_final_disjoint")) != recomputed_disjoint:
            failures.append(prefix + "selection/final seed separation mismatch")
        if seed_audit.get("unexpected_roles") != unexpected_roles:
            failures.append(prefix + "unexpected seed roles mismatch")
        selection = record.get("selection", {})
        frozen = selection.get("frozen_decision")
        preparation = record.get("preparation", {})
        final = record.get("result", {})
        if not selection.get("stopped") or not isinstance(frozen, dict):
            failures.append(prefix + "selection was not frozen")
            continue
        selected = frozen.get("selected_candidate")
        if selected != preparation.get("selected_candidate") or selected != final.get(
            "selected_candidate"
        ):
            failures.append(prefix + "selected candidate changed after freeze")
        survivors = frozen.get("surviving_candidates", [])
        if selected not in survivors:
            failures.append(prefix + "selected candidate is not a survivor")
        candidate_work = {
            item["candidate_id"]: item for item in selection.get("candidate_work", [])
        }
        if selected not in candidate_work:
            failures.append(prefix + "selected candidate lacks a work interval")
        elif not _close(
            float(frozen["selected_point_work"]),
            float(candidate_work[selected]["point_total_work"]),
        ):
            failures.append(prefix + "selected point work is inconsistent")

        allocations = preparation.get("allocations", [])
        if not allocations:
            failures.append(prefix + "allocation is empty")
            continue
        design = math.fsum(
            float(item["design_variance"]) / int(item["final_count"]) for item in allocations
        )
        if not _close(design, float(final["design_sampling_variance"])):
            failures.append(prefix + "design variance arithmetic mismatch")
        target_probability = float(record["task"]["nominal_probability"])
        relative_target = float(final["requested_relative_sampling_rmse"])
        variance_target = (target_probability * relative_target) ** 2
        expected_design_gate = design <= variance_target * (1.0 + 1e-12)
        if bool(final["design_target_attained"]) != expected_design_gate:
            failures.append(prefix + "design target gate mismatch")

        resource_censored = bool(final["resource_censored"])
        if resource_censored != bool(preparation["resource_censored"]):
            failures.append(prefix + "resource censoring changed after preparation")
        if bool(final["complete"]):
            terms = final.get("terms", [])
            if len(terms) != len(allocations):
                failures.append(prefix + "term/allocation length mismatch")
            else:
                estimate = math.fsum(float(item["mean"]) for item in terms)
                empirical = math.fsum(
                    float(item["variance"]) / int(item["count"]) for item in terms
                )
                if not _close(estimate, float(final["estimate"])):
                    failures.append(prefix + "estimate does not telescope")
                if not _close(empirical, float(final["empirical_sampling_variance"])):
                    failures.append(prefix + "empirical variance arithmetic mismatch")
                expected_empirical_gate = empirical <= variance_target
                if bool(final["empirical_target_attained"]) != expected_empirical_gate:
                    failures.append(prefix + "empirical target gate mismatch")
                for allocation, term in zip(allocations, terms, strict=True):
                    if int(allocation["final_count"]) != int(term["count"]):
                        failures.append(prefix + "executed count differs from frozen allocation")

        work_entries = final.get("work", {}).get("entries", [])
        final_work = math.fsum(
            float(item["work_units"]) for item in work_entries if item["role"] == "final"
        )
        expected_final_work = math.fsum(
            int(item["final_count"]) * float(item["cost_per_sample"]) for item in allocations
        )
        if bool(final["complete"]) and not _close(final_work, expected_final_work):
            failures.append(prefix + "final work ledger mismatch")
        preparation_hash = str(preparation.get("preparation_hash"))
        if preparation_hash in preparation_hashes:
            failures.append(prefix + "duplicate preparation hash")
        preparation_hashes.add(preparation_hash)
        if final.get("preparation_hash") != preparation_hash:
            failures.append(prefix + "final preparation hash mismatch")

    recomputed_aggregates = _aggregate_records(records)
    if not _nested_close(result.get("aggregates"), recomputed_aggregates):
        failures.append("across-cluster aggregates do not match independent recomputation")
    recomputed_gates = _qualification_gates(
        records, recomputed_aggregates, config, smoke=smoke
    )
    if result.get("gates") != recomputed_gates:
        failures.append("top-level gates do not match independently recomputed gates")
    formal_readiness = result.get("formal_readiness", {})
    expected_complete = all(recomputed_gates.values()) and (
        smoke or all(formal_readiness.values())
    )
    if bool(result.get("protocol_complete")) != expected_complete:
        failures.append("protocol_complete does not match the gate conjunction")
    expected_qualification = result.get("run_class") == "qualification" and expected_complete
    if bool(result.get("qualification_passed")) != expected_qualification:
        failures.append("qualification_passed does not match run class and gates")
    return {
        "schema": "npi.g11.v5-result-audit.v1",
        "result_path": str(result_path),
        "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "config_path": str(config_path),
        "config_sha256": config_hash,
        "record_count": len(records),
        "recomputed_aggregates": recomputed_aggregates,
        "recomputed_gates": recomputed_gates,
        "failures": failures,
        "passed": not failures,
        "independence_note": (
            "The audit reimplements arithmetic and gates and does not import the "
            "confirmatory runner or production summary functions."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = audit(arguments.result, arguments.config)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
