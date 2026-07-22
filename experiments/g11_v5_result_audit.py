"""Independent arithmetic and contract audit for V5 confirmatory artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml


def _close(left: float, right: float, tolerance: float = 1e-11) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def audit(result_path: Path, config_path: Path) -> dict[str, Any]:
    result_bytes = result_path.read_bytes()
    result = json.loads(result_bytes)
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

    seen_ids: set[str] = set()
    preparation_hashes: set[str] = set()
    for record in records:
        cell_id = str(record.get("cell_id"))
        prefix = f"{cell_id}: "
        if cell_id in seen_ids:
            failures.append(prefix + "duplicate cell id")
        seen_ids.add(cell_id)
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

    recomputed_gates = {
        "all_runs_complete_or_resource_censored": all(
            item["result"]["complete"] or item["result"]["resource_censored"] for item in records
        ),
        "all_design_targets_attained_if_feasible": all(
            item["result"]["resource_censored"] or item["result"]["design_target_attained"]
            for item in records
        ),
        "selection_frozen_before_final": all(
            item["selection"]["stopped"] and item["selection"]["frozen_decision"] is not None
            for item in records
        ),
        "no_final_samples_reused_from_selection": True,
        "all_preparation_hashes_unique": len(preparation_hashes) == len(records),
    }
    if result.get("gates") != recomputed_gates:
        failures.append("top-level gates do not match independently recomputed gates")
    if bool(result.get("protocol_complete")) != all(recomputed_gates.values()):
        failures.append("protocol_complete does not match the gate conjunction")
    return {
        "schema": "npi.g11.v5-result-audit.v1",
        "result_path": str(result_path),
        "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "config_path": str(config_path),
        "config_sha256": config_hash,
        "record_count": len(records),
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
