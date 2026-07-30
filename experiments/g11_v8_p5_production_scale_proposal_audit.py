"""Audit the production-scale proposal falsification and block-order defect."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from experiments.g11_v8_p5_production_scale_proposal import (
    DCS_METHOD,
    EXPECTED_REQUIREMENTS,
    RAW_METHOD,
    load_production_scale_config,
)

AUDIT_SCHEMAS = {
    2: "npi.g11.v8-p5-production-scale-proposal-audit.v1",
    3: "npi.g11.v8-p5-production-scale-proposal-audit.v2",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _all_json_finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_json_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_json_finite(item) for item in value)
    return True


def _mean_block_counts(candidate: dict[str, Any]) -> list[float]:
    rows = [
        replicate.get("raw_block_nonzero_counts")
        for replicate in candidate.get("replicates", [])
    ]
    if (
        not rows
        or any(not isinstance(row, list) or not row for row in rows)
        or len({len(row) for row in rows}) != 1
    ):
        raise ValueError("raw candidate block-count matrix is malformed")
    return [
        sum(float(row[index]) for row in rows) / len(rows)
        for index in range(len(rows[0]))
    ]


def audit_production_scale_result(
    config_path: Path, result_path: Path
) -> dict[str, Any]:
    config, config_sha256 = load_production_scale_config(config_path)
    raw = result_path.read_bytes()
    result = json.loads(raw.decode("utf-8"))
    if not isinstance(result, dict):
        raise ValueError("production-scale result must be a mapping")
    result_schema = result.get("schema")
    if not isinstance(result_schema, str):
        raise ValueError("production-scale result schema is malformed")
    version = {
        "npi.g11.v8-p5-production-scale-proposal-result.v2": 2,
        "npi.g11.v8-p5-production-scale-proposal-result.v3": 3,
    }.get(result_schema)
    if version is None:
        raise ValueError("unsupported production-scale result audit version")
    candidates = result.get("candidates")
    seed_records = result.get("seed_records")
    if not isinstance(candidates, list) or not isinstance(seed_records, list):
        raise ValueError("production-scale candidates or seeds are malformed")
    candidate_ids = [
        str(candidate.get("candidate_id"))
        for candidate in candidates
        if isinstance(candidate, dict)
    ]
    raw_candidates = [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict) and candidate.get("method") == RAW_METHOD
    ]
    dcs_candidates = [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict) and candidate.get("method") == DCS_METHOD
    ]
    raw_cells = {
        cell_id for cell_id, method in EXPECTED_REQUIREMENTS if method == RAW_METHOD
    }
    dcs_cells = {
        cell_id for cell_id, method in EXPECTED_REQUIREMENTS if method == DCS_METHOD
    }
    grouped_signatures: dict[str, dict[str, Any]] = {}
    for candidate in raw_candidates:
        block_means = _mean_block_counts(candidate)
        smallest = min(block_means)
        ratio = max(block_means) / max(1.0, smallest)
        grouped_signatures[str(candidate["candidate_id"])] = {
            "mean_raw_nonzero_by_stored_block": block_means,
            "maximum_to_minimum_mean_count_ratio": ratio,
            "grouped_order_signature": ratio > 10.0,
        }
    seeds = [int(record["seed"]) for record in seed_records]
    candidate_gate_consistency = all(
        isinstance(candidate, dict)
        and isinstance(candidate.get("gates"), dict)
        and bool(candidate.get("passes"))
        == all(bool(value) for value in candidate["gates"].values())
        for candidate in candidates
    )
    expected_candidate_count = 24 if version == 2 else 35
    expected_raw_candidate_count = 15 if version == 2 else 20
    expected_dcs_candidate_count = 9 if version == 2 else 15
    expected_seed_count = 405 if version == 2 else 861
    passing_by_requirement: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate.get("passes") is True:
            key = (str(candidate["cell_id"]), str(candidate["method"]))
            passing_by_requirement.setdefault(key, []).append(candidate)
    expected_selected = {
        key: min(
            values,
            key=lambda candidate: float(candidate["requested_to_cap_ratio"]),
        )["candidate_id"]
        for key, values in passing_by_requirement.items()
    }
    actual_selected = {
        (str(cell_id), str(method)): selected["candidate_id"]
        for cell_id, methods in result.get("selected_proposals", {}).items()
        for method, selected in methods.items()
    }
    block_partition_pass = (
        all(
            signature["grouped_order_signature"]
            for signature in grouped_signatures.values()
        )
        if version == 2
        else all(
            float(signature["maximum_to_minimum_mean_count_ratio"]) < 2.0
            for signature in grouped_signatures.values()
        )
        and result.get("validation_design", {}).get("block_partition")
        == "independent_seeded_uniform_permutation_before_equal_slicing"
        and all(
            replicate.get("block_permutation_seed") is not None
            for candidate in candidates
            if isinstance(candidate, dict)
            for replicate in candidate.get("replicates", [])
        )
    )
    expected_selected_count = 0 if version == 2 else 3
    checks = {
        "strict_json_and_schema_exact": result.get("schema")
        == f"npi.g11.v8-p5-production-scale-proposal-result.v{version}"
        and _all_json_finite(result),
        "config_protocol_and_clean_source_exact": result.get("config_sha256")
        == config_sha256
        and result.get("protocol_id") == config["protocol_id"]
        and result.get("dirty_worktree") is False
        and isinstance(result.get("source_commit"), str)
        and len(result["source_commit"]) == 40,
        "candidate_roster_exact": len(candidates) == expected_candidate_count
        and len(candidate_ids) == len(set(candidate_ids))
        and len(raw_candidates) == expected_raw_candidate_count
        and len(dcs_candidates) == expected_dcs_candidate_count
        and {str(candidate["cell_id"]) for candidate in raw_candidates}
        == raw_cells
        and {str(candidate["cell_id"]) for candidate in dcs_candidates}
        == dcs_cells,
        "seed_ledger_unique": len(seeds) == expected_seed_count
        and len(seeds) == len(set(seeds)),
        "candidate_gates_self_consistent": candidate_gate_consistency,
        "falsification_exact": result.get("passed") is False
        and int(result.get("selected_count", -1)) == expected_selected_count
        and int(result.get("required_selection_count", -1)) == 8
        and actual_selected == expected_selected,
        "block_partition_diagnostic_exact": bool(grouped_signatures)
        and block_partition_pass,
        "decision_fail_closed": result.get("decision", {}).get(
            "proposal_manifest_build_authorized"
        )
        is False
        and result.get("decision", {}).get("new_formal_pilot_authorized") is False
        and result.get("decision", {}).get("final_execution_authorized") is False
        and result.get("decision", {}).get("performance_claim_authorized") is False
        and result.get("decision", {}).get("submission_authorized") is False,
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    return {
        "schema": AUDIT_SCHEMAS[version],
        "config_sha256": config_sha256,
        "result_file_sha256": _sha256(result_path),
        "checks": checks,
        "grouped_order_diagnostics": grouped_signatures,
        "failures": failures,
        "passed": not failures,
        "decision": {
            "status": (
                (
                    "production_scale_falsification_and_block_order_defect_confirmed"
                    if version == 2
                    else "permuted_block_partial_falsification_confirmed"
                )
                if not failures
                else "production_scale_audit_failure"
            ),
            f"v{version}_training_namespace_burned": True,
            f"v{version}_validation_namespace_burned": True,
            "permuted_block_protocol_required": version == 2,
            "proposal_weight_optimization_required": version == 3,
            "candidate_promoted": False,
            "proposal_manifest_build_authorized": False,
            "new_formal_pilot_authorized": False,
            "final_execution_authorized": False,
            "performance_claim_authorized": False,
            "submission_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(
            f"refusing to overwrite production-scale audit: {arguments.output}"
        )
    report = audit_production_scale_result(arguments.config, arguments.result)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"passed": report["passed"], **report["decision"]}))


if __name__ == "__main__":
    main()
