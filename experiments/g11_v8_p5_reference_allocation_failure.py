"""Compact, independently reproducible receipt for a failed R2 allocation gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.g11_v8_p5_sharded_reference_common import load_context
from src.path_integral.reference_aggregation import (
    build_allocation_manifest,
    validate_allocation_manifest,
)
from src.path_integral.reference_protocol import canonical_sha256
from src.path_integral.reference_shards import (
    find_completed_shards,
    write_json_atomic_nonoverwriting,
)

PACKAGE_SCHEMA = "npi.g11.v8-p5-reference-pilot-package.v1"
RECEIPT_SCHEMA = "npi.g11.v8-p5-reference-allocation-failure.v1"
AUDIT_SCHEMA = "npi.g11.v8-p5-reference-allocation-failure-audit.v1"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError("allocation-failure input must be a mapping")
    return value


def _reconstruct_manifest(
    context: Any,
    package: dict[str, Any],
    template: dict[str, Any],
) -> dict[str, Any]:
    shards = package.get("shards")
    if not isinstance(shards, list):
        raise ValueError("pilot package shards must be a list")
    pilot_records = [
        (record["artifact"], record["sha256"]) for record in shards
    ]
    entries = template["entries"]
    targets: dict[str | tuple[str, str], float] = {
        (str(entry["cell_id"]), str(entry["method"])): float(
            entry["target_standard_error"]
        )
        for entry in entries
    }
    method_caps: dict[str, int] = {}
    for entry in entries:
        method = str(entry["method"])
        cap = int(entry["maximum_final_samples"])
        prior = method_caps.setdefault(method, cap)
        if prior != cap:
            raise ValueError("allocation entries disagree on a method-specific cap")
    return build_allocation_manifest(
        protocol_id=template["protocol_id"],
        config_sha256=template["config_sha256"],
        threshold_manifest_sha256=template["threshold_manifest_sha256"],
        pilot_parent_sha256=template["pilot_parent_sha256"],
        pilot_namespace=template["pilot_namespace"],
        final_namespace=template["final_namespace"],
        expected_cells=template["expected_cells"],
        expected_methods=template["expected_methods"],
        pilot_replicates=int(template["pilot_replicates"]),
        pilot_shards=pilot_records,
        target_standard_errors=targets,
        allocation_safety_factor=float(template["allocation_safety_factor"]),
        minimum_final_samples=int(template["minimum_final_samples"]),
        maximum_final_samples=int(template["maximum_final_samples"]),
        final_chunk_size=int(template["final_chunk_size"]),
        source_commit=template["source_commit"],
        environment_sha256=template["environment_sha256"],
        estimand=template["estimand"],
        dtype=template["dtype"],
        device=template["device"],
        design_informed_by_prior_development_outcomes=template[
            "design_informed_by_prior_development_outcomes"
        ],
        current_namespace_outcomes_inspected_before_freeze=template[
            "current_namespace_outcomes_inspected_before_freeze"
        ],
        maximum_final_samples_by_method=method_caps,
    )


def build_failure_evidence(
    config_path: Path,
    pilot_directory: Path,
    allocation_path: Path,
    package_path: Path,
    receipt_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    context = load_context(config_path)
    allocation = _load_json(allocation_path)
    validate_allocation_manifest(allocation)
    completed = find_completed_shards(pilot_directory)
    shard_records = [
        {"sha256": digest, "artifact": artifact}
        for _, artifact, digest in sorted(
            completed.values(), key=lambda item: item[1]["shard_id"]
        )
    ]
    package = {
        "schema": PACKAGE_SCHEMA,
        "protocol_id": context.config["protocol_id"],
        "config_sha256": context.config_sha256,
        "threshold_binding_sha256": context.binding_sha256,
        "threshold_manifest_sha256": context.binding["threshold_manifest_sha256"],
        "source_commit": allocation["source_commit"],
        "environment_sha256": allocation["environment_sha256"],
        "pilot_namespace": allocation["pilot_namespace"],
        "reference_parent_sha256": allocation["pilot_parent_sha256"],
        "pilot_shard_count": len(shard_records),
        "shards": shard_records,
        "formal_reference_complete": False,
        "performance_claim_authorized": False,
    }
    package_sha256 = write_json_atomic_nonoverwriting(package_path, package)
    reconstructed = _reconstruct_manifest(context, package, allocation)
    allocation_sha256 = canonical_sha256(allocation)
    if canonical_sha256(reconstructed) != allocation_sha256:
        raise RuntimeError("pilot package does not reconstruct the allocation exactly")
    summaries = [
        {
            "cell_id": entry["cell_id"],
            "method": entry["method"],
            "target_standard_error": entry["target_standard_error"],
            "pilot_variances": entry["pilot_variances"],
            "allocation_design_variance": entry["allocation_design_variance"],
            "requested_final_samples": entry["requested_final_samples"],
            "maximum_final_samples": entry["maximum_final_samples"],
            "requested_to_cap_ratio": (
                entry["requested_final_samples"] / entry["maximum_final_samples"]
            ),
            "resource_feasible": entry["resource_feasible"],
            "authorized_final_samples": entry["authorized_final_samples"],
            "chunk_count_if_feasible": len(entry["chunks"]),
        }
        for entry in allocation["entries"]
    ]
    infeasible = [entry for entry in summaries if not entry["resource_feasible"]]
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "protocol_id": context.config["protocol_id"],
        "config_sha256": context.config_sha256,
        "threshold_binding_sha256": context.binding_sha256,
        "pilot_package_sha256": package_sha256,
        "allocation_manifest_sha256": allocation_sha256,
        "source_commit": allocation["source_commit"],
        "environment_sha256": allocation["environment_sha256"],
        "reference_parent_sha256": allocation["pilot_parent_sha256"],
        "pilot_shard_count": len(shard_records),
        "entry_count": len(summaries),
        "resource_feasible_entry_count": len(summaries) - len(infeasible),
        "resource_infeasible_entry_count": len(infeasible),
        "total_requested_final_samples": sum(
            int(entry["requested_final_samples"]) for entry in summaries
        ),
        "maximum_requested_to_cap_ratio": max(
            float(entry["requested_to_cap_ratio"]) for entry in summaries
        ),
        "entries": summaries,
        "gates": {
            "complete_pilot_roster": len(shard_records) == 384,
            "allocation_exactly_reconstructed": True,
            "all_resources_feasible": allocation["all_resources_feasible"],
            "final_execution_authorized": allocation[
                "final_execution_authorized"
            ],
        },
        "decision": {
            "status": "r2_development_allocation_resource_failure",
            "pilot_namespace_burned": True,
            "final_namespace_opened": False,
            "final_execution_authorized": False,
            "reference_complete": False,
            "new_reference_design_required": True,
            "performance_claim_authorized": False,
            "submission_authorized": False,
        },
    }
    write_json_atomic_nonoverwriting(receipt_path, receipt)
    return package, receipt


def audit_failure_evidence(
    config_path: Path,
    package_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    context = load_context(config_path)
    package = _load_json(package_path)
    receipt = _load_json(receipt_path)
    sampling = context.config["sampling"]
    contract = context.config["reference_contract"]
    template = {
        "protocol_id": context.config["protocol_id"],
        "config_sha256": context.config_sha256,
        "threshold_manifest_sha256": context.binding["threshold_manifest_sha256"],
        "pilot_parent_sha256": receipt.get(
            "reference_parent_sha256", context.reference_parent_sha256
        ),
        "pilot_namespace": sampling["pilot_namespace"],
        "final_namespace": sampling["final_namespace"],
        "expected_cells": list(context.cells_by_id),
        "expected_methods": list(context.config["reference_contract"]["methods"]),
        "pilot_replicates": sampling["pilot_replicates"],
        "allocation_safety_factor": sampling["allocation_safety_factor"],
        "minimum_final_samples": sampling["minimum_final_samples"],
        "maximum_final_samples": sampling["maximum_final_samples"],
        "final_chunk_size": sampling["final_chunk_size"],
        "source_commit": receipt["source_commit"],
        "environment_sha256": receipt["environment_sha256"],
        "estimand": contract["estimand"],
        "dtype": contract["dtype"],
        "device": contract["device"],
        "design_informed_by_prior_development_outcomes": context.config[
            "design_informed_by_prior_development_outcomes"
        ],
        "current_namespace_outcomes_inspected_before_freeze": context.config[
            "current_namespace_outcomes_inspected_before_freeze"
        ],
        "entries": receipt["entries"],
    }
    try:
        reconstructed = _reconstruct_manifest(context, package, template)
        reconstructed_sha256 = canonical_sha256(reconstructed)
    except (KeyError, TypeError, ValueError):
        reconstructed_sha256 = None
    shards = package.get("shards")
    package_shard_ids = [
        record.get("artifact", {}).get("shard_id")
        for record in shards
        if isinstance(record, dict)
    ] if isinstance(shards, list) else []
    entries = receipt.get("entries")
    checks = {
        "schemas_exact": package.get("schema") == PACKAGE_SCHEMA
        and receipt.get("schema") == RECEIPT_SCHEMA,
        "config_and_binding_exact": package.get("config_sha256")
        == receipt.get("config_sha256")
        == context.config_sha256
        and package.get("threshold_binding_sha256")
        == receipt.get("threshold_binding_sha256")
        == context.binding_sha256,
        "package_hash_exact": canonical_sha256(package)
        == receipt.get("pilot_package_sha256"),
        "allocation_exactly_reconstructed": reconstructed_sha256
        == receipt.get("allocation_manifest_sha256"),
        "complete_unique_pilot_roster": package.get("pilot_shard_count") == 384
        and receipt.get("pilot_shard_count") == 384
        and len(package_shard_ids) == 384
        and len(set(package_shard_ids)) == 384,
        "entry_matrix_exact": isinstance(entries, list)
        and len(entries) == 48
        and {
            (entry.get("cell_id"), entry.get("method"))
            for entry in entries
            if isinstance(entry, dict)
        }
        == {
            (cell_id, method)
            for cell_id in context.cells_by_id
            for method in ("dcs_reference", "raw_crosscheck")
        },
        "resource_failure_nonempty": isinstance(entries, list)
        and receipt.get("resource_infeasible_entry_count", 0) > 0
        and receipt.get("resource_infeasible_entry_count")
        == sum(
            not bool(entry.get("resource_feasible"))
            for entry in entries
            if isinstance(entry, dict)
        )
        and receipt.get("resource_feasible_entry_count")
        == sum(
            bool(entry.get("resource_feasible"))
            for entry in entries
            if isinstance(entry, dict)
        )
        and receipt.get("total_requested_final_samples")
        == sum(
            int(entry.get("requested_final_samples", 0))
            for entry in entries
            if isinstance(entry, dict)
        ),
        "final_gate_closed": receipt.get("gates", {}).get(
            "all_resources_feasible"
        )
        is False
        and receipt.get("gates", {}).get("final_execution_authorized") is False
        and receipt.get("decision", {}).get("final_execution_authorized") is False
        and receipt.get("decision", {}).get("new_reference_design_required") is True
        and receipt.get("decision", {}).get("performance_claim_authorized") is False,
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    return {
        "schema": AUDIT_SCHEMA,
        "pilot_package_sha256": canonical_sha256(package),
        "failure_receipt_sha256": canonical_sha256(receipt),
        "checks": checks,
        "failures": failures,
        "passed": not failures,
        "decision": receipt.get("decision"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--pilot-directory", type=Path, required=True)
    build.add_argument("--allocation", type=Path, required=True)
    build.add_argument("--package-output", type=Path, required=True)
    build.add_argument("--receipt-output", type=Path, required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--config", type=Path, required=True)
    audit.add_argument("--package", type=Path, required=True)
    audit.add_argument("--receipt", type=Path, required=True)
    audit.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "build":
        _, receipt = build_failure_evidence(
            arguments.config,
            arguments.pilot_directory,
            arguments.allocation,
            arguments.package_output,
            arguments.receipt_output,
        )
        print(json.dumps(receipt["decision"], sort_keys=True))
        return
    report = audit_failure_evidence(
        arguments.config, arguments.package, arguments.receipt
    )
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if arguments.output is not None:
        if arguments.output.exists():
            raise FileExistsError(
                f"refusing to overwrite failure audit: {arguments.output}"
            )
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
