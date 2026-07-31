"""Reconstruct and audit the exact-count V6 resource-cap amendment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.g11_v8_p5_sharded_reference_common import load_context
from src.path_integral.reference_aggregation import build_allocation_manifest
from src.path_integral.reference_protocol import canonical_sha256
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting

AUDIT_SCHEMA = "npi.g11.v8-p5-reference-allocation-cap-amendment-audit.v1"
EXPECTED_RECEIPT_SCHEMA = "npi.g11.v8-p5-reference-allocation-cap-amendment.v1"
EXPECTED_PACKAGE_SCHEMA = "npi.g11.v8-p5-reference-pilot-package.v1"
EXPECTED_FAILURE_SCHEMA = "npi.g11.v8-p5-reference-allocation-failure.v1"
EXPECTED_CAPS = {"dcs_reference": 134_217_728, "raw_crosscheck": 33_554_432}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("allocation-amendment input must be a JSON mapping")
    return value


def reconstruct_amended_allocation(
    config_path: Path,
    package_path: Path,
    failure_receipt_path: Path,
    amendment_receipt_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    context = load_context(config_path)
    package = _load_json(package_path)
    failure = _load_json(failure_receipt_path)
    amendment = _load_json(amendment_receipt_path)
    if (
        package.get("schema") != EXPECTED_PACKAGE_SCHEMA
        or failure.get("schema") != EXPECTED_FAILURE_SCHEMA
        or amendment.get("schema") != EXPECTED_RECEIPT_SCHEMA
        or package.get("protocol_id") != context.config["protocol_id"]
        or failure.get("protocol_id") != context.config["protocol_id"]
        or amendment.get("protocol_id") != context.config["protocol_id"]
        or package.get("config_sha256") != context.config_sha256
        or failure.get("config_sha256") != context.config_sha256
        or amendment.get("config_sha256") != context.config_sha256
        or amendment.get("pilot_package_sha256") != canonical_sha256(package)
        or amendment.get("failure_receipt_sha256") != canonical_sha256(failure)
    ):
        raise ValueError("allocation-amendment bindings are inconsistent")
    shard_records = package.get("shards")
    failure_entries = failure.get("entries")
    if (
        not isinstance(shard_records, list)
        or len(shard_records) != 384
        or not isinstance(failure_entries, list)
        or len(failure_entries) != 48
    ):
        raise ValueError("allocation-amendment pilot or entry roster is incomplete")
    targets: dict[str | tuple[str, str], float] = {
        (str(entry["cell_id"]), str(entry["method"])): float(
            entry["target_standard_error"]
        )
        for entry in failure_entries
    }
    pilot_records = [
        (record["artifact"], record["sha256"]) for record in shard_records
    ]
    sampling = context.config["sampling"]
    contract = context.config["reference_contract"]
    manifest = build_allocation_manifest(
        protocol_id=context.config["protocol_id"],
        config_sha256=context.config_sha256,
        threshold_manifest_sha256=context.binding["threshold_manifest_sha256"],
        pilot_parent_sha256=context.reference_parent_sha256,
        pilot_namespace=sampling["pilot_namespace"],
        final_namespace=sampling["final_namespace"],
        expected_cells=list(context.cells_by_id),
        expected_methods=contract["methods"],
        pilot_replicates=int(sampling["pilot_replicates"]),
        pilot_shards=pilot_records,
        target_standard_errors=targets,
        allocation_safety_factor=float(sampling["allocation_safety_factor"]),
        minimum_final_samples=int(sampling["minimum_final_samples"]),
        maximum_final_samples=int(sampling["maximum_final_samples"]),
        final_chunk_size=int(sampling["final_chunk_size"]),
        source_commit=str(package["source_commit"]),
        environment_sha256=str(package["environment_sha256"]),
        estimand=str(contract["estimand"]),
        dtype=str(contract["dtype"]),
        device=str(contract["device"]),
        design_informed_by_prior_development_outcomes=True,
        current_namespace_outcomes_inspected_before_freeze=False,
        maximum_final_samples_by_method=EXPECTED_CAPS,
    )
    return manifest, amendment


def audit_amendment(
    config_path: Path,
    package_path: Path,
    failure_receipt_path: Path,
    amendment_receipt_path: Path,
) -> dict[str, Any]:
    manifest, amendment = reconstruct_amended_allocation(
        config_path,
        package_path,
        failure_receipt_path,
        amendment_receipt_path,
    )
    failure = _load_json(failure_receipt_path)
    old_counts = {
        (entry["cell_id"], entry["method"]): entry["requested_final_samples"]
        for entry in failure["entries"]
    }
    new_counts = {
        (entry["cell_id"], entry["method"]): entry["requested_final_samples"]
        for entry in manifest["entries"]
    }
    trigger_key = (
        amendment.get("trigger_cell_id"),
        amendment.get("trigger_method"),
    )
    trigger_count = int(amendment.get("trigger_requested_final_samples", 0))
    derived_raw_cap = 1 << (trigger_count - 1).bit_length() if trigger_count else 0
    checks = {
        "allocation_canonical_hash_exact": canonical_sha256(manifest)
        == amendment.get("allocation_manifest_sha256"),
        "pilot_and_statistical_counts_unchanged": old_counts == new_counts
        and amendment.get("pilot_statistics_changed") is False
        and amendment.get("statistical_targets_changed") is False
        and amendment.get("requested_final_sample_counts_changed") is False
        and amendment.get("proposal_manifest_changed") is False,
        "minimal_power_of_two_cap_rule_exact": amendment.get("cap_rule")
        == "smallest_power_of_two_not_less_than_frozen_requested_count"
        and old_counts.get(trigger_key) == trigger_count
        and derived_raw_cap == EXPECTED_CAPS["raw_crosscheck"]
        and amendment.get("amended_method_caps") == EXPECTED_CAPS,
        "complete_feasible_allocation": manifest.get("all_resources_feasible") is True
        and manifest.get("final_execution_authorized") is True
        and len(manifest.get("entries", [])) == 48
        and amendment.get("total_requested_final_samples")
        == sum(new_counts.values())
        and amendment.get("total_final_chunks")
        == sum(len(entry["chunks"]) for entry in manifest["entries"]),
        "final_namespace_unopened": amendment.get(
            "final_namespace_opened_before_amendment"
        )
        is False
        and amendment.get("final_namespace") == manifest["final_namespace"],
        "hardware_and_claims_fail_closed": amendment.get(
            "hardware_execution_authorized"
        )
        is False
        and amendment.get("performance_claim_authorized") is False
        and amendment.get("submission_authorized") is False,
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    return {
        "schema": AUDIT_SCHEMA,
        "config_sha256": manifest["config_sha256"],
        "allocation_manifest_sha256": canonical_sha256(manifest),
        "amendment_receipt_sha256": canonical_sha256(amendment),
        "checks": checks,
        "failures": failures,
        "passed": not failures,
        "decision": {
            "status": (
                "exact_count_cap_amendment_audit_pass"
                if not failures
                else "exact_count_cap_amendment_audit_fail"
            ),
            "statistical_allocation_complete": not failures,
            "hardware_execution_authorized": False,
            "final_reference_complete": False,
            "performance_claim_authorized": False,
            "submission_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("write-manifest", "audit"),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--failure-receipt", type=Path, required=True)
    parser.add_argument("--amendment-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "write-manifest":
        manifest, _ = reconstruct_amended_allocation(
            arguments.config,
            arguments.package,
            arguments.failure_receipt,
            arguments.amendment_receipt,
        )
        digest = write_json_atomic_nonoverwriting(arguments.output, manifest)
        print(
            json.dumps(
                {
                    "allocation_manifest_sha256": digest,
                    "all_resources_feasible": manifest["all_resources_feasible"],
                    "final_execution_authorized": manifest[
                        "final_execution_authorized"
                    ],
                },
                sort_keys=True,
            )
        )
        return
    report = audit_amendment(
        arguments.config,
        arguments.package,
        arguments.failure_receipt,
        arguments.amendment_receipt,
    )
    digest = write_json_atomic_nonoverwriting(arguments.output, report)
    print(json.dumps({"audit_sha256": digest, **report["decision"]}, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
