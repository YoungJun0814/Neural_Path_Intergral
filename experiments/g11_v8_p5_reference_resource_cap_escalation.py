"""Freeze and audit a proposal-invariant reference resource-cap escalation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from experiments.g11_v8_p5_sharded_reference_common import ROOT
from src.path_integral.provenance import source_provenance
from src.path_integral.reference_protocol import REFERENCE_METHODS, canonical_sha256
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting

CONFIG_SCHEMA = "npi.g11.v8-p5-reference-resource-cap-escalation.v1"
MANIFEST_SCHEMA = "npi.g11.v8-p5-reference-proposal-manifest.v3"
AUDIT_SCHEMA = "npi.g11.v8-p5-reference-proposal-manifest-audit.v3"
EXPECTED_PROTOCOL = "g11-v8-p5-sharded-reference-resource-cap-v1"
EXPECTED_PILOT = "v8-r2-reference-resource-cap-pilot-v1"
EXPECTED_FINAL = "v8-r2-reference-resource-cap-final-v1"
EXPECTED_TARGETS = {"dcs_reference": 0.02, "raw_crosscheck": 0.05}
EXPECTED_CAPS = {"dcs_reference": 134_217_728, "raw_crosscheck": 16_777_216}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bound_path(record: Any) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError("resource-cap binding is malformed")
    path = (ROOT / str(record["path"])).resolve()
    if ROOT not in path.parents or not path.is_file():
        raise ValueError("resource-cap bound path is invalid")
    if record["sha256"] != _sha256(path):
        raise ValueError("resource-cap bound artifact hash mismatch")
    return path


def _load_json(config: dict[str, Any], field: str) -> dict[str, Any]:
    value = json.loads(_bound_path(config[field]).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{field} must bind a JSON mapping")
    return value


def _next_power_of_two(value: int) -> int:
    if value < 1:
        raise ValueError("power-of-two input must be positive")
    return 1 << (value - 1).bit_length()


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(config, dict) or config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unexpected resource-cap escalation schema")
    for field in (
        "prior_manifest",
        "prior_manifest_audit",
        "failed_pilot_package",
        "failure_receipt",
        "failure_audit",
    ):
        _bound_path(config.get(field))
    protocol = config.get("reference_protocol")
    statistical = config.get("unchanged_statistical_contract")
    rule = config.get("resource_cap_rule")
    decision = config.get("decision")
    if (
        config.get("design_informed_by_prior_resource_failure") is not True
        or config.get("current_namespace_outcomes_inspected_before_freeze") is not False
        or not isinstance(protocol, dict)
        or protocol.get("id") != EXPECTED_PROTOCOL
        or protocol.get("pilot_namespace") != EXPECTED_PILOT
        or protocol.get("final_namespace") != EXPECTED_FINAL
        or protocol.get("methods") != list(REFERENCE_METHODS)
        or protocol.get("primary_method") != "dcs_reference"
        or protocol.get("estimand") != "fixed_finest_grid"
        or protocol.get("dtype") != "float64"
        or protocol.get("device") != "cpu"
        or not isinstance(statistical, dict)
        or statistical.get("method_relative_standard_error_targets")
        != EXPECTED_TARGETS
        or float(statistical.get("maximum_combined_agreement_z", 0.0)) != 4.0
        or int(statistical.get("pilot_replicates", 0)) != 8
        or int(statistical.get("pilot_samples_per_replicate", 0)) != 32768
        or float(statistical.get("allocation_safety_factor", 0.0)) != 6.0
        or statistical.get("allocation_variance_statistic")
        != "maximum_replicate_variance"
        or statistical.get("ordinary_mean_required") is not True
        or statistical.get("self_normalization_allowed") is not False
        or not isinstance(rule, dict)
        or rule.get("failed_cell_id")
        != "h0.12-discrete_lower_barrier-p1e-05"
        or rule.get("failed_method") != "dcs_reference"
        or int(rule.get("prior_requested_final_samples", 0)) != 59_685_859
        or int(rule.get("prior_maximum_final_samples", 0)) != 33_554_432
        or int(rule.get("multiplier", 0)) != 2
        or rule.get("rounding")
        != "smallest_power_of_two_not_less_than_multiplied_requirement"
        or int(rule.get("derived_dcs_maximum_final_samples", 0))
        != EXPECTED_CAPS["dcs_reference"]
        or int(rule.get("unchanged_raw_maximum_final_samples", 0))
        != EXPECTED_CAPS["raw_crosscheck"]
        or _next_power_of_two(
            int(rule["multiplier"]) * int(rule["prior_requested_final_samples"])
        )
        != int(rule["derived_dcs_maximum_final_samples"])
        or not isinstance(decision, dict)
        or decision.get("proposal_or_target_change_authorized") is not False
        or decision.get("manifest_build_authorized") is not True
        or decision.get("fresh_formal_pilot_required") is not True
        or any(
            decision.get(field) is not False
            for field in (
                "new_formal_pilot_authorized",
                "final_execution_authorized",
                "performance_claim_authorized",
                "submission_authorized",
            )
        )
    ):
        raise ValueError("resource-cap escalation contract is invalid")
    return config, hashlib.sha256(raw).hexdigest()


def build_manifest(config_path: Path) -> dict[str, Any]:
    config, config_sha256 = load_config(config_path)
    provenance = source_provenance()
    if provenance["dirty_worktree"]:
        raise RuntimeError("resource-cap manifest requires a clean Git worktree")
    prior = _load_json(config, "prior_manifest")
    prior_audit = _load_json(config, "prior_manifest_audit")
    receipt = _load_json(config, "failure_receipt")
    failure_audit = _load_json(config, "failure_audit")
    package = _load_json(config, "failed_pilot_package")
    rule = config["resource_cap_rule"]
    failed_entries = [
        entry
        for entry in receipt.get("entries", [])
        if isinstance(entry, dict) and not entry.get("resource_feasible")
    ]
    if (
        prior_audit.get("passed") is not True
        or failure_audit.get("passed") is not True
        or failure_audit.get("decision", {}).get("final_execution_authorized")
        is not False
        or package.get("pilot_shard_count") != 384
        or len(failed_entries) != 1
        or failed_entries[0].get("cell_id") != rule["failed_cell_id"]
        or failed_entries[0].get("method") != rule["failed_method"]
        or failed_entries[0].get("requested_final_samples")
        != rule["prior_requested_final_samples"]
        or failed_entries[0].get("maximum_final_samples")
        != rule["prior_maximum_final_samples"]
    ):
        raise ValueError("resource-cap escalation evidence is inconsistent")
    protocol = config["reference_protocol"]
    result = copy.deepcopy(prior)
    result.update(
        {
            "schema": MANIFEST_SCHEMA,
            "protocol_id": protocol["id"],
            "build_config_sha256": config_sha256,
            "pilot_namespace": protocol["pilot_namespace"],
            "final_namespace": protocol["final_namespace"],
            "primary_method": protocol["primary_method"],
            "resource_caps": {
                "maximum_final_samples_by_method": copy.deepcopy(EXPECTED_CAPS),
                "minimum_final_samples": 8192,
                "final_chunk_size": 4096,
                "allocation_safety_factor": 6.0,
            },
            "resource_cap_derivation": copy.deepcopy(rule),
            "design_informed_by_prior_resource_failure": True,
            "current_namespace_outcomes_inspected_before_freeze": False,
            "resource_escalation_only": True,
            "fresh_pilot_required": True,
            "final_execution_authorized": False,
            "performance_claim_authorized": False,
            "submission_authorized": False,
            **provenance,
        }
    )
    return result


def _proposal_projection(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            key: copy.deepcopy(entry.get(key))
            for key in (
                "cell_id",
                "method",
                "entry_id",
                "weights",
                "schedules",
                "source_kind",
                "source_id",
                "selection_status",
                "overrides_v4_resource_infeasible_entry",
                "development_requested_to_original_cap_ratio",
                "development_gates",
            )
        }
        for entry in manifest.get("entries", [])
        if isinstance(entry, dict)
    ]


def audit_manifest(
    config_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    config, config_sha256 = load_config(config_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prior = _load_json(config, "prior_manifest")
    if not isinstance(manifest, dict):
        raise ValueError("resource-cap proposal manifest must be a mapping")
    entries = manifest.get("entries")
    matrix = {
        (entry.get("cell_id"), entry.get("method"))
        for entry in entries
        if isinstance(entry, dict)
    } if isinstance(entries, list) else set()
    prior_precision_value = prior.get("method_role_precision")
    prior_precision = (
        prior_precision_value if isinstance(prior_precision_value, dict) else {}
    )
    checks = {
        "schema_protocol_config_exact": manifest.get("schema") == MANIFEST_SCHEMA
        and manifest.get("protocol_id") == EXPECTED_PROTOCOL
        and manifest.get("build_config_sha256") == config_sha256,
        "proposal_matrix_complete": isinstance(entries, list)
        and len(entries) == 48
        and len(matrix) == 48,
        "proposals_bitwise_unchanged": canonical_sha256(
            _proposal_projection(manifest)
        )
        == canonical_sha256(_proposal_projection(prior)),
        "statistical_targets_unchanged": manifest.get("method_role_precision")
        == prior_precision
        and prior_precision.get("relative_standard_error_targets")
        == EXPECTED_TARGETS,
        "resource_cap_rule_exact": manifest.get("resource_cap_derivation")
        == config["resource_cap_rule"]
        and manifest.get("resource_caps", {}).get(
            "maximum_final_samples_by_method"
        )
        == EXPECTED_CAPS,
        "fresh_namespace_exact": manifest.get("pilot_namespace") == EXPECTED_PILOT
        and manifest.get("final_namespace") == EXPECTED_FINAL
        and manifest.get("pilot_namespace") != prior.get("pilot_namespace")
        and manifest.get("final_namespace") != prior.get("final_namespace"),
        "clean_frozen_source": manifest.get("dirty_worktree") is False
        and isinstance(manifest.get("source_commit"), str)
        and len(manifest["source_commit"]) == 40,
        "decision_fail_closed": manifest.get(
            "design_informed_by_prior_resource_failure"
        )
        is True
        and manifest.get("current_namespace_outcomes_inspected_before_freeze")
        is False
        and manifest.get("fresh_pilot_required") is True
        and manifest.get("final_execution_authorized") is False
        and manifest.get("performance_claim_authorized") is False
        and manifest.get("submission_authorized") is False,
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    return {
        "schema": AUDIT_SCHEMA,
        "manifest_file_sha256": _sha256(manifest_path),
        "manifest_canonical_sha256": canonical_sha256(manifest),
        "checks": checks,
        "failures": failures,
        "passed": not failures,
        "decision": {
            "status": (
                "resource_cap_manifest_audit_pass"
                if not failures
                else "resource_cap_manifest_audit_fail"
            ),
            "new_execution_config_authorized": not failures,
            "new_formal_pilot_authorized": False,
            "final_execution_authorized": False,
            "performance_claim_authorized": False,
            "submission_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--config", type=Path, required=True)
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "build":
        manifest = build_manifest(arguments.config)
        digest = write_json_atomic_nonoverwriting(arguments.output, manifest)
        print(
            json.dumps(
                {
                    "manifest_sha256": digest,
                    "entry_count": manifest["entry_count"],
                    "new_formal_pilot_authorized": False,
                },
                sort_keys=True,
            )
        )
        return
    report = audit_manifest(arguments.config, arguments.manifest)
    digest = write_json_atomic_nonoverwriting(arguments.output, report)
    print(json.dumps({"audit_sha256": digest, **report["decision"]}, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
