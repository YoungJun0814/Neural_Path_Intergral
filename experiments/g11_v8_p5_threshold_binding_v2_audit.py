"""Fail-closed audit for the fresh R2 sharded-reference threshold binding."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "npi.g11.v8-p5-threshold-manifest-binding.v2"
REPORT_SCHEMA = "npi.g11.v8-p5-threshold-binding-audit.v2"
OLD_BINDING_SHA256 = "9000876f39b370457fe7a4c5e367c078712bb72505927a8642b090451af79218"
ROOT_KEYS = {
    "schema",
    "protocol_id",
    "date",
    "phase",
    "design_informed_by_prior_development_outcomes",
    "current_namespace_outcomes_inspected_before_freeze",
    "completion_status_ledger",
    "reference_infrastructure_contract",
    "reference_infrastructure_audit",
    "p5_reference_matrix_design_sha256",
    "p6_statistical_design_sha256",
    "threshold_calibration_config",
    "threshold_calibration_result",
    "threshold_manifest_sha256",
    "calibration_source_commit",
    "reference_protocol",
    "burned_namespaces",
    "decision",
}
EXPECTED_BURNED = {
    "p5-threshold-calibration-development",
    "p5-reference",
    "p5-reference-v2",
    "v8-p7-development",
}
EXPECTED_METHODS = ["dcs_reference", "raw_crosscheck"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bound_file(record: Any) -> tuple[Path | None, bool]:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        return None, False
    relative = record.get("path")
    if not isinstance(relative, str):
        return None, False
    path = (ROOT / relative).resolve()
    if ROOT not in path.parents:
        return None, False
    return path, path.is_file() and record.get("sha256") == _sha256(path)


def load_binding_v2(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("unexpected P5 threshold-binding V2 schema")
    return payload, hashlib.sha256(raw).hexdigest()


def audit_binding_v2(binding: dict[str, Any], digest: str) -> dict[str, Any]:
    completion_path, completion_bound = _bound_file(
        binding.get("completion_status_ledger")
    )
    infrastructure_path, infrastructure_bound = _bound_file(
        binding.get("reference_infrastructure_contract")
    )
    infrastructure_audit_path, infrastructure_audit_bound = _bound_file(
        binding.get("reference_infrastructure_audit")
    )
    calibration_config_path, calibration_config_bound = _bound_file(
        binding.get("threshold_calibration_config")
    )
    calibration_result_path, calibration_result_bound = _bound_file(
        binding.get("threshold_calibration_result")
    )

    def load_yaml(path: Path | None) -> dict[str, Any]:
        if path is None:
            return {}
        try:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError):
            return {}
        return value if isinstance(value, dict) else {}

    def load_json(path: Path | None) -> dict[str, Any]:
        if path is None:
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    completion = load_yaml(completion_path)
    infrastructure = load_yaml(infrastructure_path)
    infrastructure_audit = load_json(infrastructure_audit_path)
    calibration_result = load_json(calibration_result_path)
    p5_matrix = ROOT / "configs/g11_v8/p5_reference_matrix_design_v1.yaml"
    p6_design = ROOT / "configs/g11_v8/p6_statistical_design_v1.yaml"
    old_binding = ROOT / "configs/g11_v8/p5_threshold_manifest_binding_v1.yaml"
    protocol = binding.get("reference_protocol")
    decision = binding.get("decision")
    burned = binding.get("burned_namespaces")
    candidate_manifest = calibration_result.get("candidate_manifest")
    canonical_manifest_sha256 = (
        hashlib.sha256(
            json.dumps(
                candidate_manifest,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        if isinstance(candidate_manifest, dict)
        else None
    )
    p6 = load_yaml(p6_design)
    p6_namespaces = set(p6.get("seed_namespaces", {}).values())
    pilot_namespace = protocol.get("pilot_namespace") if isinstance(protocol, dict) else None
    final_namespace = protocol.get("final_namespace") if isinstance(protocol, dict) else None
    final_method_namespace = (
        protocol.get("final_method_seed_namespace")
        if isinstance(protocol, dict)
        else None
    )

    checks = {
        "schema_and_root_exact": binding.get("schema") == SCHEMA
        and set(binding) == ROOT_KEYS,
        "protocol_and_phase_exact": binding.get("protocol_id")
        == "g11-v8-p5-threshold-manifest-binding-v2"
        and binding.get("phase") == "r2_reference_development_bound",
        "prior_design_influence_disclosed": binding.get(
            "design_informed_by_prior_development_outcomes"
        )
        is True,
        "current_namespaces_unopened": binding.get(
            "current_namespace_outcomes_inspected_before_freeze"
        )
        is False,
        "completion_ledger_bound": completion_bound
        and completion.get("decision", {}).get("status") == "r0_status_freeze_ready",
        "r1_contract_bound": infrastructure_bound
        and infrastructure.get("decision", {}).get("status")
        == "r1_contract_frozen_implementation_requires_audit",
        "r1_audit_bound_and_passed": infrastructure_audit_bound
        and infrastructure_audit.get("passed") is True
        and infrastructure_audit.get("decision", {}).get(
            "r2_development_benchmark_authorized"
        )
        is True
        and infrastructure_audit.get("decision", {}).get("formal_reference_complete")
        is False,
        "old_binding_unchanged": old_binding.is_file()
        and _sha256(old_binding) == OLD_BINDING_SHA256,
        "p5_p6_designs_bound": p5_matrix.is_file()
        and p6_design.is_file()
        and binding.get("p5_reference_matrix_design_sha256") == _sha256(p5_matrix)
        and binding.get("p6_statistical_design_sha256") == _sha256(p6_design),
        "calibration_artifacts_bound": calibration_config_bound
        and calibration_result_bound,
        "clean_complete_calibration_reused": calibration_result.get("passed") is True
        and calibration_result.get("smoke") is False
        and calibration_result.get("dirty_worktree") is False
        and calibration_result.get("source_commit")
        == binding.get("calibration_source_commit")
        and isinstance(calibration_result.get("cells"), list)
        and len(calibration_result["cells"]) == 24,
        "threshold_manifest_hash_exact": canonical_manifest_sha256
        == calibration_result.get("candidate_manifest_sha256")
        == binding.get("threshold_manifest_sha256"),
        "reference_protocol_exact": isinstance(protocol, dict)
        and protocol.get("id") == "g11-v8-p5-sharded-reference-development-v1"
        and protocol.get("allocation_schema")
        == "npi.g11.v8-reference-allocation-manifest.v1"
        and protocol.get("shard_schema") == "npi.g11.v8-reference-shard.v1"
        and protocol.get("estimand") == "fixed_finest_grid"
        and protocol.get("methods") == EXPECTED_METHODS
        and protocol.get("dtype") == "float64"
        and protocol.get("device") == "cpu",
        "namespace_roster_exact": isinstance(burned, list)
        and set(burned) == EXPECTED_BURNED
        and pilot_namespace == "v8-r2-reference-development"
        and final_namespace == "v8-r2-reference-development-final",
        "new_namespaces_disjoint": isinstance(pilot_namespace, str)
        and isinstance(final_namespace, str)
        and pilot_namespace != final_namespace
        and pilot_namespace not in EXPECTED_BURNED
        and final_namespace not in EXPECTED_BURNED
        and pilot_namespace not in p6_namespaces
        and final_namespace not in p6_namespaces
        and final_method_namespace not in {
            pilot_namespace,
            final_namespace,
        },
        "decision_fail_closed": isinstance(decision, dict)
        and decision.get("status") == "r2_development_binding_frozen"
        and decision.get("representative_benchmark_authorized") is True
        and decision.get("full_pilot_execution_authorized") is False
        and decision.get("reference_complete") is False
        and decision.get("performance_claim_authorized") is False
        and decision.get("submission_authorized") is False,
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    return {
        "schema": REPORT_SCHEMA,
        "binding_sha256": digest,
        "checks": checks,
        "failure_count": len(failures),
        "failures": failures,
        "passed": not failures,
        "decision": {
            "status": (
                "r2_development_binding_pass"
                if not failures
                else "r2_development_binding_fail"
            ),
            "representative_benchmark_authorized": not failures,
            "full_pilot_execution_authorized": False,
            "performance_claim_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    binding, digest = load_binding_v2(arguments.binding)
    report = audit_binding_v2(binding, digest)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if arguments.output is not None:
        if arguments.output.exists():
            raise FileExistsError(
                f"refusing to overwrite binding audit: {arguments.output}"
            )
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
