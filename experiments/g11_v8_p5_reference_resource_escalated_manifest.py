"""Build and audit the method-role, resource-escalated reference manifest."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml

from experiments.g11_v8_p5_sharded_reference_common import ROOT, load_context
from src.path_integral import (
    REFERENCE_METHODS,
    TimePiecewiseTwoDriverControl,
    rank_one_price_control_span,
)
from src.path_integral.provenance import source_provenance
from src.path_integral.reference_protocol import canonical_sha256
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting

CONFIG_SCHEMA = "npi.g11.v8-p5-reference-resource-escalated-manifest-build.v1"
MANIFEST_SCHEMA = "npi.g11.v8-p5-reference-proposal-manifest.v2"
AUDIT_SCHEMA = "npi.g11.v8-p5-reference-proposal-manifest-audit.v2"

EXPECTED_PROTOCOL = "g11-v8-p5-sharded-reference-method-role-v1"
EXPECTED_PILOT_NAMESPACE = "v8-r2-reference-method-role-pilot-v1"
EXPECTED_FINAL_NAMESPACE = "v8-r2-reference-method-role-final-v1"
EXPECTED_RELATIVE_SE_TARGETS = {
    "dcs_reference": 0.02,
    "raw_crosscheck": 0.05,
}
EXPECTED_METHOD_CAPS = {
    "dcs_reference": 33_554_432,
    "raw_crosscheck": 16_777_216,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bound_path(record: Any) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError("resource-escalation binding is malformed")
    path = (ROOT / str(record["path"])).resolve()
    if ROOT not in path.parents or not path.is_file():
        raise ValueError("resource-escalation bound path is invalid")
    if record["sha256"] != _sha256(path):
        raise ValueError("resource-escalation artifact hash mismatch")
    return path


def _load_json(config: dict[str, Any], field: str) -> dict[str, Any]:
    value = json.loads(_bound_path(config[field]).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{field} must bind a JSON mapping")
    return value


def _roster(
    value: Any,
    *,
    candidate_required: bool,
) -> dict[tuple[str, str], str | None]:
    if not isinstance(value, list):
        raise ValueError("override roster must be a list")
    result: dict[tuple[str, str], str | None] = {}
    expected_fields = (
        {"cell_id", "method", "candidate_id"}
        if candidate_required
        else {"cell_id", "method"}
    )
    for record in value:
        if not isinstance(record, dict) or set(record) != expected_fields:
            raise ValueError("override-roster entry is malformed")
        key = (str(record["cell_id"]), str(record["method"]))
        if key in result:
            raise ValueError("override roster contains a duplicate")
        result[key] = str(record["candidate_id"]) if candidate_required else None
    return result


def load_manifest_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(config, dict) or config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unexpected resource-escalated manifest schema")
    expected_keys = {
        "schema",
        "protocol_id",
        "date",
        "phase",
        "design_informed_by_prior_development_outcomes",
        "current_namespace_outcomes_inspected_before_freeze",
        "threshold_binding",
        "prior_manifest",
        "prior_manifest_audit",
        "weight_result",
        "weight_audit",
        "method_role_result",
        "method_role_audit",
        "reference_protocol",
        "method_role_precision",
        "resource_caps",
        "retained_override_roster",
        "resource_only_override_roster",
        "decision",
    }
    if set(config) != expected_keys:
        raise ValueError("resource-escalated manifest config keys are malformed")
    for field in (
        "threshold_binding",
        "prior_manifest",
        "prior_manifest_audit",
        "weight_result",
        "weight_audit",
        "method_role_result",
        "method_role_audit",
    ):
        _bound_path(config[field])
    protocol = config.get("reference_protocol")
    precision = config.get("method_role_precision")
    caps = config.get("resource_caps")
    decision = config.get("decision")
    retained = _roster(config.get("retained_override_roster"), candidate_required=False)
    defensive = _roster(
        config.get("resource_only_override_roster"), candidate_required=True
    )
    if (
        config.get("protocol_id")
        != "g11-v8-p5-reference-resource-escalated-manifest-v1"
        or config.get("phase") != "r2_reference_resource_escalation_freeze"
        or config.get("design_informed_by_prior_development_outcomes") is not True
        or config.get("current_namespace_outcomes_inspected_before_freeze") is not False
        or not isinstance(protocol, dict)
        or protocol.get("id") != EXPECTED_PROTOCOL
        or protocol.get("pilot_namespace") != EXPECTED_PILOT_NAMESPACE
        or protocol.get("final_namespace") != EXPECTED_FINAL_NAMESPACE
        or protocol.get("methods") != list(REFERENCE_METHODS)
        or protocol.get("primary_method") != "dcs_reference"
        or protocol.get("estimand") != "fixed_finest_grid"
        or protocol.get("dtype") != "float64"
        or protocol.get("device") != "cpu"
        or not isinstance(precision, dict)
        or precision.get("relative_standard_error_targets")
        != EXPECTED_RELATIVE_SE_TARGETS
        or float(precision.get("maximum_combined_agreement_z", 0.0)) != 4.0
        or precision.get("raw_may_replace_primary_reference") is not False
        or precision.get("ordinary_mean_required") is not True
        or precision.get("self_normalization_allowed") is not False
        or not isinstance(caps, dict)
        or caps.get("maximum_final_samples_by_method") != EXPECTED_METHOD_CAPS
        or int(caps.get("minimum_final_samples", 0)) != 8192
        or int(caps.get("final_chunk_size", 0)) != 4096
        or float(caps.get("allocation_safety_factor", 0.0)) != 6.0
        or len(retained) != 4
        or len(defensive) != 4
        or set(retained) & set(defensive)
        or not isinstance(decision, dict)
        or decision.get("manifest_build_authorized") is not True
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
        raise ValueError("resource-escalated manifest contract is invalid")
    return config, hashlib.sha256(raw).hexdigest()


def _copy_proposal(
    entry: dict[str, Any],
    proposal: dict[str, Any],
    *,
    source_kind: str,
    source_id: str,
    selection_status: str,
) -> dict[str, Any]:
    result = copy.deepcopy(entry)
    weights = [float(value) for value in proposal["weights"]]
    schedules = [
        [[float(pair[0]), float(pair[1])] for pair in schedule]
        for schedule in proposal["schedules"]
    ]
    if (
        not weights
        or len(weights) != len(schedules)
        or any(weight <= 0.0 or not math.isfinite(weight) for weight in weights)
        or not math.isclose(sum(weights), 1.0)
        or any(not schedule for schedule in schedules)
        or any(abs(value) > 0.0 for pair in schedules[0] for value in pair)
    ):
        raise ValueError("selected resource-escalation proposal is malformed")
    result.update(
        {
            "weights": weights,
            "schedules": schedules,
            "source_kind": source_kind,
            "source_id": source_id,
            "selection_status": selection_status,
            "overrides_v4_resource_infeasible_entry": True,
            "development_requested_to_original_cap_ratio": float(
                proposal["requested_to_cap_ratio"]
            ),
            "development_gates": copy.deepcopy(proposal["gates"]),
        }
    )
    result.pop("overrides_failed_pilot_entry", None)
    return result


def _assert_dcs_rank_one(
    entry: dict[str, Any],
    *,
    maturity: float,
    steps: int,
) -> None:
    controls = [
        TimePiecewiseTwoDriverControl(
            tuple((float(pair[0]), float(pair[1])) for pair in schedule),
            maturity=maturity,
        )
        for schedule in entry["schedules"]
    ]
    times = torch.arange(steps, dtype=torch.float64) * (maturity / steps)
    expanded = torch.stack(
        [control.deterministic_schedule(times) for control in controls]
    )
    span = rank_one_price_control_span(expanded, step_dt=maturity / steps)
    if span.maximum_span_residual > 1e-10:
        raise ValueError("DCS reference proposal is not rank-one in price control")


def build_manifest(config_path: Path) -> dict[str, Any]:
    config, config_sha256 = load_manifest_config(config_path)
    provenance = source_provenance()
    if provenance["dirty_worktree"]:
        raise RuntimeError("resource-escalated manifest requires a clean Git worktree")
    context = load_context(
        ROOT / "configs/g11_v8/p5_sharded_reference_execution_v3.yaml"
    )
    if context.binding_sha256 != config["threshold_binding"]["sha256"]:
        raise ValueError("resource-escalated manifest binds the wrong threshold roster")
    prior = _load_json(config, "prior_manifest")
    prior_audit = _load_json(config, "prior_manifest_audit")
    weight = _load_json(config, "weight_result")
    weight_audit = _load_json(config, "weight_audit")
    method_role = _load_json(config, "method_role_result")
    method_role_audit = _load_json(config, "method_role_audit")
    if (
        prior_audit.get("passed") is not True
        or prior_audit.get("manifest_file_sha256")
        != config["prior_manifest"]["sha256"]
        or weight_audit.get("passed") is not True
        or method_role_audit.get("passed") is not True
        or weight.get("decision", {}).get("status")
        != "weight_optimized_proposal_falsification_fail"
        or method_role.get("decision", {}).get("status")
        != "method_role_proposal_falsification_fail"
    ):
        raise ValueError("development evidence is absent or incorrectly disclosed")
    prior_entries = prior.get("entries")
    if not isinstance(prior_entries, list) or len(prior_entries) != 48:
        raise ValueError("prior proposal manifest is incomplete")
    entries_by_key = {
        (str(entry["cell_id"]), str(entry["method"])): copy.deepcopy(entry)
        for entry in prior_entries
        if isinstance(entry, dict)
    }
    retained = _roster(config["retained_override_roster"], candidate_required=False)
    defensive = _roster(
        config["resource_only_override_roster"], candidate_required=True
    )
    retained_proposals = weight.get("complete_selected_proposals")
    candidates = method_role.get("candidates")
    if not isinstance(retained_proposals, dict) or not isinstance(candidates, list):
        raise ValueError("development proposal collections are malformed")
    candidates_by_id = {
        str(candidate["candidate_id"]): candidate
        for candidate in candidates
        if isinstance(candidate, dict)
    }
    for key in retained:
        cell_id, method = key
        proposal = retained_proposals.get(cell_id, {}).get(method)
        if not isinstance(proposal, dict) or not all(proposal["gates"].values()):
            raise ValueError("retained proposal did not pass every frozen gate")
        entries_by_key[key] = _copy_proposal(
            entries_by_key[key],
            proposal,
            source_kind="retained_frozen_pass",
            source_id=str(proposal["candidate_id"]),
            selection_status="passed_all_frozen_development_gates",
        )
    for key, candidate_id in defensive.items():
        proposal = candidates_by_id.get(str(candidate_id))
        if (
            not isinstance(proposal, dict)
            or (proposal.get("cell_id"), proposal.get("method")) != key
            or proposal.get("passes") is not False
        ):
            raise ValueError("resource-only proposal identity or disclosure is invalid")
        entries_by_key[key] = _copy_proposal(
            entries_by_key[key],
            proposal,
            source_kind="resource_only_after_falsification",
            source_id=str(candidate_id),
            selection_status="did_not_pass_all_development_concentration_gates",
        )
    expected_keys = {
        (cell_id, method)
        for cell_id in context.cells_by_id
        for method in REFERENCE_METHODS
    }
    if set(entries_by_key) != expected_keys:
        raise ValueError("resource-escalated proposal matrix is incomplete")
    for key, entry in entries_by_key.items():
        if key not in retained and key not in defensive:
            entry["source_kind"] = "prior_v4_resource_feasible"
            entry["source_id"] = str(entry.get("source_id", "prior-v4"))
            entry["selection_status"] = "passed_prior_formal_pilot_resource_gate"
            entry["overrides_v4_resource_infeasible_entry"] = False
            entry["development_requested_to_original_cap_ratio"] = None
            entry["development_gates"] = None
        entry.pop("overrides_failed_pilot_entry", None)
    entries = [entries_by_key[key] for key in sorted(entries_by_key)]
    protocol = config["reference_protocol"]
    return {
        "schema": MANIFEST_SCHEMA,
        "protocol_id": protocol["id"],
        "build_config_sha256": config_sha256,
        "threshold_binding_sha256": context.binding_sha256,
        "threshold_manifest_sha256": context.binding["threshold_manifest_sha256"],
        "pilot_namespace": protocol["pilot_namespace"],
        "final_namespace": protocol["final_namespace"],
        "methods": protocol["methods"],
        "primary_method": protocol["primary_method"],
        "estimand": protocol["estimand"],
        "dtype": protocol["dtype"],
        "device": protocol["device"],
        "method_role_precision": copy.deepcopy(config["method_role_precision"]),
        "resource_caps": copy.deepcopy(config["resource_caps"]),
        "entry_count": len(entries),
        "override_count": sum(
            bool(entry["overrides_v4_resource_infeasible_entry"]) for entry in entries
        ),
        "entries": entries,
        "resource_escalation_only": True,
        "fresh_pilot_required": True,
        "final_execution_authorized": False,
        "performance_claim_authorized": False,
        "submission_authorized": False,
        **provenance,
    }


def audit_manifest(config_path: Path, manifest_path: Path) -> dict[str, Any]:
    config, config_sha256 = load_manifest_config(config_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("resource-escalated proposal manifest must be a mapping")
    context = load_context(
        ROOT / "configs/g11_v8/p5_sharded_reference_execution_v3.yaml"
    )
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        entries = []
    matrix = {
        (entry.get("cell_id"), entry.get("method"))
        for entry in entries
        if isinstance(entry, dict)
    }
    expected_matrix = {
        (cell_id, method)
        for cell_id in context.cells_by_id
        for method in REFERENCE_METHODS
    }
    source_counts = {
        source: sum(
            isinstance(entry, dict) and entry.get("source_kind") == source
            for entry in entries
        )
        for source in (
            "prior_v4_resource_feasible",
            "retained_frozen_pass",
            "resource_only_after_falsification",
        )
    }
    proposal_structure_pass = True
    dcs_rank_one_pass = True
    for entry in entries:
        try:
            weights = [float(value) for value in entry["weights"]]
            schedules = entry["schedules"]
            if (
                len(weights) != len(schedules)
                or any(weight <= 0.0 or not math.isfinite(weight) for weight in weights)
                or not math.isclose(sum(weights), 1.0)
                or any(not schedule for schedule in schedules)
                or any(
                    len(pair) != 2
                    or any(not math.isfinite(float(value)) for value in pair)
                    for schedule in schedules
                    for pair in schedule
                )
                or any(abs(float(value)) > 0.0 for pair in schedules[0] for value in pair)
            ):
                proposal_structure_pass = False
            if entry["method"] == "dcs_reference":
                cell = context.cells_by_id[entry["cell_id"]]
                _assert_dcs_rank_one(
                    entry,
                    maturity=float(cell["maturity"]),
                    steps=int(cell["finest_steps"]),
                )
        except (KeyError, TypeError, ValueError):
            proposal_structure_pass = False
            if isinstance(entry, dict) and entry.get("method") == "dcs_reference":
                dcs_rank_one_pass = False
    resource_only_entries = [
        entry
        for entry in entries
        if isinstance(entry, dict)
        and entry.get("source_kind") == "resource_only_after_falsification"
    ]
    checks = {
        "schema_protocol_config_exact": manifest.get("schema") == MANIFEST_SCHEMA
        and manifest.get("protocol_id") == EXPECTED_PROTOCOL
        and manifest.get("build_config_sha256") == config_sha256,
        "threshold_binding_exact": manifest.get("threshold_binding_sha256")
        == config["threshold_binding"]["sha256"],
        "complete_unique_matrix": len(entries) == 48
        and len(matrix) == 48
        and matrix == expected_matrix,
        "override_and_source_roster_exact": manifest.get("override_count") == 8
        and source_counts
        == {
            "prior_v4_resource_feasible": 40,
            "retained_frozen_pass": 4,
            "resource_only_after_falsification": 4,
        },
        "method_roles_and_precision_exact": manifest.get("primary_method")
        == "dcs_reference"
        and manifest.get("method_role_precision")
        == config["method_role_precision"],
        "resource_caps_exact": manifest.get("resource_caps")
        == config["resource_caps"],
        "proposal_structure_exact": proposal_structure_pass,
        "dcs_rank_one_structure_exact": dcs_rank_one_pass,
        "resource_only_failure_disclosed": len(resource_only_entries) == 4
        and all(
            entry.get("selection_status")
            == "did_not_pass_all_development_concentration_gates"
            and isinstance(entry.get("development_gates"), dict)
            and entry["development_gates"].get("contribution_concentration_pass")
            is False
            for entry in resource_only_entries
        ),
        "clean_frozen_source": manifest.get("dirty_worktree") is False
        and isinstance(manifest.get("source_commit"), str)
        and len(manifest["source_commit"]) == 40,
        "decision_fail_closed": manifest.get("resource_escalation_only") is True
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
        "source_counts": source_counts,
        "failures": failures,
        "passed": not failures,
        "decision": {
            "status": (
                "resource_escalated_manifest_audit_pass"
                if not failures
                else "resource_escalated_manifest_audit_fail"
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
                    "override_count": manifest["override_count"],
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
