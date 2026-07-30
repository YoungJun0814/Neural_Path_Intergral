"""Fail-closed audit for the G11 V8 R0 completion-status ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "npi.g11.v8-completion-status-ledger.v1"
REPORT_SCHEMA = "npi.g11.v8-completion-status-audit.v1"
EXPECTED_BASELINE_COMMIT = "134d15350ee8f40ff5d32395aa27991cd15ed0de"
VALID_STATUSES = {"passed", "conditional", "failed", "interrupted", "open"}
EXPECTED_ARTIFACT_STATUSES = {
    "p0_claim_contract": "conditional",
    "p1_novelty_ledger": "conditional",
    "p2_theorem_ledger": "conditional",
    "p3_rate_complexity_ledger": "conditional",
    "p4_baseline_framework_ledger": "conditional",
    "p5_reference_matrix_design": "passed",
    "p5_threshold_binding": "passed",
    "p6_statistical_design": "passed",
    "p7_development_config": "passed",
    "p7_development_result": "conditional",
    "p5_threshold_calibration_result": "passed",
    "p5_reference_v1_receipt": "failed",
    "p5_reference_v2_receipt": "interrupted",
    "completion_technical_plan": "passed",
    "completion_korean_plan": "passed",
}
EXPECTED_BURNED = {
    "p5-threshold-calibration-development": "completed",
    "p5-reference": "failed",
    "p5-reference-v2": "interrupted",
    "v8-p7-development": "completed",
}
EXPECTED_RESERVED = {
    "v8-r1-reference-smoke",
    "v8-r2-reference-development",
    "v8-r2-reference-qualification",
    "v8-p8-qualification",
    "v8-p9-freeze-receipt",
    "v8-p10-confirmation",
    "v8-p10-bootstrap",
    "v8-p11-independent-hardware",
}
EXPECTED_OPEN = {
    "sharded_checkpointed_reference_infrastructure",
    "fresh_high_precision_reference",
    "production_strong_external_baselines",
    "p7_falsification_benchmark",
    "quantitative_or_model_level_new_theorem",
    "updated_primary_source_novelty_review",
    "p8_independent_seed_qualification",
    "p9_outcome_blind_freeze",
    "p10_uncensored_confirmation",
    "p11_independent_physical_hardware",
    "external_mathematical_review",
    "external_code_review",
    "claim_to_evidence_manuscript_audit",
}
EXPECTED_AUTHORIZED = {
    "r1_reference_infrastructure_implementation",
    "t1_novelty_and_theory_development",
}
EXPECTED_PROHIBITED = {
    "full_p5_reference_complete",
    "external_comparator_superiority",
    "unconditional_rbergomi_mlmc_complexity",
    "continuous_barrier_exactness",
    "independent_physical_reproduction_complete",
    "top_journal_submission_ready",
}
ROOT_KEYS = {
    "schema",
    "protocol_id",
    "date",
    "phase",
    "baseline_repository_commit",
    "design_informed_by_prior_development_outcomes",
    "current_namespace_outcomes_inspected_before_freeze",
    "artifacts",
    "burned_namespaces",
    "reserved_namespaces",
    "open_obligations",
    "authorized_actions",
    "prohibited_claims",
    "decision",
}
ARTIFACT_KEYS = {
    "id",
    "path",
    "sha256",
    "status",
    "evidence_class",
    "performance_authorized",
}
BURNED_KEYS = {"id", "terminal_status", "reusable", "reason"}
DECISION_KEYS = {
    "status",
    "next_blocking_phase",
    "performance_claim_authorized",
    "p8_qualification_authorized",
    "p9_freeze_authorized",
    "submission_authorized",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_ledger(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("unexpected R0 completion-status schema")
    return payload, hashlib.sha256(raw).hexdigest()


def _records(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _strings(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {item for item in value if isinstance(item, str)}


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _baseline_commit_is_ancestor(commit: Any, root: Path) -> bool:
    if commit != EXPECTED_BASELINE_COMMIT:
        return False
    try:
        result = subprocess.run(
            ("git", "merge-base", "--is-ancestor", commit, "HEAD"),
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return result.returncode == 0


def audit_completion_status(
    ledger: dict[str, Any],
    ledger_sha256: str,
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Audit artifact truth labels, hashes, namespaces, and authorization gates."""

    checks: dict[str, bool] = {}

    def record(name: str, condition: bool) -> None:
        checks[name] = bool(condition)

    artifacts = _records(ledger.get("artifacts"))
    artifact_by_id = {
        item.get("id"): item for item in artifacts if isinstance(item.get("id"), str)
    }
    burned = _records(ledger.get("burned_namespaces"))
    burned_by_id = {
        item.get("id"): item for item in burned if isinstance(item.get("id"), str)
    }
    reserved = _strings(ledger.get("reserved_namespaces"))
    decision = _mapping(ledger.get("decision"))

    record("schema_exact", ledger.get("schema") == SCHEMA)
    record("root_keys_exact", set(ledger) == ROOT_KEYS)
    record(
        "protocol_exact",
        ledger.get("protocol_id") == "g11-v8-completion-status-ledger-v1",
    )
    record("date_exact", ledger.get("date") == "2026-07-31")
    record("phase_exact", ledger.get("phase") == "r0_status_freeze")
    baseline_commit = ledger.get("baseline_repository_commit")
    record(
        "baseline_commit_well_formed",
        isinstance(baseline_commit, str)
        and len(baseline_commit) == 40
        and all(character in "0123456789abcdef" for character in baseline_commit),
    )
    record(
        "baseline_commit_exact_and_ancestor",
        _baseline_commit_is_ancestor(baseline_commit, root),
    )
    record(
        "prior_outcome_use_disclosed",
        ledger.get("design_informed_by_prior_development_outcomes") is True,
    )
    record(
        "current_namespace_unopened",
        ledger.get("current_namespace_outcomes_inspected_before_freeze") is False,
    )

    record(
        "artifact_ids_exact",
        set(artifact_by_id) == set(EXPECTED_ARTIFACT_STATUSES)
        and len(artifacts) == len(EXPECTED_ARTIFACT_STATUSES),
    )
    record(
        "artifact_keys_exact",
        bool(artifacts) and all(set(item) == ARTIFACT_KEYS for item in artifacts),
    )
    record(
        "artifact_statuses_exact",
        all(
            artifact_by_id.get(artifact_id, {}).get("status") == status
            for artifact_id, status in EXPECTED_ARTIFACT_STATUSES.items()
        ),
    )
    record(
        "artifact_status_values_valid",
        bool(artifacts)
        and all(item.get("status") in VALID_STATUSES for item in artifacts),
    )
    record(
        "artifact_evidence_classes_present",
        bool(artifacts)
        and all(
            isinstance(item.get("evidence_class"), str)
            and bool(item["evidence_class"].strip())
            for item in artifacts
        ),
    )
    record(
        "all_artifacts_refuse_performance",
        bool(artifacts)
        and all(item.get("performance_authorized") is False for item in artifacts),
    )

    paths_valid = True
    hashes_valid = True
    portable_paths = True
    for artifact in artifacts:
        path_value = artifact.get("path")
        digest = artifact.get("sha256")
        if not isinstance(path_value, str):
            paths_valid = False
            hashes_valid = False
            portable_paths = False
            continue
        portable_paths &= (
            not Path(path_value).is_absolute()
            and "\\" not in path_value
            and ".." not in Path(path_value).parts
        )
        path = root / path_value
        paths_valid &= path.is_file()
        hashes_valid &= (
            path.is_file()
            and isinstance(digest, str)
            and len(digest) == 64
            and digest == _sha256(path)
        )
    record("artifact_paths_portable", portable_paths)
    record("all_artifact_paths_exist", paths_valid)
    record("all_artifact_hashes_match", hashes_valid)

    record(
        "burned_ids_exact",
        set(burned_by_id) == set(EXPECTED_BURNED)
        and len(burned) == len(EXPECTED_BURNED),
    )
    record(
        "burned_keys_exact",
        bool(burned) and all(set(item) == BURNED_KEYS for item in burned),
    )
    record(
        "burned_terminal_statuses_exact",
        all(
            burned_by_id.get(namespace, {}).get("terminal_status") == status
            for namespace, status in EXPECTED_BURNED.items()
        ),
    )
    record(
        "burned_namespaces_not_reusable",
        bool(burned)
        and all(
            item.get("reusable") is False
            and isinstance(item.get("reason"), str)
            and bool(item["reason"].strip())
            for item in burned
        ),
    )
    record("reserved_namespaces_exact", reserved == EXPECTED_RESERVED)
    record(
        "burned_and_reserved_disjoint",
        set(burned_by_id).isdisjoint(reserved),
    )

    record(
        "open_obligations_exact",
        _strings(ledger.get("open_obligations")) == EXPECTED_OPEN,
    )
    record(
        "authorized_actions_exact",
        _strings(ledger.get("authorized_actions")) == EXPECTED_AUTHORIZED,
    )
    record(
        "prohibited_claims_exact",
        _strings(ledger.get("prohibited_claims")) == EXPECTED_PROHIBITED,
    )
    record("decision_keys_exact", set(decision) == DECISION_KEYS)
    record("decision_status_exact", decision.get("status") == "r0_status_freeze_ready")
    record(
        "r1_is_next_blocker",
        decision.get("next_blocking_phase") == "r1_reference_infrastructure",
    )
    record(
        "all_future_claims_refused",
        decision.get("performance_claim_authorized") is False
        and decision.get("p8_qualification_authorized") is False
        and decision.get("p9_freeze_authorized") is False
        and decision.get("submission_authorized") is False,
    )

    failures = [name for name, passed in checks.items() if not passed]
    return {
        "schema": REPORT_SCHEMA,
        "ledger_schema": ledger.get("schema"),
        "ledger_sha256": ledger_sha256,
        "artifact_count": len(artifacts),
        "burned_namespace_count": len(burned),
        "reserved_namespace_count": len(reserved),
        "checks": checks,
        "failure_count": len(failures),
        "failures": failures,
        "passed": not failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {args.output}")
    ledger, digest = _load_ledger(args.ledger)
    report = audit_completion_status(ledger, digest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
