"""Fail-closed audit for the post-R2 G11 V8 completion-status ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "npi.g11.v8-completion-status-ledger.v2"
REPORT_SCHEMA = "npi.g11.v8-completion-status-audit.v2"
REFERENCE_COMMIT = "e66c1d37a2201aba4e03ebc51efed0c7a9f70bc5"
EXPECTED_ARTIFACTS = {
    "p0_claim_contract": "conditional",
    "p4_baseline_framework": "conditional",
    "p5_matrix": "passed",
    "p5_threshold_binding_v2": "passed",
    "p6_statistical_design": "passed",
    "r2_failed_execution_receipt": "failed",
    "r2_log_spot_remediation_audit": "passed",
    "r2_benchmark": "passed",
    "r2_authorization": "passed",
    "r2_aggregate": "passed",
    "r2_aggregate_audit": "passed",
    "r2_execution_receipt": "passed",
    "r2_final_audit": "passed",
}
EXPECTED_CLOSED = {
    "sharded_checkpointed_reference_infrastructure",
    "fresh_high_precision_reference",
}
EXPECTED_OPEN = {
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
    "b1_strong_baseline_implementation",
    "b1_correctness_oracles",
    "d1_protocol_implementation",
    "t1_novelty_and_theory_development",
}
EXPECTED_PROHIBITED = {
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
    "predecessor",
    "reference_repository_commit",
    "reference_outcomes_inspected",
    "current_b1_outcomes_inspected_before_freeze",
    "artifacts",
    "consumed_namespaces",
    "closed_obligations",
    "open_obligations",
    "reserved_namespaces",
    "authorized_actions",
    "prohibited_claims",
    "decision",
}
ARTIFACT_KEYS = {"id", "path", "sha256", "status", "performance_authorized"}
DECISION_KEYS = {
    "status",
    "next_blocking_phase",
    "reference_complete",
    "performance_claim_authorized",
    "p8_qualification_authorized",
    "p9_freeze_authorized",
    "submission_authorized",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_ledger(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    value = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("unexpected R2 completion-status schema")
    return value, hashlib.sha256(raw).hexdigest()


def _strings(value: Any) -> set[str]:
    return {item for item in value if isinstance(item, str)} if isinstance(value, list) else set()


def _records(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _ancestor(commit: Any, root: Path) -> bool:
    if commit != REFERENCE_COMMIT:
        return False
    result = subprocess.run(
        ("git", "merge-base", "--is-ancestor", commit, "HEAD"),
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def audit_completion_status_v2(
    ledger: dict[str, Any], ledger_sha256: str, *, root: Path = ROOT
) -> dict[str, Any]:
    checks: dict[str, bool] = {}

    def record(name: str, value: bool) -> None:
        checks[name] = bool(value)

    artifacts = _records(ledger.get("artifacts"))
    by_id = {item.get("id"): item for item in artifacts if isinstance(item.get("id"), str)}
    predecessor = ledger.get("predecessor")
    decision = ledger.get("decision")
    record("schema_exact", ledger.get("schema") == SCHEMA)
    record("root_keys_exact", set(ledger) == ROOT_KEYS)
    record("protocol_exact", ledger.get("protocol_id") == "g11-v8-completion-status-ledger-v2")
    record(
        "date_phase_exact",
        ledger.get("date") == "2026-08-02" and ledger.get("phase") == "r2_reference_close",
    )
    record(
        "reference_commit_exact_and_ancestor",
        _ancestor(ledger.get("reference_repository_commit"), root),
    )
    record("reference_outcome_use_disclosed", ledger.get("reference_outcomes_inspected") is True)
    record(
        "b1_namespace_unopened", ledger.get("current_b1_outcomes_inspected_before_freeze") is False
    )
    record(
        "predecessor_exact",
        isinstance(predecessor, dict)
        and set(predecessor) == {"path", "sha256"}
        and predecessor.get("path") == "configs/g11_v8/completion_status_ledger_v1.yaml"
        and (root / str(predecessor.get("path"))).is_file()
        and predecessor.get("sha256") == _sha256(root / str(predecessor.get("path"))),
    )
    record(
        "artifact_ids_exact",
        set(by_id) == set(EXPECTED_ARTIFACTS) and len(artifacts) == len(EXPECTED_ARTIFACTS),
    )
    record(
        "artifact_keys_exact",
        bool(artifacts) and all(set(item) == ARTIFACT_KEYS for item in artifacts),
    )
    record(
        "artifact_statuses_exact",
        all(
            by_id.get(key, {}).get("status") == status for key, status in EXPECTED_ARTIFACTS.items()
        ),
    )
    record(
        "artifacts_refuse_performance",
        bool(artifacts) and all(item.get("performance_authorized") is False for item in artifacts),
    )
    paths_ok = True
    hashes_ok = True
    for item in artifacts:
        relative = item.get("path")
        digest = item.get("sha256")
        portable = (
            isinstance(relative, str)
            and not Path(relative).is_absolute()
            and "\\" not in relative
            and ".." not in Path(relative).parts
        )
        paths_ok &= portable and (root / str(relative)).is_file()
        hashes_ok &= (
            portable
            and (root / str(relative)).is_file()
            and digest == _sha256(root / str(relative))
        )
    record("artifact_paths_exist_and_portable", paths_ok)
    record("artifact_hashes_match", hashes_ok)

    aggregate_path = root / str(by_id.get("r2_aggregate", {}).get("path", ""))
    audit_path = root / str(by_id.get("r2_aggregate_audit", {}).get("path", ""))
    receipt_path = root / str(by_id.get("r2_execution_receipt", {}).get("path", ""))
    try:
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        aggregate, audit, receipt = {}, {}, {}
    record(
        "r2_reference_evidence_passes",
        aggregate.get("reference_acceptance_pass") is True
        and aggregate.get("all_target_standard_errors") is True
        and aggregate.get("all_likelihood_normalizations") is True
        and aggregate.get("all_independent_methods_agree") is True
        and audit.get("passed") is True
        and audit.get("decision", {}).get("reference_complete") is True
        and receipt.get("decision", {}).get("reference_complete") is True,
    )
    record(
        "r2_performance_claim_refused",
        aggregate.get("performance_claim_authorized") is False
        and audit.get("decision", {}).get("performance_claim_authorized") is False
        and receipt.get("decision", {}).get("performance_claim_authorized") is False,
    )
    record(
        "closed_obligations_exact", _strings(ledger.get("closed_obligations")) == EXPECTED_CLOSED
    )
    record("open_obligations_exact", _strings(ledger.get("open_obligations")) == EXPECTED_OPEN)
    record(
        "closed_open_disjoint",
        not (_strings(ledger.get("closed_obligations")) & _strings(ledger.get("open_obligations"))),
    )
    record(
        "authorized_actions_exact",
        _strings(ledger.get("authorized_actions")) == EXPECTED_AUTHORIZED,
    )
    record(
        "prohibited_claims_exact", _strings(ledger.get("prohibited_claims")) == EXPECTED_PROHIBITED
    )
    record("decision_keys_exact", isinstance(decision, dict) and set(decision) == DECISION_KEYS)
    record(
        "decision_fail_closed",
        isinstance(decision, dict)
        and decision.get("status") == "r2_reference_complete_b1_authorized"
        and decision.get("next_blocking_phase") == "b1_strong_baseline_implementation"
        and decision.get("reference_complete") is True
        and decision.get("performance_claim_authorized") is False
        and decision.get("p8_qualification_authorized") is False
        and decision.get("p9_freeze_authorized") is False
        and decision.get("submission_authorized") is False,
    )
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "schema": REPORT_SCHEMA,
        "ledger_sha256": ledger_sha256,
        "artifact_count": len(artifacts),
        "checks": checks,
        "failures": failures,
        "passed": not failures,
        "decision": {
            "reference_complete": not failures,
            "b1_implementation_authorized": not failures,
            "performance_claim_authorized": False,
            "p8_qualification_authorized": False,
            "submission_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    ledger, digest = load_ledger(args.ledger)
    report = audit_completion_status_v2(ledger, digest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], **report["decision"]}, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
