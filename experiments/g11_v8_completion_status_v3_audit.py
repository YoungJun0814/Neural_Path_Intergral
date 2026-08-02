"""Fail-closed final closure audit for the falsified G11 V8 program."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "npi.g11.v8-completion-status-ledger.v3"
REPORT_SCHEMA = "npi.g11.v8-completion-status-audit.v3"
EVIDENCE_COMMIT = "276e2bbe4c9e927450458b57bcbebdb2d643286c"
PREDECESSOR_SHA256 = "412b7ec67191dbe4fc395211e1a51ef89e1923bb9bb593e8f5302ecfd8466816"
EXPECTED_ARTIFACT_STATUS = {
    "b1_implementation_audit": "passed",
    "d1_stage_a_result": "passed",
    "d1_stage_a_audit": "passed",
    "dcs_proposal_bank": "constructed",
    "dcs_proposal_bank_audit": "passed",
    "d1_stage_b_result": "scientifically_falsified",
    "d1_stage_b_audit": "audit_passed",
    "d1_stage_b_decision": "stop",
    "t1_novelty_audit": "passed_with_blocking_decision",
    "t1_theory_novelty_decision": "terminal_only_internal_candidate",
}
EXPECTED_CLOSED = {
    "r2_independent_reference",
    "b1_strong_baseline_implementation",
    "d1_stage_a_falsification",
    "fully_costed_replayable_dcs_bank",
    "d1_stage_b_full_matrix_falsification",
    "independent_d1_stage_b_audit",
    "t1_primary_source_update",
    "finite_grid_moment_localized_strictness",
    "terminal_rate_internal_proof_obligations_o1_o5",
}
EXPECTED_STOPPED = {
    "d1_stage_c_robustness",
    "p8_independent_seed_qualification",
    "p9_outcome_blind_freeze",
    "p10_uncensored_confirmation",
    "p11_independent_physical_hardware",
    "broad_training_inclusive_superiority_claim",
    "current_v8_top_journal_submission",
}
EXPECTED_EXTERNAL = {
    "independent_stochastic_analysis_review_o6",
    "mathscinet_zbmath_scopus_webofscience_search",
    "external_novelty_challenge",
    "external_code_review",
    "barrier_active_time_and_mesh_crossing_theorem",
}
EXPECTED_PROHIBITED = {
    "universal_factor_two_variance_reduction",
    "broad_superiority_over_strong_baselines",
    "continuous_barrier_exactness",
    "barrier_mesh_rate",
    "canonical_rough_bergomi_mlmc_complexity",
    "submission_ready",
    "top_journal_ready",
    "first_conditional_monte_carlo",
    "first_rao_blackwellized_importance_sampling",
}
ROOT_KEYS = {
    "schema",
    "protocol_id",
    "date",
    "phase",
    "predecessor",
    "evidence_repository_commit",
    "artifacts",
    "closed_obligations",
    "stopped_obligations",
    "external_obligations",
    "prohibited_claims",
    "decision",
}
ARTIFACT_KEYS = {"id", "path", "sha256", "status"}
DECISION_KEYS = {
    "status",
    "all_currently_authorized_work_complete",
    "scientific_implementation_error_found",
    "broad_performance_hypothesis_passed",
    "stage_c_authorized",
    "p8_qualification_authorized",
    "p9_freeze_authorized",
    "p10_confirmation_authorized",
    "p11_reproduction_authorized",
    "submission_authorized",
    "top_journal_route_authorized",
    "future_v9_requires_new_protocol",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _is_ancestor(commit: Any, root: Path) -> bool:
    if commit != EVIDENCE_COMMIT:
        return False
    completed = subprocess.run(
        ("git", "merge-base", "--is-ancestor", commit, "HEAD"),
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def load_ledger(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("unexpected V8 final-closure schema")
    return payload, hashlib.sha256(raw).hexdigest()


def _json_artifact(root: Path, item: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = json.loads((root / str(item.get("path"))).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def audit_completion_status_v3(
    ledger: dict[str, Any], ledger_sha256: str, *, root: Path = ROOT
) -> dict[str, Any]:
    """Audit final evidence integrity and enforce the falsification stop."""

    checks: dict[str, bool] = {}

    def record(name: str, condition: bool) -> None:
        checks[name] = bool(condition)

    predecessor = _mapping(ledger.get("predecessor"))
    artifacts = _records(ledger.get("artifacts"))
    by_id = {item.get("id"): item for item in artifacts if isinstance(item.get("id"), str)}
    decision = _mapping(ledger.get("decision"))

    record("schema_exact", ledger.get("schema") == SCHEMA)
    record("root_keys_exact", set(ledger) == ROOT_KEYS)
    record(
        "protocol_exact",
        ledger.get("protocol_id") == "g11-v8-completion-status-ledger-v3",
    )
    record(
        "date_phase_exact",
        ledger.get("date") == "2026-08-02" and ledger.get("phase") == "final_falsification_closure",
    )
    predecessor_path = root / str(predecessor.get("path", ""))
    record(
        "predecessor_hash_bound",
        predecessor.get("path") == "configs/g11_v8/completion_status_ledger_v2.yaml"
        and predecessor.get("sha256") == PREDECESSOR_SHA256
        and predecessor_path.is_file()
        and _sha256(predecessor_path) == PREDECESSOR_SHA256,
    )
    record(
        "evidence_commit_exact_and_ancestor",
        _is_ancestor(ledger.get("evidence_repository_commit"), root),
    )
    record(
        "artifact_ids_and_statuses_exact",
        len(artifacts) == len(EXPECTED_ARTIFACT_STATUS)
        and set(by_id) == set(EXPECTED_ARTIFACT_STATUS)
        and all(
            by_id.get(artifact_id, {}).get("status") == expected
            for artifact_id, expected in EXPECTED_ARTIFACT_STATUS.items()
        ),
    )
    record(
        "artifact_keys_exact",
        bool(artifacts) and all(set(item) == ARTIFACT_KEYS for item in artifacts),
    )
    paths_ok = True
    hashes_ok = True
    for item in artifacts:
        relative = item.get("path")
        portable = (
            isinstance(relative, str)
            and not Path(relative).is_absolute()
            and "\\" not in relative
            and ".." not in Path(relative).parts
        )
        path = root / str(relative)
        paths_ok &= portable and path.is_file()
        hashes_ok &= portable and path.is_file() and item.get("sha256") == _sha256(path)
    record("artifact_paths_exist_and_portable", paths_ok)
    record("artifact_hashes_match", hashes_ok)

    stage_a = _json_artifact(root, by_id.get("d1_stage_a_audit", {}))
    bank = _json_artifact(root, by_id.get("dcs_proposal_bank_audit", {}))
    stage_b_result = _json_artifact(root, by_id.get("d1_stage_b_result", {}))
    stage_b_audit = _json_artifact(root, by_id.get("d1_stage_b_audit", {}))
    t1 = _json_artifact(root, by_id.get("t1_novelty_audit", {}))
    stage_b_aggregate = _mapping(stage_b_result.get("aggregate"))
    stage_b_audit_aggregate = _mapping(stage_b_audit.get("recomputed_aggregate"))

    record(
        "stage_a_passed_and_only_stage_b_authorized",
        stage_a.get("passed") is True
        and _mapping(stage_a.get("recomputed_aggregate")).get("stage_b_authorized") is True
        and stage_a.get("p8_qualification_authorized") is False
        and stage_a.get("submission_authorized") is False,
    )
    record(
        "proposal_bank_passed_without_claim_authority",
        bank.get("passed") is True
        and bank.get("performance_claim_authorized") is False
        and bank.get("p8_qualification_authorized") is False
        and bank.get("submission_authorized") is False,
    )
    record(
        "stage_b_scientific_failure_exact",
        stage_b_aggregate.get("exactness_pass") is True
        and stage_b_aggregate.get("dcs_accuracy_pass") is True
        and stage_b_aggregate.get("mechanism_pass") is False
        and math.isclose(
            float(stage_b_aggregate.get("mechanism_geometric_variance_ratio", math.nan)),
            1.908641712567905,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and stage_b_aggregate.get("primary_accuracy_pass") is False
        and stage_b_aggregate.get("primary_resource_censoring_count") == 24
        and _mapping(stage_b_result.get("decision")).get("stage_c_authorized") is False,
    )
    record(
        "stage_b_independent_audit_passed_and_stop_preserved",
        stage_b_audit.get("passed") is True
        and stage_b_audit.get("stage_c_authorized") is False
        and stage_b_audit.get("p8_qualification_authorized") is False
        and stage_b_audit.get("submission_authorized") is False
        and stage_b_audit_aggregate == stage_b_aggregate,
    )
    t1_checks = _mapping(t1.get("checks"))
    record(
        "t1_passed_but_submission_blocked",
        t1.get("passed") is True
        and t1_checks.get("top_journal_blocked") is True
        and t1_checks.get("submission_blocked") is True
        and t1_checks.get("external_review_required") is True
        and t1_checks.get("database_search_required") is True,
    )

    record(
        "closed_obligations_exact", _strings(ledger.get("closed_obligations")) == EXPECTED_CLOSED
    )
    record(
        "stopped_obligations_exact", _strings(ledger.get("stopped_obligations")) == EXPECTED_STOPPED
    )
    record(
        "external_obligations_exact",
        _strings(ledger.get("external_obligations")) == EXPECTED_EXTERNAL,
    )
    record(
        "prohibited_claims_exact", _strings(ledger.get("prohibited_claims")) == EXPECTED_PROHIBITED
    )
    record(
        "obligation_sets_disjoint",
        not (
            (
                _strings(ledger.get("closed_obligations"))
                & _strings(ledger.get("stopped_obligations"))
            )
            or (
                _strings(ledger.get("closed_obligations"))
                & _strings(ledger.get("external_obligations"))
            )
            or (
                _strings(ledger.get("stopped_obligations"))
                & _strings(ledger.get("external_obligations"))
            )
        ),
    )
    record("decision_keys_exact", set(decision) == DECISION_KEYS)
    record(
        "decision_closes_current_program_without_claim_escalation",
        decision.get("status") == "v8_closed_by_falsification"
        and decision.get("all_currently_authorized_work_complete") is True
        and decision.get("scientific_implementation_error_found") is False
        and decision.get("broad_performance_hypothesis_passed") is False
        and all(
            decision.get(key) is False
            for key in (
                "stage_c_authorized",
                "p8_qualification_authorized",
                "p9_freeze_authorized",
                "p10_confirmation_authorized",
                "p11_reproduction_authorized",
                "submission_authorized",
                "top_journal_route_authorized",
            )
        )
        and decision.get("future_v9_requires_new_protocol") is True,
    )

    failures = [name for name, passed in checks.items() if not passed]
    return {
        "schema": REPORT_SCHEMA,
        "ledger_sha256": ledger_sha256,
        "artifact_count": len(artifacts),
        "checks": checks,
        "failure_count": len(failures),
        "failures": failures,
        "passed": not failures,
        "decision": {
            "v8_closed_by_falsification": not failures,
            "all_currently_authorized_work_complete": not failures,
            "stage_c_authorized": False,
            "p8_qualification_authorized": False,
            "submission_authorized": False,
            "future_v9_requires_new_protocol": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    ledger, digest = load_ledger(args.ledger)
    report = audit_completion_status_v3(ledger, digest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
