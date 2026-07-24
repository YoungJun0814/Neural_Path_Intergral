"""Fail-closed audit for the G11 V8 P2 theorem and proof ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

EXPECTED_SCHEMA = "npi.g11.v8-theorem-ledger.v1"
REPORT_SCHEMA = "npi.g11.v8-theorem-ledger-audit.v1"
ROOT = Path(__file__).resolve().parents[1]

EXPECTED_THEOREM_STATUS = {
    "T-V8-1": "proved_finite_dimensional",
    "T-V8-2": "proved_finite_dimensional",
    "T-V8-3": "proved_finite_dimensional",
    "T-V8-4": "proved_with_explicit_scope",
    "T-V8-5": "proved_finite_grid",
}
EXPECTED_CLAIM_LEVELS = {
    "C0_finite_grid_target": "proved",
    "C1_finite_grid_exactness": "proved",
    "C2_variance_nonincrease": "proved",
    "C3_strict_improvement": "proved_for_finite_scalar_thresholds",
    "C4_model_mesh_or_rate": "open",
    "C5_end_to_end_mlmc_complexity": "prohibited",
}
EXPECTED_OPEN_OBLIGATIONS = {
    "nonlinear_rbergomi_weak_bias",
    "discrete_barrier_fine_only_mesh_crossings",
    "correction_variance_model_rate",
    "per_sample_cost_exponent",
    "rare_event_asymptotic_ratio",
    "external_mathematical_review",
}
EXPECTED_PROHIBITED_CLAIMS = {
    "continuous_monitoring_exactness",
    "unconditional_rbergomi_weak_rate",
    "barrier_mesh_rate_without_fine_only_crossings",
    "exact_rate_doubling_without_matching_lower_bound",
    "end_to_end_mlmc_complexity_without_alpha_beta_gamma",
    "universal_or_uniform_variance_ratio",
    "training_inclusive_superiority_from_theory_alone",
}
EXPECTED_ROOT_KEYS = {
    "schema",
    "protocol_id",
    "date",
    "phase",
    "outcome_data_used",
    "claim_contract_sha256",
    "novelty_ledger_sha256",
    "proof_document",
    "proof_document_sha256",
    "theorems",
    "claim_levels",
    "open_obligations",
    "prohibited_claims",
    "decision",
}
EXPECTED_THEOREM_KEYS = {
    "id",
    "title",
    "status",
    "scope",
    "assumptions",
    "conclusion",
    "code_evidence",
    "test_evidence",
    "external_review_required",
}
EXPECTED_DECISION_KEYS = {
    "status",
    "p3_mesh_and_complexity_implementation_authorized",
    "submission_theory_authorized",
    "top_journal_theory_complete",
    "external_proof_review_required",
}
PROOF_REQUIRED_FRAGMENTS = {
    "T-V8-1",
    "T-V8-2",
    "T-V8-3",
    "T-V8-4",
    "T-V8-5",
    "Cauchy--Schwarz",
    "finite-grid",
    "continuously monitored",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_ledger(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("theorem ledger root must be a mapping")
    if payload.get("schema") != EXPECTED_SCHEMA:
        raise ValueError(f"unexpected theorem-ledger schema: {payload.get('schema')!r}")
    return payload, hashlib.sha256(raw).hexdigest()


def _records(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string_set(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {item for item in value if isinstance(item, str)}


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def audit_theorem_ledger(
    ledger: dict[str, Any],
    ledger_sha256: str,
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Audit theorem status, evidence paths, upstream hashes, and claim boundaries."""

    checks: dict[str, bool] = {}

    def record(name: str, condition: bool) -> None:
        checks[name] = bool(condition)

    theorems = _records(ledger.get("theorems"))
    theorem_by_id = {
        theorem.get("id"): theorem
        for theorem in theorems
        if isinstance(theorem.get("id"), str)
    }
    claim_levels = _mapping(ledger.get("claim_levels"))
    decision = _mapping(ledger.get("decision"))

    record("schema_exact", ledger.get("schema") == EXPECTED_SCHEMA)
    record("root_keys_exact", set(ledger) == EXPECTED_ROOT_KEYS)
    record(
        "protocol_exact",
        ledger.get("protocol_id") == "g11-v8-theorem-ledger-v1",
    )
    record("date_exact", ledger.get("date") == "2026-07-25")
    record("phase_exact", ledger.get("phase") == "p2_development")
    record("outcome_blind", ledger.get("outcome_data_used") is False)

    claim_contract = root / "configs" / "g11_v8" / "top_journal_claim_contract_v1.yaml"
    novelty_ledger = root / "configs" / "g11_v8" / "novelty_search_ledger_v1.yaml"
    record(
        "claim_contract_hash_bound",
        claim_contract.is_file()
        and ledger.get("claim_contract_sha256") == _sha256(claim_contract),
    )
    record(
        "novelty_ledger_hash_bound",
        novelty_ledger.is_file()
        and ledger.get("novelty_ledger_sha256") == _sha256(novelty_ledger),
    )

    proof_value = ledger.get("proof_document")
    proof_path = root / proof_value if isinstance(proof_value, str) else root / "__invalid__"
    proof_text = proof_path.read_text(encoding="utf-8") if proof_path.is_file() else ""
    record("proof_document_exists", proof_path.is_file())
    record(
        "proof_document_hash_bound",
        proof_path.is_file()
        and ledger.get("proof_document_sha256") == _sha256(proof_path),
    )
    record(
        "proof_fragments_complete",
        bool(proof_text)
        and all(fragment in proof_text for fragment in PROOF_REQUIRED_FRAGMENTS),
    )

    record(
        "theorem_keys_exact",
        len(theorems) == len(EXPECTED_THEOREM_STATUS)
        and all(set(theorem) == EXPECTED_THEOREM_KEYS for theorem in theorems),
    )
    record(
        "theorem_ids_exact",
        set(theorem_by_id) == set(EXPECTED_THEOREM_STATUS),
    )
    record(
        "theorem_statuses_exact",
        all(
            theorem_by_id.get(theorem_id, {}).get("status") == status
            for theorem_id, status in EXPECTED_THEOREM_STATUS.items()
        ),
    )
    record(
        "theorem_text_complete",
        bool(theorems)
        and all(
            _nonempty_text(theorem.get("title"))
            and _nonempty_text(theorem.get("scope"))
            and _nonempty_text(theorem.get("conclusion"))
            for theorem in theorems
        ),
    )
    record(
        "theorem_assumptions_complete",
        bool(theorems)
        and all(bool(_string_set(theorem.get("assumptions"))) for theorem in theorems),
    )
    record(
        "external_review_required_for_every_theorem",
        bool(theorems)
        and all(
            theorem.get("external_review_required") is True for theorem in theorems
        ),
    )

    evidence_paths: list[str] = []
    evidence_well_formed = bool(theorems)
    for theorem in theorems:
        code_paths = theorem.get("code_evidence")
        test_paths = theorem.get("test_evidence")
        if (
            not isinstance(code_paths, list)
            or not code_paths
            or not isinstance(test_paths, list)
            or not test_paths
            or not all(isinstance(item, str) for item in code_paths + test_paths)
        ):
            evidence_well_formed = False
            continue
        if not all(path.startswith("src/") for path in code_paths):
            evidence_well_formed = False
        if not all(path.startswith("tests/") for path in test_paths):
            evidence_well_formed = False
        evidence_paths.extend(code_paths + test_paths)
    record("evidence_paths_well_formed", evidence_well_formed)
    record(
        "all_evidence_paths_exist",
        bool(evidence_paths) and all((root / path).is_file() for path in evidence_paths),
    )

    record("claim_levels_exact", claim_levels == EXPECTED_CLAIM_LEVELS)
    record(
        "open_obligations_exact",
        _string_set(ledger.get("open_obligations")) == EXPECTED_OPEN_OBLIGATIONS,
    )
    record(
        "prohibited_claims_exact",
        _string_set(ledger.get("prohibited_claims"))
        == EXPECTED_PROHIBITED_CLAIMS,
    )

    record("decision_keys_exact", set(decision) == EXPECTED_DECISION_KEYS)
    record("decision_conditional_pass", decision.get("status") == "conditional_pass")
    record(
        "p3_only_authorized",
        decision.get("p3_mesh_and_complexity_implementation_authorized") is True,
    )
    record(
        "submission_theory_not_authorized",
        decision.get("submission_theory_authorized") is False,
    )
    record(
        "top_journal_theory_not_complete",
        decision.get("top_journal_theory_complete") is False,
    )
    record(
        "external_proof_review_still_required",
        decision.get("external_proof_review_required") is True,
    )

    failures = [name for name, passed in checks.items() if not passed]
    return {
        "schema": REPORT_SCHEMA,
        "ledger_schema": ledger.get("schema"),
        "ledger_sha256": ledger_sha256,
        "theorem_count": len(theorems),
        "evidence_path_count": len(set(evidence_paths)),
        "checks": checks,
        "failure_count": len(failures),
        "failures": failures,
        "passed": not failures,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {args.output}")
    ledger, ledger_sha256 = _load_ledger(args.ledger)
    report = audit_theorem_ledger(ledger, ledger_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
