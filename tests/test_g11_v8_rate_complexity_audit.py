"""Fail-closed tests for the V8 P3 rate and complexity ledger."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from experiments.g11_v8_rate_complexity_audit import (
    EXPECTED_SCHEMA,
    REPORT_SCHEMA,
    _load_ledger,
    audit_rate_complexity_ledger,
    main,
)

ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = ROOT / "configs" / "g11_v8" / "rate_complexity_ledger_v1.yaml"
DECISION_PATH = (
    ROOT
    / "docs"
    / "audits"
    / "G11_V8_P3_RATE_COMPLEXITY_DECISION_2026-07-25.md"
)


def _canonical() -> tuple[dict[str, Any], str]:
    raw = LEDGER_PATH.read_bytes()
    ledger = yaml.safe_load(raw.decode("utf-8"))
    assert isinstance(ledger, dict)
    return ledger, hashlib.sha256(raw).hexdigest()


def test_canonical_rate_complexity_ledger_passes() -> None:
    ledger, digest = _canonical()
    report = audit_rate_complexity_ledger(ledger, digest)

    assert report["schema"] == REPORT_SCHEMA
    assert report["ledger_schema"] == EXPECTED_SCHEMA
    assert report["ledger_sha256"] == digest
    assert report["check_count"] >= 40
    assert report["evidence_path_count"] >= 7
    assert report["failure_count"] == 0
    assert report["failures"] == []
    assert report["passed"] is True
    assert all(report["checks"].values())


def test_p3_decision_binds_canonical_rate_complexity_ledger_hash() -> None:
    _, digest = _canonical()

    assert digest in DECISION_PATH.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        (
            lambda ledger: ledger.__setitem__("outcome_data_used", True),
            "outcome_blind",
        ),
        (
            lambda ledger: ledger.__setitem__(
                "theorem_ledger_sha256", "0" * 64
            ),
            "theorem_ledger_hash_bound",
        ),
        (
            lambda ledger: ledger.__setitem__(
                "proof_document_sha256", "0" * 64
            ),
            "proof_document_hash_bound",
        ),
        (
            lambda ledger: ledger["finite_grid_decomposition"][
                "signed_terms"
            ].pop(),
            "decomposition_terms_exact",
        ),
        (
            lambda ledger: ledger["terminal_rate_chain"]["weak_bias"].__setitem__(
                "evidence", "proved_internal"
            ),
            "alpha_conditional",
        ),
        (
            lambda ledger: ledger["terminal_rate_chain"].__setitem__(
                "unconditional_claim_authorized", True
            ),
            "terminal_unconditional_claim_refused",
        ),
        (
            lambda ledger: ledger["barrier_rate"].__setitem__(
                "complexity_claim_authorized", True
            ),
            "barrier_complexity_refused",
        ),
        (
            lambda ledger: ledger["barrier_rate"][
                "unresolved_obligations"
            ].pop(),
            "barrier_obligations_exact",
        ),
        (
            lambda ledger: ledger["evidence_policy"].__setitem__(
                "empirical_slopes_are_proof", True
            ),
            "empirical_not_proof",
        ),
        (
            lambda ledger: ledger["claim_levels"].__setitem__(
                "C4_barrier_model_rate", "proved"
            ),
            "claim_levels_exact",
        ),
        (
            lambda ledger: ledger["prohibited_claims"].remove(
                "continuous_barrier_exactness"
            ),
            "prohibited_claims_exact",
        ),
        (
            lambda ledger: ledger["decision"].__setitem__(
                "submission_complexity_claim_authorized", True
            ),
            "submission_complexity_not_authorized",
        ),
        (
            lambda ledger: ledger["finite_grid_decomposition"][
                "code_evidence"
            ].__setitem__(0, "../outside.py"),
            "evidence_paths_well_formed",
        ),
        (
            lambda ledger: ledger.__setitem__("shadow_rate", "proved"),
            "root_keys_exact",
        ),
    ],
)
def test_rate_complexity_ledger_corruption_fails_closed(
    mutation,
    failed_check: str,
) -> None:
    ledger, digest = _canonical()
    corrupted = copy.deepcopy(ledger)
    mutation(corrupted)

    report = audit_rate_complexity_ledger(corrupted, digest)

    assert report["passed"] is False
    assert failed_check in report["failures"]
    assert report["checks"][failed_check] is False


def test_loader_rejects_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("schema: npi.g11.unknown\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected rate-complexity"):
        _load_ledger(path)


def test_cli_writes_passing_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "p3-audit.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "g11_v8_rate_complexity_audit.py",
            "--ledger",
            str(LEDGER_PATH),
            "--output",
            str(output),
        ],
    )

    main()
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["schema"] == REPORT_SCHEMA
    assert report["passed"] is True


def test_cli_writes_failure_before_nonzero_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, _ = _canonical()
    ledger["decision"]["submission_complexity_claim_authorized"] = True
    corrupted_path = tmp_path / "corrupted.yaml"
    corrupted_path.write_text(
        yaml.safe_dump(ledger, sort_keys=False),
        encoding="utf-8",
    )
    output = tmp_path / "failed-audit.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "g11_v8_rate_complexity_audit.py",
            "--ledger",
            str(corrupted_path),
            "--output",
            str(output),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert "submission_complexity_not_authorized" in report["failures"]


def test_cli_refuses_to_overwrite_existing_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "p3-audit.json"
    output.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "g11_v8_rate_complexity_audit.py",
            "--ledger",
            str(LEDGER_PATH),
            "--output",
            str(output),
        ],
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main()
