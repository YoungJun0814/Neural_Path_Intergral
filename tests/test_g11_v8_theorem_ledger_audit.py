from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from experiments.g11_v8_theorem_ledger_audit import (
    EXPECTED_SCHEMA,
    REPORT_SCHEMA,
    _load_ledger,
    audit_theorem_ledger,
    main,
)

ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = ROOT / "configs" / "g11_v8" / "theorem_ledger_v1.yaml"
V7_CONTRACT_PATH = (
    ROOT / "docs" / "theory" / "G11_V7_RAO_BLACKWELL_MECHANISM_CONTRACT.md"
)
DECISION_PATH = (
    ROOT / "docs" / "audits" / "G11_V8_P2_THEOREM_DECISION_2026-07-25.md"
)


def _canonical() -> tuple[dict[str, object], str]:
    raw = LEDGER_PATH.read_bytes()
    ledger = yaml.safe_load(raw.decode("utf-8"))
    assert isinstance(ledger, dict)
    return ledger, hashlib.sha256(raw).hexdigest()


def test_canonical_theorem_ledger_passes() -> None:
    ledger, digest = _canonical()
    report = audit_theorem_ledger(ledger, digest)

    assert report["schema"] == REPORT_SCHEMA
    assert report["ledger_sha256"] == digest
    assert report["theorem_count"] == 5
    assert report["evidence_path_count"] >= 8
    assert report["failure_count"] == 0
    assert report["failures"] == []
    assert report["passed"] is True
    assert all(report["checks"].values())


def test_v7_total_variance_typo_is_corrected() -> None:
    text = V7_CONTRACT_PATH.read_text(encoding="utf-8")

    assert (
        "=\\operatorname{Var}_Q(D)\n"
        "+E_Q[\\operatorname{Var}_Q(Y\\mid R)]"
    ) in text


def test_p2_decision_binds_canonical_theorem_ledger_hash() -> None:
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
                "claim_contract_sha256", "0" * 64
            ),
            "claim_contract_hash_bound",
        ),
        (
            lambda ledger: ledger["theorems"].pop(),
            "theorem_ids_exact",
        ),
        (
            lambda ledger: ledger.__setitem__(
                "proof_document_sha256", "0" * 64
            ),
            "proof_document_hash_bound",
        ),
        (
            lambda ledger: ledger["theorems"][3].__setitem__(
                "status", "unconditional_rbergomi_rate"
            ),
            "theorem_statuses_exact",
        ),
        (
            lambda ledger: ledger["claim_levels"].__setitem__(
                "C4_model_mesh_or_rate", "proved"
            ),
            "claim_levels_exact",
        ),
        (
            lambda ledger: ledger["open_obligations"].remove(
                "discrete_barrier_fine_only_mesh_crossings"
            ),
            "open_obligations_exact",
        ),
        (
            lambda ledger: ledger["prohibited_claims"].remove(
                "continuous_monitoring_exactness"
            ),
            "prohibited_claims_exact",
        ),
        (
            lambda ledger: ledger["decision"].__setitem__(
                "submission_theory_authorized", True
            ),
            "submission_theory_not_authorized",
        ),
        (
            lambda ledger: ledger["decision"].__setitem__(
                "external_proof_review_required", False
            ),
            "external_proof_review_still_required",
        ),
        (
            lambda ledger: ledger["theorems"][0]["code_evidence"].__setitem__(
                0, "src/path_integral/does_not_exist.py"
            ),
            "all_evidence_paths_exist",
        ),
        (
            lambda ledger: ledger.__setitem__("post_hoc_theorem", "shadow"),
            "root_keys_exact",
        ),
    ],
)
def test_theorem_ledger_corruption_fails_closed(
    mutation,
    failed_check: str,
) -> None:
    ledger, digest = _canonical()
    corrupted = copy.deepcopy(ledger)
    mutation(corrupted)

    report = audit_theorem_ledger(corrupted, digest)

    assert report["passed"] is False
    assert failed_check in report["failures"]
    assert report["checks"][failed_check] is False


def test_loader_rejects_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("schema: npi.g11.unknown\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected theorem-ledger schema"):
        _load_ledger(path)


def test_cli_writes_passing_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "theorem-audit.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "g11_v8_theorem_ledger_audit.py",
            "--ledger",
            str(LEDGER_PATH),
            "--output",
            str(output),
        ],
    )

    main()
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["schema"] == REPORT_SCHEMA
    assert report["ledger_schema"] == EXPECTED_SCHEMA
    assert report["passed"] is True


def test_cli_writes_failure_before_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger, _ = _canonical()
    ledger["decision"]["submission_theory_authorized"] = True
    corrupted_path = tmp_path / "corrupted.yaml"
    corrupted_path.write_text(
        yaml.safe_dump(ledger, sort_keys=False),
        encoding="utf-8",
    )
    output = tmp_path / "failed-audit.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "g11_v8_theorem_ledger_audit.py",
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
    assert "submission_theory_not_authorized" in report["failures"]


def test_cli_refuses_to_overwrite_existing_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "theorem-audit.json"
    output.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "g11_v8_theorem_ledger_audit.py",
            "--ledger",
            str(LEDGER_PATH),
            "--output",
            str(output),
        ],
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main()
