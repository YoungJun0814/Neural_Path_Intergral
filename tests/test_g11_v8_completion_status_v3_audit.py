from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from experiments.g11_v8_completion_status_v3_audit import (
    REPORT_SCHEMA,
    audit_completion_status_v3,
    load_ledger,
    main,
)

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "configs" / "g11_v8" / "completion_status_ledger_v3.yaml"


def _canonical() -> tuple[dict[str, object], str]:
    raw = LEDGER.read_bytes()
    payload = yaml.safe_load(raw.decode("utf-8"))
    assert isinstance(payload, dict)
    return payload, hashlib.sha256(raw).hexdigest()


def test_final_v8_completion_status_closes_only_the_falsified_program() -> None:
    ledger, digest = _canonical()
    report = audit_completion_status_v3(ledger, digest)

    assert report["schema"] == REPORT_SCHEMA
    assert report["ledger_sha256"] == digest
    assert report["artifact_count"] == 10
    assert report["failure_count"] == 0
    assert report["failures"] == []
    assert report["passed"] is True
    assert all(report["checks"].values())
    assert report["decision"] == {
        "v8_closed_by_falsification": True,
        "all_currently_authorized_work_complete": True,
        "stage_c_authorized": False,
        "p8_qualification_authorized": False,
        "submission_authorized": False,
        "future_v9_requires_new_protocol": True,
    }


@pytest.mark.parametrize(
    ("mutation", "failure"),
    [
        (
            lambda ledger: ledger["predecessor"].__setitem__("sha256", "0" * 64),
            "predecessor_hash_bound",
        ),
        (
            lambda ledger: ledger["artifacts"][6].__setitem__("sha256", "0" * 64),
            "artifact_hashes_match",
        ),
        (
            lambda ledger: ledger["decision"].__setitem__(
                "broad_performance_hypothesis_passed", True
            ),
            "decision_closes_current_program_without_claim_escalation",
        ),
        (
            lambda ledger: ledger["decision"].__setitem__("stage_c_authorized", True),
            "decision_closes_current_program_without_claim_escalation",
        ),
        (
            lambda ledger: ledger["decision"].__setitem__("submission_authorized", True),
            "decision_closes_current_program_without_claim_escalation",
        ),
        (
            lambda ledger: ledger["stopped_obligations"].remove(
                "p8_independent_seed_qualification"
            ),
            "stopped_obligations_exact",
        ),
    ],
)
def test_final_closure_corruption_fails_closed(mutation, failure: str) -> None:
    ledger, digest = _canonical()
    corrupted = copy.deepcopy(ledger)
    mutation(corrupted)

    report = audit_completion_status_v3(corrupted, digest)

    assert report["passed"] is False
    assert failure in report["failures"]


def test_final_closure_cli_writes_a_passing_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "closure-audit.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "g11_v8_completion_status_v3_audit.py",
            "--ledger",
            str(LEDGER),
            "--output",
            str(output),
        ],
    )

    main()
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["passed"] is True
    assert report["decision"]["stage_c_authorized"] is False


def test_final_closure_loader_rejects_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("schema: bad\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected"):
        load_ledger(path)
