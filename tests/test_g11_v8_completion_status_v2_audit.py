from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import yaml

from experiments.g11_v8_completion_status_v2_audit import audit_completion_status_v2

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "configs/g11_v8/completion_status_ledger_v2.yaml"


def _canonical() -> tuple[dict[str, object], str]:
    raw = LEDGER.read_bytes()
    value = yaml.safe_load(raw.decode("utf-8"))
    assert isinstance(value, dict)
    return value, hashlib.sha256(raw).hexdigest()


def test_r2_status_ledger_authorizes_only_b1() -> None:
    ledger, digest = _canonical()
    report = audit_completion_status_v2(ledger, digest)
    assert report["passed"] is True
    assert report["artifact_count"] == 13
    assert report["decision"] == {
        "reference_complete": True,
        "b1_implementation_authorized": True,
        "performance_claim_authorized": False,
        "p8_qualification_authorized": False,
        "submission_authorized": False,
    }


def test_r2_status_ledger_fails_closed_on_reference_or_claim_corruption() -> None:
    ledger, digest = _canonical()
    bad_hash = copy.deepcopy(ledger)
    bad_hash["artifacts"][9]["sha256"] = "0" * 64
    hash_report = audit_completion_status_v2(bad_hash, digest)
    assert hash_report["passed"] is False
    assert "artifact_hashes_match" in hash_report["failures"]

    bad_claim = copy.deepcopy(ledger)
    bad_claim["decision"]["performance_claim_authorized"] = True
    claim_report = audit_completion_status_v2(bad_claim, digest)
    assert claim_report["passed"] is False
    assert "decision_fail_closed" in claim_report["failures"]


def test_r2_status_ledger_rejects_p8_execution_authority() -> None:
    ledger, digest = _canonical()
    corrupted = copy.deepcopy(ledger)
    corrupted["authorized_actions"].append("p8_qualification_execution")
    report = audit_completion_status_v2(corrupted, digest)
    assert report["passed"] is False
    assert "authorized_actions_exact" in report["failures"]
