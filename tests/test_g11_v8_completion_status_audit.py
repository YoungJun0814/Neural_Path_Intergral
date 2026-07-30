from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from experiments.g11_v8_completion_status_audit import (
    REPORT_SCHEMA,
    _load_ledger,
    audit_completion_status,
    main,
)

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "configs" / "g11_v8" / "completion_status_ledger_v1.yaml"


def _canonical() -> tuple[dict[str, object], str]:
    raw = LEDGER.read_bytes()
    payload = yaml.safe_load(raw.decode("utf-8"))
    assert isinstance(payload, dict)
    return payload, hashlib.sha256(raw).hexdigest()


def test_canonical_completion_status_passes() -> None:
    payload, digest = _canonical()
    report = audit_completion_status(payload, digest)

    assert report["schema"] == REPORT_SCHEMA
    assert report["artifact_count"] == 15
    assert report["burned_namespace_count"] == 4
    assert report["reserved_namespace_count"] == 8
    assert report["passed"] is True
    assert report["failures"] == []
    assert all(report["checks"].values())


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        (
            lambda payload: payload["artifacts"][11].__setitem__("status", "passed"),
            "artifact_statuses_exact",
        ),
        (
            lambda payload: payload.__setitem__(
                "baseline_repository_commit", "0" * 40
            ),
            "baseline_commit_exact_and_ancestor",
        ),
        (
            lambda payload: payload["artifacts"][0].__setitem__("sha256", "0" * 64),
            "all_artifact_hashes_match",
        ),
        (
            lambda payload: payload["burned_namespaces"][2].__setitem__(
                "reusable", True
            ),
            "burned_namespaces_not_reusable",
        ),
        (
            lambda payload: payload["reserved_namespaces"].append("p5-reference-v2"),
            "burned_and_reserved_disjoint",
        ),
        (
            lambda payload: payload["authorized_actions"].append(
                "p8_qualification_execution"
            ),
            "authorized_actions_exact",
        ),
        (
            lambda payload: payload["decision"].__setitem__(
                "performance_claim_authorized", True
            ),
            "all_future_claims_refused",
        ),
    ],
)
def test_completion_status_corruption_fails_closed(
    mutation, failed_check: str
) -> None:
    payload, digest = _canonical()
    corrupted = copy.deepcopy(payload)
    mutation(corrupted)

    report = audit_completion_status(corrupted, digest)

    assert report["passed"] is False
    assert failed_check in report["failures"]


def test_loader_rejects_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("schema: npi.invalid\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected R0"):
        _load_ledger(path)


def test_cli_writes_report_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "audit.json"
    monkeypatch.setattr(
        "sys.argv",
        ["audit", "--ledger", str(LEDGER), "--output", str(output)],
    )
    main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["passed"] is True

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main()


def test_cli_writes_failure_receipt_before_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, _ = _canonical()
    payload["decision"]["submission_authorized"] = True
    corrupted = tmp_path / "corrupted.yaml"
    corrupted.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    output = tmp_path / "failed.json"
    monkeypatch.setattr(
        "sys.argv",
        ["audit", "--ledger", str(corrupted), "--output", str(output)],
    )

    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert "all_future_claims_refused" in report["failures"]
