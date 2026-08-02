from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from experiments.g11_v8_t1_novelty_audit import (
    REPORT_SCHEMA,
    _load,
    audit_t1_novelty_update,
    main,
)

ROOT = Path(__file__).resolve().parents[1]
UPDATE = ROOT / "configs" / "g11_v8" / "t1_novelty_update_v1.yaml"


def _canonical() -> tuple[dict[str, object], str]:
    raw = UPDATE.read_bytes()
    payload = yaml.safe_load(raw.decode("utf-8"))
    assert isinstance(payload, dict)
    return payload, hashlib.sha256(raw).hexdigest()


def test_canonical_t1_novelty_update_passes() -> None:
    payload, digest = _canonical()
    report = audit_t1_novelty_update(payload, digest)

    assert report["schema"] == REPORT_SCHEMA
    assert report["payload_sha256"] == digest
    assert report["counts"] == {"interfaces": 3, "queries": 7, "sources": 7}
    assert report["failure_count"] == 0
    assert report["failures"] == []
    assert report["passed"] is True
    assert all(report["checks"].values())


@pytest.mark.parametrize(
    ("mutation", "failure"),
    [
        (
            lambda payload: payload["base_ledger"].__setitem__("sha256", "0" * 64),
            "base_ledger_hash_bound",
        ),
        (
            lambda payload: payload["sources"].pop(),
            "source_ids_exact",
        ),
        (
            lambda payload: payload["decision"].__setitem__("submission_novelty_authorized", True),
            "submission_blocked",
        ),
        (
            lambda payload: payload["decision"].__setitem__("top_journal_route_authorized", True),
            "top_journal_blocked",
        ),
        (
            lambda payload: payload["decision"].__setitem__(
                "external_expert_review_required", False
            ),
            "external_review_required",
        ),
    ],
)
def test_t1_novelty_update_corruption_fails_closed(mutation, failure: str) -> None:
    payload, digest = _canonical()
    corrupted = copy.deepcopy(payload)
    mutation(corrupted)

    report = audit_t1_novelty_update(corrupted, digest)

    assert report["passed"] is False
    assert failure in report["failures"]


def test_t1_novelty_cli_writes_passing_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "audit.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "g11_v8_t1_novelty_audit.py",
            "--update",
            str(UPDATE),
            "--output",
            str(output),
        ],
    )

    main()
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["passed"] is True


def test_t1_novelty_loader_rejects_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "unknown.yaml"
    path.write_text("schema: unknown\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected"):
        _load(path)
