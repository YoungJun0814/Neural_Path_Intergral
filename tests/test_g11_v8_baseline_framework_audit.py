"""Fail-closed tests for the V8 P4 baseline-framework ledger."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from experiments.g11_v8_baseline_framework_audit import (
    REPORT_SCHEMA,
    _load_ledger,
    audit_baseline_framework,
    main,
)

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "configs" / "g11_v8" / "baseline_framework_ledger_v1.yaml"
DECISION = (
    ROOT
    / "docs"
    / "audits"
    / "G11_V8_P4_BASELINE_FRAMEWORK_DECISION_2026-07-25.md"
)


def _canonical() -> tuple[dict[str, Any], str]:
    raw = LEDGER.read_bytes()
    value = yaml.safe_load(raw.decode("utf-8"))
    assert isinstance(value, dict)
    return value, hashlib.sha256(raw).hexdigest()


def test_canonical_p4_framework_passes_all_checks() -> None:
    ledger, digest = _canonical()
    report = audit_baseline_framework(ledger, digest)

    assert report["schema"] == REPORT_SCHEMA
    assert report["check_count"] >= 38
    assert report["method_count"] == 8
    assert report["failure_count"] == 0
    assert report["failures"] == []
    assert report["passed"] is True
    assert all(report["checks"].values())


def test_p4_decision_binds_canonical_ledger_hash() -> None:
    _, digest = _canonical()

    assert digest in DECISION.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("mutation", "failed"),
    [
        (
            lambda value: value.__setitem__("outcome_data_used", True),
            "outcome_blind",
        ),
        (
            lambda value: value.__setitem__(
                "rate_complexity_ledger_sha256", "0" * 64
            ),
            "upstream_hash_bound",
        ),
        (
            lambda value: value["methods"].pop(),
            "method_count_exact",
        ),
        (
            lambda value: value["methods"][5].__setitem__(
                "inferential_unit", "iid_path"
            ),
            "inferential_units_exact",
        ),
        (
            lambda value: value["flow_constraint"].__setitem__(
                "dcs_extension_claim_authorized", True
            ),
            "flow_dcs_claim_refused",
        ),
        (
            lambda value: value["cost_categories"].remove("failed_restarts"),
            "cost_categories_exact",
        ),
        (
            lambda value: value["decision"].__setitem__(
                "numerical_performance_qualified", True
            ),
            "performance_not_qualified",
        ),
        (
            lambda value: value.__setitem__("shadow_baseline", "best"),
            "root_keys_exact",
        ),
    ],
)
def test_corruption_fails_closed(mutation, failed: str) -> None:
    ledger, digest = _canonical()
    corrupted = copy.deepcopy(ledger)
    mutation(corrupted)
    report = audit_baseline_framework(corrupted, digest)

    assert report["passed"] is False
    assert failed in report["failures"]


def test_loader_rejects_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("schema: unknown\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected baseline"):
        _load_ledger(path)


def test_cli_writes_pass_and_refuses_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "audit.json"
    monkeypatch.setattr(
        "sys.argv",
        ["audit", "--ledger", str(LEDGER), "--output", str(output)],
    )
    main()
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is True

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main()


def test_cli_writes_failure_before_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, _ = _canonical()
    ledger["decision"]["superiority_claim_authorized"] = True
    source = tmp_path / "bad.yaml"
    source.write_text(yaml.safe_dump(ledger, sort_keys=False), encoding="utf-8")
    output = tmp_path / "audit.json"
    monkeypatch.setattr(
        "sys.argv",
        ["audit", "--ledger", str(source), "--output", str(output)],
    )
    with pytest.raises(SystemExit) as caught:
        main()
    assert caught.value.code == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert "superiority_refused" in report["failures"]
