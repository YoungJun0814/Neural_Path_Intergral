"""Training-inclusive V6 work-ledger tests."""

from __future__ import annotations

import pytest

from src.path_integral.v6_work_ledger import V6WorkLedger, V6WorkRecord


def _record(category="final", *, work=10.0, successful=True, attempt=0):
    return V6WorkRecord(
        category=category,
        method="v6_policy",
        cell_id="h012-terminal-p1e-03",
        attempt=attempt,
        samples=16,
        work_units=work,
        wall_seconds=0.2,
        cpu_seconds=0.3,
        peak_memory_bytes=1024,
        successful=successful,
    )


def test_v6_work_ledger_round_trip_charges_failures_and_retries() -> None:
    ledger = (
        V6WorkLedger()
        .append(_record("screening", work=2.0))
        .append(_record("failed_attempt", work=3.0, successful=False, attempt=1))
        .append(_record("retry", work=4.0, attempt=2))
        .append(_record("final", work=10.0, attempt=2))
    )
    restored = V6WorkLedger.from_dict(ledger.to_dict())
    assert restored == ledger
    assert ledger.total_work_units == 19.0
    assert ledger.total_wall_seconds == pytest.approx(0.8)
    assert ledger.total_cpu_seconds == pytest.approx(1.2)
    assert ledger.peak_memory_bytes == 1024
    assert len(ledger.sha256) == 64


def test_v6_work_ledger_rejects_unknown_fields_and_missing_required_categories() -> None:
    payload = V6WorkLedger((_record(),)).to_dict()
    payload["records"][0]["ignored"] = 1
    with pytest.raises(ValueError, match="fields"):
        V6WorkLedger.from_dict(payload)

    with pytest.raises(ValueError, match="missing"):
        V6WorkLedger((_record(),)).require_categories(("screening", "final"))


def test_failed_attempt_cannot_be_marked_successful() -> None:
    with pytest.raises(ValueError, match="cannot be successful"):
        _record("failed_attempt", successful=True)
