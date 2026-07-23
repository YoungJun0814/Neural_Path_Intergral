"""Seed-block split audit tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.g11_v6_split_audit import audit_stage_splits
from src.path_integral import SeedKey, SeedLedger


def _artifact(path: Path, *, protocol: str, role: str) -> None:
    ledger = SeedLedger()
    ledger.allocate(SeedKey(protocol, role, "cell", "task", 0, 0, "proposal"))
    path.write_text(
        json.dumps(
            {
                "seed_ledger": ledger.to_dict(),
                "seed_ledger_sha256": ledger.sha256,
                "records": [{"cell_id": "shared-cell"}],
            }
        ),
        encoding="utf-8",
    )


def test_split_audit_allows_cell_overlap_when_seed_blocks_are_disjoint(
    tmp_path: Path,
) -> None:
    training = tmp_path / "training.json"
    qualification = tmp_path / "qualification.json"
    _artifact(training, protocol="g11-v6-training-v1", role="training")
    _artifact(
        qualification,
        protocol="g11-v6-qualification-v1",
        role="qualification",
    )
    result = audit_stage_splits(
        {"proposal_training": training, "qualification": qualification}
    )
    assert result["passed"]
    assert result["pairwise"][0]["shared_cell_ids"] == ["shared-cell"]
    assert result["pairwise"][0]["disjoint_protocol_namespaces"]


def test_split_audit_rejects_protocol_namespace_reuse(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _artifact(first, protocol="g11-v6-reused-v1", role="first")
    _artifact(second, protocol="g11-v6-reused-v1", role="second")
    with pytest.raises(ValueError, match="reuse"):
        audit_stage_splits({"first": first, "second": second})
