"""Strict tests for outcome-blind V6 manifest freezing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.g11_v6_freeze_manifest import freeze_manifest
from src.path_integral import V6_CELL_MANIFEST_SCHEMA, V6CellManifest, V6RBergomiCell


def _candidate(*, dirty: bool = False) -> V6CellManifest:
    cell = V6RBergomiCell(
        cell_id="h012-terminal-p1e-03",
        hurst=0.12,
        eta=1.1,
        xi=0.04,
        rho=-0.6,
        spot=100.0,
        maturity=0.25,
        finest_steps=128,
        task="terminal_left_tail",
        event_threshold=70.0,
        nominal_probability=1e-3,
        probability_band=(0.5e-3, 2e-3),
    )
    return V6CellManifest(
        schema=V6_CELL_MANIFEST_SCHEMA,
        protocol="g11-v6-calibration-development-test",
        phase="development",
        frozen=False,
        source_commit="a" * 40,
        dirty_tree=dirty,
        config_sha256="b" * 64,
        smoke=False,
        cells=(cell,),
    )


def _write_calibration(path: Path, candidate: V6CellManifest) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "npi.g11.v6-rarity-calibration.v1",
                "passed": True,
                "smoke": False,
                "source_commit": candidate.source_commit,
                "dirty_worktree": candidate.dirty_tree,
                "candidate_manifest": candidate.to_dict(),
                "candidate_manifest_sha256": candidate.sha256,
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("phase", ["qualification", "confirmation"])
def test_freeze_manifest_changes_only_protocol_metadata(tmp_path: Path, phase: str) -> None:
    candidate = _candidate()
    calibration = tmp_path / "calibration.json"
    _write_calibration(calibration, candidate)
    frozen, receipt = freeze_manifest(calibration, phase=phase)
    assert frozen.phase == phase
    assert frozen.frozen and not frozen.dirty_tree and not frozen.smoke
    assert frozen.cells == candidate.cells
    assert receipt["estimands_unchanged"]
    assert receipt["frozen_manifest_sha256"] == frozen.sha256


def test_freeze_manifest_rejects_dirty_calibration(tmp_path: Path) -> None:
    candidate = _candidate(dirty=True)
    calibration = tmp_path / "calibration.json"
    _write_calibration(calibration, candidate)
    with pytest.raises(ValueError, match="clean"):
        freeze_manifest(calibration, phase="qualification")
