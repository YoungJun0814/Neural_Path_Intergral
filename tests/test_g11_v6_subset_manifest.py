"""Phase-preserving V6 manifest subset tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.g11_v6_subset_manifest import subset_manifest
from src.path_integral import (
    V6_CELL_MANIFEST_SCHEMA,
    V6CellManifest,
    V6RBergomiCell,
)


def _cell(cell_id: str, task: str) -> V6RBergomiCell:
    return V6RBergomiCell(
        cell_id=cell_id,
        hurst=0.12,
        eta=1.1,
        xi=0.04,
        rho=-0.6,
        spot=100.0,
        maturity=0.25,
        finest_steps=128,
        task=task,
        event_threshold=85.0,
        nominal_probability=1.0e-4,
        probability_band=(5.0e-5, 2.0e-4),
    )


def _manifest(path: Path) -> V6CellManifest:
    manifest = V6CellManifest(
        schema=V6_CELL_MANIFEST_SCHEMA,
        protocol="g11-v6-test-qualification",
        phase="qualification",
        frozen=True,
        source_commit="a" * 40,
        dirty_tree=False,
        config_sha256="b" * 64,
        smoke=False,
        cells=(
            _cell("terminal", "terminal_left_tail"),
            _cell("barrier", "discrete_lower_barrier"),
        ),
    )
    path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    return manifest


def test_subset_preserves_frozen_phase_and_exact_estimand(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    original = _manifest(path)
    subset, receipt = subset_manifest(
        path,
        ("barrier",),
        protocol="g11-v6-test-resource-subset",
    )
    assert subset.phase == "qualification"
    assert subset.frozen
    assert subset.cells == (original.cells[1],)
    assert receipt["estimands_unchanged"]
    assert receipt["subset_manifest_sha256"] == subset.sha256


def test_subset_rejects_unknown_cells(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    _manifest(path)
    with pytest.raises(ValueError, match="absent"):
        subset_manifest(
            path,
            ("missing",),
            protocol="g11-v6-test-resource-subset",
        )
