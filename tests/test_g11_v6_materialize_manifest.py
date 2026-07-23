"""V6 calibrated-manifest materialization tests."""

from __future__ import annotations

import json

import pytest

from experiments.g11_v6_materialize_manifest import run
from src.path_integral import V6_CELL_MANIFEST_SCHEMA, V6CellManifest, V6RBergomiCell


def _cell(cell_id: str, probability: float) -> V6RBergomiCell:
    return V6RBergomiCell(
        cell_id=cell_id,
        hurst=0.1,
        eta=1.1,
        xi=0.04,
        rho=-0.6,
        spot=100.0,
        maturity=0.25,
        finest_steps=128,
        task="terminal_left_tail",
        event_threshold=60.0,
        nominal_probability=probability,
        probability_band=(0.5 * probability, 2.0 * probability),
    )


def test_materializer_preserves_hash_and_applies_declared_probability_filter(tmp_path) -> None:
    manifest = V6CellManifest(
        schema=V6_CELL_MANIFEST_SCHEMA,
        protocol="g11-v6-materialize-test",
        phase="development",
        frozen=False,
        source_commit="1" * 40,
        dirty_tree=True,
        config_sha256="2" * 64,
        smoke=False,
        cells=(_cell("rare", 1e-3), _cell("sentinel", 1e-4)),
    )
    calibration = {
        "schema": "npi.g11.v6-rarity-calibration.v1",
        "passed": True,
        "candidate_manifest": manifest.to_dict(),
        "candidate_manifest_sha256": manifest.sha256,
    }
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
    output = tmp_path / "manifest.json"
    receipt = run(
        calibration_path,
        output,
        minimum_nominal_probability=1e-3,
    )
    materialized = V6CellManifest.from_dict(json.loads(output.read_text(encoding="utf-8")))
    assert receipt["cell_count"] == 1
    assert materialized.cells[0].cell_id == "rare"
    assert materialized.sha256 == receipt["manifest_sha256"]


def test_materializer_selects_declared_cells_in_manifest_order(tmp_path) -> None:
    manifest = V6CellManifest(
        schema=V6_CELL_MANIFEST_SCHEMA,
        protocol="g11-v6-materialize-test",
        phase="development",
        frozen=False,
        source_commit="1" * 40,
        dirty_tree=True,
        config_sha256="2" * 64,
        smoke=False,
        cells=(_cell("first", 1e-2), _cell("second", 1e-3), _cell("third", 1e-4)),
    )
    calibration = {
        "schema": "npi.g11.v6-rarity-calibration.v1",
        "passed": True,
        "candidate_manifest": manifest.to_dict(),
        "candidate_manifest_sha256": manifest.sha256,
    }
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
    output = tmp_path / "manifest.json"
    receipt = run(
        calibration_path,
        output,
        selected_cell_ids=("third", "first"),
    )
    materialized = V6CellManifest.from_dict(json.loads(output.read_text(encoding="utf-8")))
    assert receipt["selected_cell_ids"] == ["third", "first"]
    assert [cell.cell_id for cell in materialized.cells] == ["first", "third"]
    assert materialized.sha256 == receipt["manifest_sha256"]


@pytest.mark.parametrize(
    ("selected_cell_ids", "match"),
    [
        ((), "must not be empty"),
        (("first", "first"), "must be unique"),
        (("missing",), "absent from the manifest"),
    ],
)
def test_materializer_rejects_invalid_cell_selection(
    tmp_path, selected_cell_ids, match
) -> None:
    manifest = V6CellManifest(
        schema=V6_CELL_MANIFEST_SCHEMA,
        protocol="g11-v6-materialize-test",
        phase="development",
        frozen=False,
        source_commit="1" * 40,
        dirty_tree=True,
        config_sha256="2" * 64,
        smoke=False,
        cells=(_cell("first", 1e-2),),
    )
    calibration = {
        "schema": "npi.g11.v6-rarity-calibration.v1",
        "passed": True,
        "candidate_manifest": manifest.to_dict(),
        "candidate_manifest_sha256": manifest.sha256,
    }
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        run(
            calibration_path,
            tmp_path / "manifest.json",
            selected_cell_ids=selected_cell_ids,
        )
