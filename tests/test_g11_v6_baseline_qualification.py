"""V6 achieved-RMSE baseline smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.g11_v6_baseline_qualification import _load_config, run
from experiments.g11_v6_rarity_calibration import run as run_calibration
from experiments.g11_v6_reference import run as run_reference

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / "configs" / "g11_v6" / "rarity_calibration_development.yaml"
REFERENCE = ROOT / "configs" / "g11_v6" / "reference_development.yaml"
BASELINE = ROOT / "configs" / "g11_v6" / "baseline_qualification_development.yaml"


def test_v6_baseline_config_is_strict() -> None:
    config, digest = _load_config(BASELINE)
    assert config["sampling"]["relative_sampling_rmse"] == 0.20
    assert len(digest) == 64


@pytest.mark.slow
def test_v6_baseline_smoke_executes_actual_allocations(tmp_path: Path) -> None:
    calibration = run_calibration(CALIBRATION, smoke=True)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(calibration["candidate_manifest"]), encoding="utf-8")
    reference = run_reference(REFERENCE, manifest_path, smoke=True)
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(reference), encoding="utf-8")
    checkpoint_directory = tmp_path / "baseline-progress"
    result = run(
        BASELINE,
        manifest_path,
        reference_path,
        smoke=True,
        checkpoint_directory=checkpoint_directory,
    )
    assert result["schema"] == "npi.g11.v6-baseline-qualification.v1"
    assert len(result["records"]) == 6
    assert result["gates"]["complete_matrix"]
    assert result["gates"]["all_cem_training_charged"]
    assert not result["baseline_qualified"]
    assert {record["method"] for record in result["records"]} == {
        "crude",
        "pure_cem",
        "defensive_cem",
    }
    resumed = run(
        BASELINE,
        manifest_path,
        reference_path,
        smoke=True,
        checkpoint_directory=checkpoint_directory,
        resume=True,
    )
    assert [record["result"]["result_hash"] for record in resumed["records"]] == [
        record["result"]["result_hash"] for record in result["records"]
    ]
