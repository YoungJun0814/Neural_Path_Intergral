"""V6 achieved-RMSE baseline smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from experiments.g11_v6_baseline_qualification import (
    _design_from_pilot,
    _load_config,
    run,
)
from experiments.g11_v6_rarity_calibration import run as run_calibration
from experiments.g11_v6_reference import run as run_reference

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / "configs" / "g11_v6" / "rarity_calibration_development.yaml"
REFERENCE = ROOT / "configs" / "g11_v6" / "reference_development.yaml"
BASELINE = ROOT / "configs" / "g11_v6" / "baseline_qualification_development.yaml"
PRIMARY_PILOT = ROOT / "configs" / "g11_v6" / "baseline_primary_resource_pilot_v2.yaml"
QUALIFICATION_V2 = (
    ROOT / "configs" / "g11_v6" / "baseline_qualification_v2.yaml"
)
QUALIFICATION_V3 = (
    ROOT / "configs" / "g11_v6" / "baseline_qualification_v3.yaml"
)
QUALIFICATION_V4 = (
    ROOT / "configs" / "g11_v6" / "baseline_qualification_v4.yaml"
)


def test_v6_baseline_config_is_strict() -> None:
    config, digest = _load_config(BASELINE)
    assert config["sampling"]["relative_sampling_rmse"] == 0.20
    assert len(digest) == 64

    primary, primary_digest = _load_config(PRIMARY_PILOT)
    assert primary["schema"] == "npi.g11.v6-baseline-qualification.config.v2"
    assert primary["methods"] == ["pure_cem"]
    # The CEM retains scores above this CDF quantile.  Therefore 0.90,
    # rather than 0.10, implements the intended upper approximately 10%.
    assert primary["training"]["elite_quantile"] == 0.90
    assert len(primary_digest) == 64

    qualification, qualification_digest = _load_config(QUALIFICATION_V2)
    assert qualification["protocol_id"] == "g11-v6-baseline-qualification-v2"
    assert qualification["phase"] == "qualification"
    assert qualification["frozen"]
    assert qualification["training"]["elite_quantile"] == 0.90
    assert len(qualification_digest) == 64

    qualification_v3, qualification_v3_digest = _load_config(QUALIFICATION_V3)
    assert qualification_v3["protocol_id"] == "g11-v6-baseline-qualification-v3"
    assert qualification_v3["defensive_design"] == {
        "nominal_probability_upper_multiplier": 2.0,
        "reference_certificate_z": 4.0,
    }
    assert len(qualification_v3_digest) == 64

    qualification_v4, qualification_v4_digest = _load_config(QUALIFICATION_V4)
    assert qualification_v4["protocol_id"] == "g11-v6-baseline-qualification-v4"
    assert qualification_v4["rarity_band_design"] == {
        "nominal_probability_upper_multiplier": 2.0,
        "reference_certificate_z": 4.0,
    }
    assert len(qualification_v4_digest) == 64


def test_v6_defensive_design_uses_frozen_rarity_band_second_moment_bound() -> None:
    values = torch.tensor([0.0, 0.0, 0.0, 0.01], dtype=torch.float64)
    uncapped = _design_from_pilot(
        "defensive_cem",
        values,
        cost_per_sample=1.0,
        nominal_probability=1.0e-4,
        confidence_level=0.95,
        pure_safety=5.0,
        defensive_bound=5.0,
        bounded_alpha=0.05,
    )
    capped = _design_from_pilot(
        "defensive_cem",
        values,
        cost_per_sample=1.0,
        nominal_probability=1.0e-4,
        confidence_level=0.95,
        pure_safety=5.0,
        defensive_bound=5.0,
        bounded_alpha=0.05,
        defensive_probability_upper=2.0e-4,
    )
    assert capped.design_variance <= 5.0 * 2.0e-4
    assert capped.design_variance >= capped.pilot_variance
    assert capped.design_variance < uncapped.design_variance


def test_v6_crude_design_uses_frozen_rarity_band_variance_bound() -> None:
    values = torch.tensor([0.0] * 4095 + [1.0], dtype=torch.float64)
    pilot_only = _design_from_pilot(
        "crude",
        values,
        cost_per_sample=1.0,
        nominal_probability=1.0e-3,
        confidence_level=0.95,
        pure_safety=5.0,
        defensive_bound=5.0,
        bounded_alpha=0.05,
    )
    rarity_band = _design_from_pilot(
        "crude",
        values,
        cost_per_sample=1.0,
        nominal_probability=1.0e-3,
        confidence_level=0.95,
        pure_safety=5.0,
        defensive_bound=5.0,
        bounded_alpha=0.05,
        crude_probability_upper=2.0e-3,
    )
    assert rarity_band.design_variance == pytest.approx(2.0e-3 * (1.0 - 2.0e-3))
    assert rarity_band.design_variance > pilot_only.design_variance


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
    assert "all_cem_fits_converged" in result["gates"]
    assert result["gates"]["all_cem_controls_finite_and_bounded"]
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
