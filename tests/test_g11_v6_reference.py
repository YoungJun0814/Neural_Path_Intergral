"""Strict V6 reference artifact tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from experiments.g11_v6_rarity_calibration import run as run_calibration
from experiments.g11_v6_reference import _load_config, _load_v2_state, _v2_state_payload
from experiments.g11_v6_reference import run as run_reference
from src.path_integral import OnlineMoments, SeedLedger

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / "configs" / "g11_v6" / "rarity_calibration_development.yaml"
REFERENCE = ROOT / "configs" / "g11_v6" / "reference_development.yaml"
REFERENCE_V2 = ROOT / "configs" / "g11_v6" / "reference_development_v2.yaml"


def test_v6_reference_config_is_strict(tmp_path: Path) -> None:
    config, digest = _load_config(REFERENCE)
    assert config["phase"] == "development"
    assert len(digest) == 64

    payload = yaml.safe_load(REFERENCE.read_text(encoding="utf-8"))
    payload["reference_probability"] = 1e-3
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fields"):
        _load_config(invalid)


def test_v6_reference_v2_config_declares_independent_planning() -> None:
    config, digest = _load_config(REFERENCE_V2)
    assert config["schema"] == "npi.g11.v6-reference.config.v2"
    assert config["sampling"]["pilot_replicates"] >= 3
    assert config["sampling"]["allocation_variance_statistic"] == (
        "median_replicate_variance"
    )
    assert config["sampling"]["raw_crosscheck_standard_error_multiplier"] >= 1.0
    assert len(digest) == 64


def test_v6_reference_v2_checkpoint_round_trip_and_tamper_rejection(tmp_path: Path) -> None:
    identity = {
        "config_sha256": "1" * 64,
        "manifest_sha256": "2" * 64,
        "protocol_id": "test",
        "cell": {"cell_id": "cell"},
        "method": "dcs_reference",
        "target_standard_error": 1e-5,
        "smoke": False,
    }
    contribution = OnlineMoments(count=3, mean=0.2, m2=0.1)
    normalization = OnlineMoments(count=3, mean=1.0, m2=0.2)
    payload = _v2_state_payload(
        identity=identity,
        pilot_variances=[0.1, 0.2, 0.3],
        requested_final_samples=5,
        final_samples=5,
        contribution_moments=contribution,
        normalization_moments=normalization,
        chunks=[{"offset": 0, "count": 3}],
        ledger=SeedLedger(),
    )
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    restored = _load_v2_state(path, identity=identity, pilot_replicates=3)
    assert restored[0] == [0.1, 0.2, 0.3]
    assert restored[3] == contribution
    assert restored[4] == normalization

    payload["contribution_moments"]["count"] = 4
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="counts disagree"):
        _load_v2_state(path, identity=identity, pilot_replicates=3)


def test_v6_reference_v2_resume_requires_checkpoint_directory() -> None:
    with pytest.raises(ValueError, match="checkpoint directory"):
        run_reference(REFERENCE_V2, Path("unused.json"), resume=True)


@pytest.mark.slow
def test_v6_reference_v2_completed_resume_is_exact(tmp_path: Path) -> None:
    calibration = run_calibration(CALIBRATION, smoke=True)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(calibration["candidate_manifest"]), encoding="utf-8")
    checkpoints = tmp_path / "checkpoints"
    initial = run_reference(
        REFERENCE_V2,
        manifest_path,
        smoke=True,
        checkpoint_directory=checkpoints,
    )
    resumed = run_reference(
        REFERENCE_V2,
        manifest_path,
        smoke=True,
        checkpoint_directory=checkpoints,
        resume=True,
    )
    assert resumed == initial
    with pytest.raises(FileExistsError, match="existing checkpoint"):
        run_reference(
            REFERENCE_V2,
            manifest_path,
            smoke=True,
            checkpoint_directory=checkpoints,
        )


@pytest.mark.slow
def test_v6_reference_smoke_uses_disjoint_methods_and_never_qualifies(tmp_path: Path) -> None:
    calibration = run_calibration(CALIBRATION, smoke=True)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(calibration["candidate_manifest"]), encoding="utf-8")
    result = run_reference(REFERENCE, manifest_path, smoke=True)
    assert result["schema"] == "npi.g11.v6-reference.v1"
    assert len(result["cells"]) == 2
    assert not result["reference_qualified"]
    assert not result["formal_readiness"]["non_smoke"]
    roles = {
        record["key"]["role"] for record in result["seed_ledger"]["records"]
    }
    assert any(role.startswith("reference-a") for role in roles)
    assert any(role.startswith("reference-b") for role in roles)
    assert all(cell["target_standard_error"] > 0.0 for cell in result["cells"])
