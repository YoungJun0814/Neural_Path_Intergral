"""Smoke and schema tests for V6 rare-cell calibration."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from experiments.g11_v6_rarity_calibration import _load, run
from src.path_integral import V6CellManifest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "g11_v6" / "rarity_calibration_development.yaml"


def test_v6_calibration_config_is_strict() -> None:
    config, digest = _load(CONFIG)
    assert config["target_probabilities"] == [1e-2, 1e-3, 1e-4]
    assert len(digest) == 64


def test_v6_calibration_rejects_unknown_config_field(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["oracle_probability"] = 1e-3
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fields"):
        _load(path)


@pytest.mark.slow
def test_v6_calibration_smoke_emits_strict_candidate_manifest() -> None:
    result = run(CONFIG, smoke=True)
    assert result["schema"] == "npi.g11.v6-rarity-calibration.v1"
    assert len(result["cells"]) == 6
    assert result["formal_readiness"] is False
    assert result["continuous_time_claim"] is False
    assert result["gates"]["calibration_validation_seed_roles_disjoint"]
    manifest = V6CellManifest.from_dict(result["candidate_manifest"])
    assert manifest.smoke and manifest.phase == "development"
    assert manifest.sha256 == result["candidate_manifest_sha256"]
