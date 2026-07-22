"""Strict V6 reference artifact tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from experiments.g11_v6_rarity_calibration import run as run_calibration
from experiments.g11_v6_reference import _load_config
from experiments.g11_v6_reference import run as run_reference

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / "configs" / "g11_v6" / "rarity_calibration_development.yaml"
REFERENCE = ROOT / "configs" / "g11_v6" / "reference_development.yaml"


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
