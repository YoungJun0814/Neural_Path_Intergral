"""Route B V6 diagnostic experiment tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.g11_v6_rarity_calibration import run as run_calibration
from experiments.g11_v6_theory_diagnostics import _load_config, run

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / "configs" / "g11_v6" / "rarity_calibration_development.yaml"
CONFIG = ROOT / "configs" / "g11_v6" / "theory_diagnostics_development.yaml"


def test_v6_theory_diagnostic_config_is_strict() -> None:
    config, digest = _load_config(CONFIG)
    assert config["target_probability"] == 1e-3
    assert len(digest) == 64


@pytest.mark.slow
def test_v6_theory_diagnostic_smoke_keeps_claim_conditional(tmp_path: Path) -> None:
    calibration = run_calibration(CALIBRATION, smoke=True)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(calibration["candidate_manifest"]), encoding="utf-8")
    result = run(CONFIG, manifest_path, smoke=True)
    assert result["schema"] == "npi.g11.v6-theory-diagnostics.v1"
    assert "inverse-slope moments" in result["claim_scope"]
    assert "not a model-rate proof" in result["claim_scope"]
    assert result["gates"]["direction_geometry"]
    assert result["gates"]["pathwise_exactness"]
    assert result["gates"]["terminal_analytic_inverse_moment_bounds_finite"]
    assert not result["diagnostics_qualified"]
