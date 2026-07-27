from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from experiments.g11_v8_p7_calibration import (
    _require_finite_path_sample,
    load_config,
    weighted_threshold,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "g11_v8" / "p7_development_calibration_v1.yaml"
P5_THRESHOLD_CONFIG = ROOT / "configs" / "g11_v8" / "p5_threshold_calibration_execution_v1.yaml"
RESULT = ROOT / "results" / "g11_v8_p7_calibration_development_v1_2026-07-27.json"


def test_p7_calibration_config_binds_the_v8_matrix_and_statistics() -> None:
    config, digest = load_config(CONFIG)
    assert config["phase"] == "p7_development"
    assert config["outcome_data_used"] is False
    assert config["model"]["hurst_values"] == [0.05, 0.12, 0.20]
    assert config["nominal_probabilities"][-1] == 1e-5
    assert config["decision"]["performance_claim_authorized"] is False
    assert len(digest) == 64


def test_p5_threshold_config_uses_the_p5_namespace_not_development_namespace() -> None:
    config, digest = load_config(P5_THRESHOLD_CONFIG)
    assert config["phase"] == "p5_threshold_calibration"
    assert config["seed_namespace"] == "p5-threshold-calibration-development"
    assert config["schema"] == "npi.g11.v8-p5-threshold-calibration-execution.v1"
    assert len(digest) == 64


def test_p7_calibration_rejects_unbound_or_malformed_matrix(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["p5_reference_matrix_design_sha256"] = "0" * 64
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="P5 matrix"):
        load_config(path)


def test_weighted_threshold_uses_the_first_weighted_quantile_crossing() -> None:
    score = torch.tensor([90.0, 70.0, 80.0, 60.0], dtype=torch.float64)
    likelihood = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    assert weighted_threshold(score, likelihood, 0.50) == 70.0
    assert weighted_threshold(score, likelihood, 0.25) == 60.0
    with pytest.raises(ValueError, match="requested target mass"):
        weighted_threshold(score, torch.zeros_like(likelihood), 0.01)


def test_p7_rejects_zero_path_values_before_dcs() -> None:
    sample = SimpleNamespace(
        paths=SimpleNamespace(
            spot=torch.tensor([[100.0, 0.0]], dtype=torch.float64),
            variance=torch.tensor([[0.04, 0.04]], dtype=torch.float64),
        )
    )
    with pytest.raises(FloatingPointError, match="H=0.20.*terminal"):
        _require_finite_path_sample(
            sample, stage="calibration", hurst=0.20, task="terminal", offset=17
        )


def test_preserved_p7_development_artifact_is_hash_consistent_and_nonconfirmatory() -> None:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    assert payload["passed"]
    assert len(payload["cells"]) == 24
    assert all(payload["gates"].values())
    assert payload["dirty_worktree"] is True
    assert payload["thresholds_hash_bound"] is False
    assert payload["reference_execution_complete"] is False
    assert payload["performance_claim_authorized"] is False
    assert payload["config_sha256"] == hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    manifest = payload["candidate_manifest"]
    canonical = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    assert payload["candidate_manifest_sha256"] == hashlib.sha256(canonical).hexdigest()
