"""Contracts for the paired V7 raw/DCS mechanism probe."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from experiments.g11_v6_secondary_baselines import (
    _load_config as load_fixed_config,
)
from experiments.g11_v7_mechanism_analysis import (
    _load_config as load_analysis_config,
)
from experiments.g11_v7_mechanism_probe import _load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "g11_v7" / "mechanism_probe_development_v1.yaml"
CONFIG_V2 = ROOT / "configs" / "g11_v7" / "mechanism_probe_development_v2.yaml"
FIXED_CONFIG = (
    ROOT / "configs" / "g11_v7" / "fixed_estimators_development_v1.yaml"
)
ANALYSIS_CONFIG = (
    ROOT / "configs" / "g11_v7" / "mechanism_analysis_development_v2.yaml"
)


def test_v7_mechanism_probe_config_is_strict_and_development_only() -> None:
    config, digest = _load_config(CONFIG_V2)
    assert config["phase"] == "development"
    assert not config["frozen"]
    assert config["sampling"]["clusters"] == 8
    assert config["sampling"]["samples_per_cell_cluster"] == 4096
    assert config["development_thresholds"][
        "maximum_absolute_orthogonality_z"
    ] == 4.5
    assert config["requirements"]["expected_cells"] == 18
    assert len(digest) == 64


def test_v7_mechanism_probe_rejects_unknown_fields(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["unfrozen_escape_hatch"] = True
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        _load_config(path)


def test_v7_fixed_estimators_remove_the_floor_mask() -> None:
    config, digest = load_fixed_config(FIXED_CONFIG)
    assert config["protocol_id"].startswith("g11-v7-")
    assert config["methods"] == ["fixed_dcs_slis", "fixed_raw_defensive"]
    assert config["sampling"]["relative_sampling_rmse"] == 0.10
    assert config["sampling"]["minimum_final_samples"] == 512
    assert config["proposal"]["training_amortization_record_count"] == 18 * 8
    assert len(digest) == 64


def test_v7_joint_analysis_predeclares_mechanism_thresholds() -> None:
    config, digest = load_analysis_config(ANALYSIS_CONFIG)
    thresholds = config["development_thresholds"]
    assert thresholds["minimum_probe_variance_ratio_lower"] == 1.5
    assert thresholds["minimum_execution_variance_ratio_lower"] == 1.5
    assert thresholds["minimum_final_work_ratio_lower"] == 1.2
    assert thresholds["maximum_floor_fraction_per_method"] == 0.10
    assert len(digest) == 64
