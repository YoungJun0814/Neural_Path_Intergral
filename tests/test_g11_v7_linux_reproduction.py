"""V7 Linux reproduction freeze and effect tests."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from experiments.g11_v6_secondary_baselines import (
    _load_config as load_fixed_config,
)
from experiments.g11_v7_accuracy_analysis import (
    _load_config as load_accuracy_config,
)
from experiments.g11_v7_freeze_linux_reproduction import (
    _normalized,
    build_reproduction_configs,
)
from experiments.g11_v7_hardware_reproduction import _effect_z
from experiments.g11_v7_mechanism_analysis import (
    _load_config as load_analysis_config,
)
from experiments.g11_v7_mechanism_probe import (
    _load_config as load_probe_config,
)

ROOT = Path(__file__).parents[1]


def _canonical_configs() -> dict:
    directory = ROOT / "configs" / "g11_v7"
    return {
        "probe": load_probe_config(directory / "mechanism_probe_confirmation_v1.yaml")[0],
        "fixed": load_fixed_config(directory / "fixed_estimators_confirmation_v1.yaml")[0],
        "analysis": load_analysis_config(directory / "mechanism_analysis_confirmation_v1.yaml")[0],
        "accuracy": load_accuracy_config(directory / "accuracy_confirmation_v1.yaml")[0],
    }


def test_linux_reproduction_changes_only_seed_namespaces() -> None:
    source = _canonical_configs()
    reproduction = build_reproduction_configs(**source)
    mapping = {
        "mechanism_probe": "probe",
        "fixed_estimators": "fixed",
        "joint_analysis": "analysis",
        "simultaneous_accuracy": "accuracy",
    }
    for kind, source_name in mapping.items():
        assert _normalized(reproduction[kind], kind) == _normalized(
            source[source_name],
            kind,
        )
        assert "linux-reproduction" in reproduction[kind]["protocol_id"]


def test_linux_reproduction_normalizer_detects_scientific_drift() -> None:
    source = _canonical_configs()
    changed = copy.deepcopy(source["analysis"])
    changed["development_thresholds"]["minimum_final_work_ratio_lower"] = 1.4
    assert _normalized(changed, "joint_analysis") != _normalized(
        source["analysis"],
        "joint_analysis",
    )


def test_hardware_effect_z_uses_combined_standard_error() -> None:
    canonical = {"mean_log_ratio": 1.0, "standard_error": 0.3}
    reproduction = {"mean_log_ratio": 1.5, "standard_error": 0.4}
    assert _effect_z(canonical, reproduction) == pytest.approx(1.0)
