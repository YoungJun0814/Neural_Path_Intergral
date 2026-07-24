"""V7 confirmation-freeze transition tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.g11_v6_secondary_baselines import (
    _load_config as load_fixed_config,
)
from experiments.g11_v7_accuracy_analysis import (
    _load_config as load_accuracy_config,
)
from experiments.g11_v7_freeze_confirmation import (
    validate_config_transition,
)
from experiments.g11_v7_mechanism_analysis import (
    _load_config as load_analysis_config,
)
from experiments.g11_v7_mechanism_probe import (
    _load_config as load_probe_config,
)

ROOT = Path(__file__).parents[1]


def _configs() -> dict[str, dict]:
    directory = ROOT / "configs" / "g11_v7"
    loaders = {
        "probe": (load_probe_config, "mechanism_probe"),
        "fixed": (load_fixed_config, "fixed_estimators"),
        "analysis": (load_analysis_config, "mechanism_analysis"),
        "accuracy": (load_accuracy_config, "accuracy"),
    }
    result = {}
    for name, (loader, stem) in loaders.items():
        result[f"qualification_{name}"] = loader(directory / f"{stem}_qualification_v1.yaml")[0]
        result[f"confirmation_{name}"] = loader(directory / f"{stem}_confirmation_v1.yaml")[0]
    return result


def test_v7_confirmation_changes_only_predeclared_fields() -> None:
    validate_config_transition(**_configs())


def test_v7_confirmation_rejects_threshold_drift() -> None:
    configs = _configs()
    configs["confirmation_analysis"]["development_thresholds"]["minimum_final_work_ratio_lower"] = (
        1.4
    )
    with pytest.raises(ValueError, match="joint-analysis design drifted"):
        validate_config_transition(**configs)
