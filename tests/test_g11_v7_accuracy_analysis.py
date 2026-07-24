"""V7 simultaneous-accuracy protocol tests."""

from __future__ import annotations

from pathlib import Path

from experiments.g11_v7_accuracy_analysis import _load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "g11_v7" / "accuracy_qualification_v1.yaml"


def test_v7_accuracy_family_is_frozen_before_qualification() -> None:
    config, digest = _load_config(CONFIG)
    assert config["phase"] == "qualification"
    assert config["frozen"]
    assert config["requirements"]["expected_claims"] == 72
    assert config["statistics"]["bootstrap_repetitions"] == 50000
    assert config["matrix"]["expected_clusters"] == 24
    assert len(digest) == 64
