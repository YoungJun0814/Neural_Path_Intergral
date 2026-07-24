"""Strict config and claim-family tests for V6 multiplicity sensitivity."""

from __future__ import annotations

from pathlib import Path

from experiments.g11_v6_accuracy_multiplicity_sensitivity import _load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs" / "g11_v6" / "accuracy_multiplicity_sensitivity_v1.yaml"
)


def test_accuracy_multiplicity_config_freezes_post_hoc_disclosure() -> None:
    config, digest = _load_config(CONFIG)
    requirements = config["requirements"]
    claims = (
        requirements["expected_cells"]
        * requirements["expected_methods"]
        * requirements["expected_accuracy_gates_per_group"]
    )
    assert config["post_hoc"]
    assert config["frozen"]
    assert claims == 72
    assert requirements["expected_claims"] == claims
    assert config["statistics"]["bootstrap_repetitions"] == 20000
    assert len(digest) == 64
