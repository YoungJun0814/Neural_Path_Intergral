"""Outcome-blind hardware-reproduction freeze tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from experiments.g11_v6_freeze_hardware_reproduction import (
    build_frozen_hardware_config,
)

ROOT = Path(__file__).resolve().parents[1]


def _template() -> dict:
    return yaml.safe_load(
        (
            ROOT
            / "configs"
            / "g11_v6"
            / "hardware_reproduction_development.yaml"
        ).read_bytes()
    )


def test_hardware_freeze_authenticates_canonical_and_reproduction_protocol() -> None:
    frozen = build_frozen_hardware_config(
        _template(),
        canonical_confirmation_sha256="1" * 64,
        reproduction_hashes={
            "baseline_config": "2" * 64,
            "policy_config": "3" * 64,
            "manifest": "4" * 64,
            "reference": "5" * 64,
            "power": "6" * 64,
            "audit_config": "7" * 64,
        },
    )
    assert frozen["frozen"]
    assert frozen["protocol_id"].endswith("-v1")
    assert frozen["expected_sha256"]["canonical_confirmation"] == "1" * 64
    assert frozen["expected_sha256"]["reproduction_policy_config"] == "3" * 64


def test_hardware_freeze_rejects_incomplete_or_malformed_hashes() -> None:
    with pytest.raises(ValueError, match="incomplete"):
        build_frozen_hardware_config(
            _template(),
            canonical_confirmation_sha256="1" * 64,
            reproduction_hashes={},
        )
    with pytest.raises(ValueError, match="malformed"):
        build_frozen_hardware_config(
            _template(),
            canonical_confirmation_sha256="x" * 64,
            reproduction_hashes={
                "baseline_config": "2" * 64,
                "policy_config": "3" * 64,
                "manifest": "4" * 64,
                "reference": "5" * 64,
                "power": "6" * 64,
                "audit_config": "7" * 64,
            },
        )
