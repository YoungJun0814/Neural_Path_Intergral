from __future__ import annotations

from pathlib import Path

import pytest

from experiments.g11_v8_p5_reference_resource_cap_escalation import (
    _next_power_of_two,
    audit_manifest,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/g11_v8/p5_reference_resource_cap_escalation_v1.yaml"
MANIFEST = ROOT / "results/g11_v8_p5_reference_proposal_manifest_v3_2026-07-31.json"


def test_resource_cap_rule_is_objectively_derived() -> None:
    config, digest = load_config(CONFIG)
    rule = config["resource_cap_rule"]
    assert _next_power_of_two(
        rule["multiplier"] * rule["prior_requested_final_samples"]
    ) == 134_217_728
    assert rule["unchanged_raw_maximum_final_samples"] == 16_777_216
    assert len(digest) == 64


def test_resource_cap_manifest_audits_if_present() -> None:
    if not MANIFEST.exists():
        pytest.skip("frozen resource-cap manifest has not been built yet")
    report = audit_manifest(CONFIG, MANIFEST)
    assert report["passed"] is True
    assert report["checks"]["proposals_bitwise_unchanged"] is True
