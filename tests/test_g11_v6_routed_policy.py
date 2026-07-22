"""End-to-end V6 routed-policy smoke test."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.g11_v6_rarity_calibration import run as run_calibration
from experiments.g11_v6_reference import run as run_reference
from experiments.g11_v6_result_audit import run as run_audit
from experiments.g11_v6_routed_policy import _load_config
from experiments.g11_v6_routed_policy import run as run_policy

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / "configs" / "g11_v6" / "rarity_calibration_development.yaml"
REFERENCE = ROOT / "configs" / "g11_v6" / "reference_development.yaml"
POLICY = ROOT / "configs" / "g11_v6" / "routed_policy_development.yaml"
AUDIT = ROOT / "configs" / "g11_v6" / "result_audit_development.yaml"


def test_v6_routed_policy_config_is_strict() -> None:
    config, digest = _load_config(POLICY)
    assert config["router"]["probability_cutoff"] == 0.05
    assert len(digest) == 64


@pytest.mark.slow
def test_v6_routed_policy_smoke_is_auditable_and_unqualified(tmp_path: Path) -> None:
    calibration = run_calibration(CALIBRATION, smoke=True)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(calibration["candidate_manifest"]), encoding="utf-8")
    reference = run_reference(REFERENCE, manifest_path, smoke=True)
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(reference), encoding="utf-8")
    result = run_policy(POLICY, manifest_path, reference_path, smoke=True)
    assert result["schema"] == "npi.g11.v6-routed-policy.v1"
    assert len(result["records"]) == 2
    assert result["gates"]["complete_matrix"]
    assert result["gates"]["all_routes_resolved"]
    assert result["gates"]["all_independent_audits"]
    assert not result["policy_qualified"]
    artifact_path = tmp_path / "routed.json"
    artifact_path.write_text(json.dumps(result), encoding="utf-8")
    offline = run_audit(AUDIT, artifact_path)
    assert offline["gates"]["all_records_pass"]
    assert not offline["qualification_audit_passed"]
