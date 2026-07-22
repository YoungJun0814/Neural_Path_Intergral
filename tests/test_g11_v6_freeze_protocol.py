"""Outcome-blind V6 protocol-freeze tests."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from experiments.g11_v6_confirmatory import _load_config as load_confirmatory
from experiments.g11_v6_freeze_protocol import build_frozen_configs

ROOT = Path(__file__).resolve().parents[1]


def _yaml(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / "g11_v6" / name).read_bytes())


def test_freeze_builder_changes_phase_and_seed_namespace_without_mutating_templates(
    tmp_path: Path,
) -> None:
    baseline = _yaml("baseline_qualification_development.yaml")
    policy = _yaml("routed_policy_development.yaml")
    audit = _yaml("result_audit_development.yaml")
    confirmatory = _yaml("confirmatory_development.yaml")
    originals = copy.deepcopy((baseline, policy, audit, confirmatory))
    payloads, hashes = build_frozen_configs(
        baseline,
        policy,
        audit,
        confirmatory,
        planned_clusters=24,
        manifest_sha256="1" * 64,
        reference_sha256="2" * 64,
        power_sha256="3" * 64,
    )
    assert (baseline, policy, audit, confirmatory) == originals
    assert payloads["baseline_confirmation.yaml"]["phase"] == "confirmation"
    assert payloads["routed_policy_confirmation.yaml"]["phase"] == "confirmation"
    assert payloads["baseline_confirmation.yaml"]["sampling"]["clusters"] == 24
    assert payloads["routed_policy_confirmation.yaml"]["sampling"]["clusters"] == 24
    assert payloads["baseline_confirmation.yaml"]["protocol_id"].endswith(
        "confirmation-v1"
    )
    assert payloads["routed_policy_confirmation.yaml"]["protocol_id"].endswith(
        "confirmation-v1"
    )
    assert payloads["confirmatory.yaml"]["expected_sha256"] == hashes
    path = tmp_path / "confirmatory.yaml"
    path.write_text(
        yaml.safe_dump(payloads["confirmatory.yaml"], sort_keys=False), encoding="utf-8"
    )
    parsed, _digest = load_confirmatory(path)
    assert parsed["frozen"]
    assert parsed["phase"] == "confirmation"
