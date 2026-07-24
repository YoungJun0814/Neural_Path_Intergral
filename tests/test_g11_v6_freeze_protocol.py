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
        manifest_cell_count=2,
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


def test_freeze_builder_updates_v4_training_matrix_without_changing_totals() -> None:
    baseline = _yaml("baseline_primary_development_v7.yaml")
    policy = _yaml("routed_policy_development_v9.yaml")
    audit = _yaml("result_audit_development.yaml")
    confirmatory = _yaml("confirmatory_development.yaml")
    original_training_totals = {
        key: policy["proposal"][key]
        for key in (
            "training_total_samples",
            "training_total_work_units",
            "training_total_wall_seconds",
            "training_total_cpu_seconds",
        )
    }
    payloads, _hashes = build_frozen_configs(
        baseline,
        policy,
        audit,
        confirmatory,
        planned_clusters=64,
        manifest_cell_count=18,
        manifest_sha256="1" * 64,
        reference_sha256="2" * 64,
        power_sha256="3" * 64,
    )
    frozen = payloads["routed_policy_confirmation.yaml"]
    assert frozen["schema"] == "npi.g11.v6-routed-policy.config.v4"
    assert frozen["proposal"]["training_amortization_record_count"] == 18 * 64
    assert {
        key: frozen["proposal"][key] for key in original_training_totals
    } == original_training_totals
    assert payloads["baseline_confirmation.yaml"]["schema"].endswith(".config.v6")


def test_freeze_builder_versions_every_seed_namespace() -> None:
    payloads, _hashes = build_frozen_configs(
        _yaml("baseline_primary_development_v7.yaml"),
        _yaml("routed_policy_development_v9.yaml"),
        _yaml("result_audit_development.yaml"),
        _yaml("confirmatory_development.yaml"),
        planned_clusters=64,
        manifest_cell_count=18,
        manifest_sha256="1" * 64,
        reference_sha256="2" * 64,
        power_sha256="3" * 64,
        protocol_version=2,
    )
    assert payloads["baseline_confirmation.yaml"]["protocol_id"].endswith("-v2")
    assert payloads["routed_policy_confirmation.yaml"]["protocol_id"].endswith(
        "-v2"
    )
    assert payloads["result_audit_confirmation.yaml"]["protocol_id"].endswith(
        "-v2"
    )
    assert payloads["confirmatory.yaml"]["protocol_id"].endswith("-v2")
