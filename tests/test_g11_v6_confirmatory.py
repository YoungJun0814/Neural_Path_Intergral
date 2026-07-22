"""V6 fail-closed confirmatory-analysis tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.g11_v6_confirmatory import _load_config, run

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "g11_v6" / "confirmatory_development.yaml"


def _record(cell: str, cluster: int, work: float, *, method: str | None) -> dict:
    estimate = 0.01 + (cluster % 2) * 1e-5
    value = {
        "cell_id": cell,
        "cluster": cluster,
        "nominal_probability": 0.01,
        "reference_probability": 0.01,
        "reference_standard_error": 1e-5,
        "result": {
            "core": {
                "complete": True,
                "resource_censored": False,
                "estimate": estimate,
                "requested_relative_sampling_rmse": 0.20,
                "empirical_target_attained": True,
            },
            "total_work": {"records": [{"work_units": work}]},
        },
    }
    if method is not None:
        value["method"] = method
    return value


def _write(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v6_confirmatory_config_is_strict_and_unfrozen() -> None:
    config, digest = _load_config(CONFIG)
    assert config["phase"] == "development"
    assert not config["frozen"]
    assert len(digest) == 64


def test_confirmatory_analysis_uses_equal_cell_pairs_and_accuracy_co_gates(
    tmp_path: Path,
) -> None:
    baseline_records = []
    policy_records = []
    for cluster in range(8):
        for cell_index, cell in enumerate(("cell-a", "cell-b")):
            policy_work = 100.0 + cluster + cell_index
            ratio = 1.8 + 0.04 * cluster
            baseline_records.append(
                _record(cell, cluster, ratio * policy_work, method="pure_cem")
            )
            policy_records.append(_record(cell, cluster, policy_work, method=None))
    baseline = {
        "schema": "npi.g11.v6-baseline-qualification.v1",
        "config_sha256": "1" * 64,
        "manifest_sha256": "2" * 64,
        "reference_artifact_sha256": "3" * 64,
        "smoke": False,
        "formal_readiness": {"frozen_config": False},
        "records": baseline_records,
    }
    policy = {
        "schema": "npi.g11.v6-routed-policy.v1",
        "config_sha256": "4" * 64,
        "manifest_sha256": "2" * 64,
        "reference_artifact_sha256": "3" * 64,
        "smoke": False,
        "formal_readiness": {"frozen_config": False},
        "records": policy_records,
    }
    baseline_path = tmp_path / "baseline.json"
    policy_path = tmp_path / "policy.json"
    baseline_hash = _write(baseline_path, baseline)
    policy_hash = _write(policy_path, policy)
    baseline_audit = {
        "schema": "npi.g11.v6-independent-audit.v1",
        "source_artifact_sha256": baseline_hash,
        "config_sha256": "5" * 64,
        "gates": {"all_records_pass": True},
        "qualification_audit_passed": False,
    }
    policy_audit = {
        "schema": "npi.g11.v6-independent-audit.v1",
        "source_artifact_sha256": policy_hash,
        "config_sha256": "5" * 64,
        "gates": {"all_records_pass": True},
        "qualification_audit_passed": False,
    }
    baseline_audit_path = tmp_path / "baseline-audit.json"
    policy_audit_path = tmp_path / "policy-audit.json"
    _write(baseline_audit_path, baseline_audit)
    _write(policy_audit_path, policy_audit)
    power = {
        "schema": "npi.g11.v6-power-analysis.v1",
        "forecast": {"required_clusters_normal_approximation": 6},
    }
    power_path = tmp_path / "power.json"
    _write(power_path, power)
    result = run(
        CONFIG,
        baseline_path,
        policy_path,
        baseline_audit_path,
        policy_audit_path,
        power_path,
    )
    assert result["cell_count"] == 2
    assert result["cluster_count"] == 8
    assert result["gates"]["all_accuracy_co_gates"]
    assert result["gates"]["one_sided_efficiency_lower_exceeds_one"]
    assert result["gates"]["shared_protocol_identities"]
    assert result["scientific_gates_passed"]
    assert not result["confirmation_passed"]
