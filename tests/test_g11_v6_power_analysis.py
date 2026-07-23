"""Equal-cell V6 power-planning tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.g11_v6_power_analysis import _load_config, run

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "g11_v6" / "power_analysis_development.yaml"
CONFIG_V2 = ROOT / "configs" / "g11_v6" / "power_analysis_development_v2.yaml"


def _work_result(work: float) -> dict:
    return {
        "core": {
            "complete": True,
            "resource_censored": False,
            "design_target_attained": True,
            "empirical_target_attained": True,
        },
        "total_work": {
            "records": [
                {
                    "work_units": work,
                    "wall_seconds": work / 100.0,
                    "cpu_seconds": work / 50.0,
                    "peak_memory_bytes": 1024,
                }
            ]
        },
    }


def test_v6_power_analysis_uses_equal_cell_cluster_pairs(tmp_path: Path) -> None:
    baseline_records = []
    policy_records = []
    for cluster in range(4):
        for cell_index, cell in enumerate(("cell-a", "cell-b")):
            policy_work = 100.0 + cluster + 3.0 * cell_index
            baseline_records.append(
                {
                    "cell_id": cell,
                    "cluster": cluster,
                    "method": "pure_cem",
                    "result": _work_result(2.0 * policy_work),
                }
            )
            policy_records.append(
                {
                    "cell_id": cell,
                    "cluster": cluster,
                    "result": _work_result(policy_work),
                }
            )
    baseline = {
        "schema": "npi.g11.v6-baseline-qualification.v1",
        "baseline_qualified": False,
        "records": baseline_records,
    }
    policy = {
        "schema": "npi.g11.v6-routed-policy.v1",
        "policy_qualified": False,
        "records": policy_records,
    }
    baseline_path = tmp_path / "baseline.json"
    policy_path = tmp_path / "policy.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    result = run(CONFIG, baseline_path, policy_path)
    assert result["cell_count"] == 2
    assert result["qualification_cluster_count"] == 4
    assert result["observed_geometric_mean_ratio"] == 2.0
    assert result["gates"]["no_pairs_excluded"]
    assert result["gates"]["observed_direction_favors_policy"]
    assert result["gates"]["resource_measurements_complete"]
    assert result["projected_wall_hours"] == max(
        result["throughput_projected_wall_hours"],
        result["measured_projected_wall_hours"],
    )
    assert not result["freeze_power_ready"]


def test_v6_power_config_is_strict() -> None:
    config, digest = _load_config(CONFIG)
    assert config["statistics"]["practical_geometric_mean_ratio"] == 1.20
    assert len(digest) == 64
    aggregate, aggregate_digest = _load_config(CONFIG_V2)
    assert aggregate["schema"] == "npi.g11.v6-power-analysis.config.v2"
    assert aggregate["design"]["planned_clusters"] == 24
    assert len(aggregate_digest) == 64


def test_v6_power_rejects_empirical_rmse_failure(tmp_path: Path) -> None:
    baseline_result = _work_result(200.0)
    baseline_result["core"]["empirical_target_attained"] = False
    baseline = {
        "schema": "npi.g11.v6-baseline-qualification.v1",
        "baseline_qualified": False,
        "records": [
            {"cell_id": "cell", "cluster": cluster, "method": "pure_cem", "result": baseline_result}
            for cluster in range(3)
        ],
    }
    policy = {
        "schema": "npi.g11.v6-routed-policy.v1",
        "policy_qualified": False,
        "records": [
            {"cell_id": "cell", "cluster": cluster, "result": _work_result(100.0)}
            for cluster in range(3)
        ],
    }
    baseline_path = tmp_path / "baseline.json"
    policy_path = tmp_path / "policy.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(ValueError, match="sampling-RMSE target"):
        run(CONFIG, baseline_path, policy_path)


def test_v2_power_uses_prespecified_aggregate_accuracy_co_gates(tmp_path: Path) -> None:
    baseline_records = []
    policy_records = []
    for cluster in range(24):
        baseline_result = _work_result(200.0)
        baseline_result["core"].update(
            {
                "requested_relative_sampling_rmse": 0.20,
                "estimate": 0.001 + (1e-5 if cluster % 2 else -1e-5),
                "empirical_target_attained": cluster >= 2,
            }
        )
        policy_result = _work_result(100.0)
        policy_result["core"].update(
            {
                "requested_relative_sampling_rmse": 0.20,
                "estimate": 0.001 + (8e-6 if cluster % 2 else -8e-6),
            }
        )
        common = {
            "cell_id": "cell",
            "cluster": cluster,
            "nominal_probability": 0.001,
            "reference_probability": 0.001,
            "reference_standard_error": 1e-6,
        }
        baseline_records.append({**common, "method": "pure_cem", "result": baseline_result})
        policy_records.append({**common, "result": policy_result})
    baseline_path = tmp_path / "baseline-v2.json"
    policy_path = tmp_path / "policy-v2.json"
    baseline_path.write_text(
        json.dumps(
            {
                "schema": "npi.g11.v6-baseline-qualification.v1",
                "baseline_qualified": False,
                "records": baseline_records,
            }
        ),
        encoding="utf-8",
    )
    policy_path.write_text(
        json.dumps(
            {
                "schema": "npi.g11.v6-routed-policy.v1",
                "policy_qualified": False,
                "records": policy_records,
            }
        ),
        encoding="utf-8",
    )
    result = run(CONFIG_V2, baseline_path, policy_path)
    assert result["gates"]["all_accuracy_co_gates"]
    baseline_accuracy = next(
        item for item in result["accuracy"] if item["method"] == "pure_cem"
    )
    assert baseline_accuracy["target_attainment_count"] == 22
    assert baseline_accuracy["attainment_gate"]
