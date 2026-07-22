"""Equal-cell V6 power-planning tests."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.g11_v6_power_analysis import _load_config, run

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "g11_v6" / "power_analysis_development.yaml"


def _work_result(work: float) -> dict:
    return {
        "core": {"complete": True, "resource_censored": False},
        "total_work": {"records": [{"work_units": work}]},
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
    assert not result["freeze_power_ready"]


def test_v6_power_config_is_strict() -> None:
    config, digest = _load_config(CONFIG)
    assert config["statistics"]["practical_geometric_mean_ratio"] == 1.20
    assert len(digest) == 64
