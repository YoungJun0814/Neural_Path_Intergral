"""Independent V5 audit pass and tamper-detection tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from experiments.g11_v5_result_audit import audit


def _artifact(tmp_path: Path) -> tuple[Path, Path]:
    config = {
        "schema": "npi.g11.v5-confirmatory.config.v1",
        "protocol_id": "audit-test",
        "estimand": "finite_grid",
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    gates = {
        "all_runs_complete_or_resource_censored": True,
        "all_design_targets_attained_if_feasible": True,
        "selection_frozen_before_final": True,
        "no_final_samples_reused_from_selection": True,
        "all_preparation_hashes_unique": True,
    }
    record = {
        "cell_id": "cell-0",
        "task": {"nominal_probability": 0.2},
        "selection": {
            "stopped": True,
            "frozen_decision": {
                "selected_candidate": "start_0",
                "surviving_candidates": ["start_0"],
                "selected_point_work": 100.0,
            },
            "candidate_work": [{"candidate_id": "start_0", "point_total_work": 100.0}],
        },
        "preparation": {
            "selected_candidate": "start_0",
            "preparation_hash": "a" * 64,
            "resource_censored": False,
            "allocations": [
                {
                    "profile_id": "single_0",
                    "design_variance": 0.04,
                    "final_count": 100,
                    "cost_per_sample": 1.0,
                }
            ],
        },
        "result": {
            "selected_candidate": "start_0",
            "preparation_hash": "a" * 64,
            "complete": True,
            "resource_censored": False,
            "design_sampling_variance": 0.0004,
            "design_target_attained": True,
            "requested_relative_sampling_rmse": 0.1,
            "estimate": 0.1,
            "empirical_sampling_variance": 0.0004,
            "empirical_target_attained": True,
            "terms": [
                {
                    "profile_id": "single_0",
                    "count": 100,
                    "mean": 0.1,
                    "variance": 0.04,
                }
            ],
            "work": {
                "entries": [
                    {
                        "role": "final",
                        "level": 0,
                        "samples": 100,
                        "work_units": 100.0,
                        "wall_seconds": 0.0,
                    }
                ]
            },
        },
    }
    result = {
        "schema": "npi.g11.v5-confirmatory.v1",
        "protocol_id": "audit-test",
        "config_sha256": config_hash,
        "records": [record],
        "gates": gates,
        "protocol_complete": True,
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    return result_path, config_path


def test_independent_audit_accepts_consistent_artifact(tmp_path: Path) -> None:
    result_path, config_path = _artifact(tmp_path)
    report = audit(result_path, config_path)
    assert report["passed"]
    assert report["failures"] == []


def test_independent_audit_detects_tampered_estimate(tmp_path: Path) -> None:
    result_path, config_path = _artifact(tmp_path)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["records"][0]["result"]["estimate"] = 0.2
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    report = audit(result_path, config_path)
    assert not report["passed"]
    assert any("does not telescope" in item for item in report["failures"])
