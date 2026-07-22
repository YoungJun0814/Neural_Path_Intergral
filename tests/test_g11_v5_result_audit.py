"""Independent V5 audit pass and tamper-detection tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from experiments.g11_v5_result_audit import audit
from src.path_integral import SeedKey, SeedLedger


def _artifact(tmp_path: Path) -> tuple[Path, Path]:
    config = {
        "schema": "npi.g11.v5-confirmatory.config.v1",
        "protocol_id": "audit-test",
        "frozen": True,
        "estimand": "finite_grid",
        "run_class": "qualification",
        "source_commit": "a" * 40,
        "models": [{"id": "model"}],
        "tasks": {"terminal": {"kind": "terminal", "level": 90.0}},
        "references": {
            "model": {"terminal": {"probability": 0.2, "standard_error": 0.001}}
        },
        "sampling": {"clusters": 1},
        "qualification_gates": {
            "minimum_empirical_target_attainment": 0.8,
            "maximum_relative_rmse_ratio": 1.1,
            "minimum_combined_asymptotic_coverage": 0.8,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    gates = {
        "complete_cluster_matrix": True,
        "no_resource_censoring": True,
        "all_runs_complete": True,
        "all_design_targets_attained": True,
        "minimum_empirical_target_attainment": True,
        "across_cluster_relative_rmse": True,
        "minimum_combined_asymptotic_coverage": True,
        "selection_frozen_before_final": True,
        "no_final_samples_reused_from_selection": True,
        "all_preparation_hashes_unique": True,
    }
    ledger = SeedLedger()
    for role in ("selection", "final"):
        ledger.allocate(
            SeedKey(
                "audit-test",
                role,
                "model:cluster-0",
                "terminal",
                0,
                0,
                "proposal",
            )
        )
    record = {
        "cell_id": "model:terminal:cluster-0",
        "model_id": "model",
        "task_name": "terminal",
        "reference": {"probability": 0.2, "standard_error": 0.001},
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
        "seed_role_audit": {
            "selection_seed_count": 1,
            "final_seed_count": 1,
            "selection_final_disjoint": True,
            "unexpected_roles": [],
        },
        "result": {
            "selected_candidate": "start_0",
            "preparation_hash": "a" * 64,
            "complete": True,
            "resource_censored": False,
            "design_sampling_variance": 0.0004,
            "design_target_attained": True,
            "requested_relative_sampling_rmse": 0.1,
            "estimate": 0.2,
            "empirical_sampling_variance": 0.0004,
            "empirical_target_attained": True,
            "bounded_confidence_interval": [0.0, 1.0],
            "terms": [
                {
                    "profile_id": "single_0",
                    "count": 100,
                    "mean": 0.2,
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
    aggregates = [
        {
            "model_id": "model",
            "task_name": "terminal",
            "clusters_planned": 1,
            "clusters_complete": 1,
            "resource_censored": 0,
            "reference": {"probability": 0.2, "standard_error": 0.001},
            "requested_relative_sampling_rmse": 0.1,
            "empirical_rmse_against_reference": 0.0,
            "empirical_relative_rmse_against_reference": 0.0,
            "empirical_target_attainment_fraction": 1.0,
            "combined_asymptotic_95_coverage": 1.0,
            "bounded_interval_coverage": 1.0,
            "work_units_median": 100.0,
            "work_units_p90": 100.0,
            "selected_candidate_counts": {"start_0": 1},
        }
    ]
    result = {
        "schema": "npi.g11.v5-confirmatory.v1",
        "protocol_id": "audit-test",
        "config_sha256": config_hash,
        "run_class": "qualification",
        "source_commit": "a" * 40,
        "dirty_worktree": False,
        "records": [record],
        "aggregates": aggregates,
        "gates": gates,
        "formal_readiness": {
            "frozen_config": True,
            "clean_source": True,
            "source_commit_match": True,
            "qualification_inputs_passed": True,
            "non_smoke": True,
        },
        "protocol_complete": True,
        "qualification_passed": True,
        "seed_ledger": ledger.to_dict(),
        "seed_ledger_sha256": ledger.sha256,
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
    payload["records"][0]["result"]["estimate"] = 0.3
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    report = audit(result_path, config_path)
    assert not report["passed"]
    assert any("does not telescope" in item for item in report["failures"])
