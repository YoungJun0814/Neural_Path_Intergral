"""Independent post-run audit tests for the frozen G11 M7 protocol."""

from __future__ import annotations

import hashlib
from pathlib import Path

from experiments.g11_m7_result_audit import (
    _cell_failures,
    _independent_complete_seed_hash,
    _load_config,
    _variance_exponent,
    expected_cell_keys,
    frozen_source_failures,
    independent_summary,
)

ROOT = Path(__file__).resolve().parents[1]


def _method(*, target: bool, censored: bool, work: float) -> dict[str, object]:
    variance = 0.01 if target else 0.09
    standard_error = variance**0.5
    return {
        "estimate": 0.5,
        "empirical_sampling_variance": variance,
        "standard_error": standard_error,
        "confidence_interval_95": [
            0.5 - 1.959963984540054 * standard_error,
            0.5 + 1.959963984540054 * standard_error,
        ],
        "target_attained": target,
        "allocation_capped": censored,
        "pilot_resource_censored": False,
        "resource_censored": censored,
        "censor_stage": "final_allocation" if censored else None,
        "total_work_units": work,
        "total_wall_seconds": work / 10.0,
        "process_cpu_seconds": work / 5.0,
        "allocations": [{"final_count": 10}],
        "seed_evidence_kind": "complete_mlmc_seed_ledger",
        "seed_evidence_sha256": "a" * 64,
    }


def _cell() -> dict[str, object]:
    raw = _method(target=True, censored=False, work=20.0)
    dcs = _method(target=True, censored=False, work=10.0)
    return {
        "regime": "oracle",
        "task": "terminal_1e3",
        "target_probability": 1e-3,
        "rmse_target": 0.5,
        "replicate": 0,
        "methods": {"raw_defensive": raw, "dcs_mgi": dcs},
        "allocated_work_ratio_raw_over_dcs": 2.0,
        "matched_work_ratio_raw_over_dcs": 2.0,
        "censored_work_ratio_lower_bound": None,
        "wall_ratio_raw_over_dcs": 2.0,
        "paired_estimate_difference": 0.0,
    }


def test_expected_frozen_matrix_contains_640_unique_cells() -> None:
    path = ROOT / "configs" / "g11_m7_confirmatory_v3.yaml"
    config, config_hash = _load_config(path)
    assert config_hash == hashlib.sha256(path.read_bytes()).hexdigest()
    keys = expected_cell_keys(config)
    assert len(keys) == 640
    assert ("primary_h012", "excursion_1e6", 19) in keys
    assert ("low_h007", "excursion_1e6", 0) not in keys


def test_frozen_core_source_manifest_matches_declared_commit() -> None:
    config, _config_hash = _load_config(ROOT / "configs" / "g11_m7_confirmatory_v3.yaml")
    assert frozen_source_failures(config, ROOT) == []


def test_cell_audit_reconstructs_derived_ratios() -> None:
    cell = _cell()
    assert not _cell_failures(cell, raw_cap=100)
    cell["matched_work_ratio_raw_over_dcs"] = 99.0
    assert any("matched_work_ratio" in item for item in _cell_failures(cell, 100))


def test_independent_summary_never_turns_censoring_into_matched_speedup() -> None:
    config = {
        "sampling": {"repetitions": 1},
        "regimes": [
            {
                "name": "oracle",
                "included_tasks": ["terminal"],
                "target_probabilities": [1e-3],
            }
        ],
        "gates": {
            "minimum_dcs_target_attainment_fraction": 0.9,
            "minimum_matched_target_cells": 1,
            "minimum_geometric_work_ratio": 1.25,
            "minimum_seed_clusters_for_uncertainty": 1,
        },
    }
    cell = _cell()
    raw = cell["methods"]["raw_defensive"]
    raw.update(
        {
            "empirical_sampling_variance": 0.09,
            "standard_error": 0.3,
            "target_attained": False,
            "allocation_capped": True,
            "resource_censored": True,
            "censor_stage": "final_allocation",
        }
    )
    cell["matched_work_ratio_raw_over_dcs"] = None
    cell["censored_work_ratio_lower_bound"] = 2.0
    summary, gates = independent_summary(config, [cell], [])
    assert summary["matched_cell_count"] == 0
    assert summary["resource_censored_cell_count"] == 1
    assert summary["matched_geometric_work_ratio"] is None
    assert gates["matched_geometric_work_ratio_above_threshold"] is False


def test_complete_seed_hash_is_reconstructed_from_batch_metadata() -> None:
    config = {
        "protocol_id": "audit-test",
        "sampling": {"pilot_samples": 4, "chunk_size": 4},
    }
    cell = {"regime": "base", "task": "terminal_1e3", "replicate": 2}
    method = {
        "pilot": [{"level": 0, "count": 8}],
        "allocations": [{"level": 0, "final_count": 5}],
    }
    digest = _independent_complete_seed_hash(config, cell, method)
    assert len(digest) == 64
    method["allocations"][0]["final_count"] = 9
    assert _independent_complete_seed_hash(config, cell, method) != digest


def test_variance_exponent_uses_only_correction_levels() -> None:
    method = {
        "levels": [
            {"level": 0, "variance": 1000.0},
            {"level": 1, "variance": 0.25},
            {"level": 2, "variance": 0.0625},
            {"level": 3, "variance": 0.015625},
        ]
    }
    assert _variance_exponent(method) == 2.0
