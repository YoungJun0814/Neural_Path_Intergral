"""Protocol, selection, censoring, and aggregation tests for G11 M7."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.g11_m7_confirmatory import (
    _load_config,
    _load_regimes,
    cap_raw_allocation,
    expected_cell_count,
    preflight,
    summarize,
)
from src.path_integral.mlmc import (
    FixedFinestGridTarget,
    LevelAllocation,
    MLMCHierarchy,
    MLMCPreparedRun,
    WorkLedger,
)
from src.path_integral.seed_ledger import SeedLedger

ROOT = Path(__file__).resolve().parents[1]


def _config(name: str):
    config, _ = _load_config(ROOT / "configs" / name)
    return config


def test_frozen_protocol_selects_640_declared_cells_and_excludes_low_h_excursion() -> None:
    config = _config("g11_m7_confirmatory_v1.yaml")
    contexts, inputs = _load_regimes(config)
    task_counts = {context["name"]: len(context["tasks"]) for context in contexts}
    assert task_counts == {"low_h007": 8, "primary_h012": 12, "high_h030": 12}
    low_tasks = next(context["tasks"] for context in contexts if context["name"] == "low_h007")
    assert {task["name"] for task in low_tasks} == {"terminal", "barrier"}
    assert expected_cell_count(config) == 640
    assert len(inputs) == 6


def test_qualification_protocol_is_disjoint_and_contains_one_cell() -> None:
    qualification = _config("g11_m7_local_qualification.yaml")
    confirmatory = _config("g11_m7_confirmatory_v1.yaml")
    assert qualification["protocol_id"] != confirmatory["protocol_id"]
    assert not qualification["frozen"] and confirmatory["frozen"]
    assert expected_cell_count(qualification) == 1


def test_qualification_preflight_allocates_no_random_seed() -> None:
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(8)
        result = preflight(ROOT / "configs" / "g11_m7_local_qualification.yaml")
    finally:
        torch.set_num_threads(previous)
    assert result["expected_cell_count"] == 1
    assert result["random_seeds_allocated"] == 0
    assert len(result["seed_namespace_sha256"]) == 64


def test_raw_cap_preserves_uncapped_counts_and_does_not_modify_core_types() -> None:
    hierarchy = MLMCHierarchy(8, 2, FixedFinestGridTarget(1))
    prepared = MLMCPreparedRun(
        hierarchy=hierarchy,
        protocol="cap-test",
        regime="oracle",
        task="linear",
        streams=("proposal", "labels"),
        pilot=(),
        allocations=(
            LevelAllocation(0, 5000.0, 5000, 0.5, 1.0),
            LevelAllocation(1, 200.0, 200, 0.1, 2.0),
        ),
        sampling_variance_target=1e-3,
        chunk_size=64,
        ledger=SeedLedger(),
        work=WorkLedger.empty(),
    )
    capped, uncapped, was_capped = cap_raw_allocation(prepared, 1024)
    assert uncapped == (5000, 200)
    assert tuple(item.final_count for item in capped.allocations) == (1024, 200)
    assert was_capped
    assert tuple(item.final_count for item in prepared.allocations) == (5000, 200)


def test_summary_never_counts_censored_baseline_as_matched_speedup() -> None:
    config = _config("g11_m7_local_qualification.yaml")
    cell = {
        "regime": "primary_h012",
        "task": "terminal_1e3",
        "target_probability": 1e-3,
        "rmse_target": 2e-4,
        "replicate": 0,
        "allocated_work_ratio_raw_over_dcs": 20.0,
        "matched_work_ratio_raw_over_dcs": None,
        "censored_work_ratio_lower_bound": 20.0,
        "methods": {
            "raw_defensive": {
                "target_attained": False,
                "resource_censored": True,
            },
            "dcs_mgi": {
                "target_attained": True,
                "resource_censored": False,
            },
        },
    }
    summary, gates = summarize(config, [cell], [])
    assert summary["matched_cell_count"] == 0
    assert summary["resource_censored_cell_count"] == 1
    assert summary["matched_geometric_work_ratio"] is None
    assert summary["censored_geometric_work_ratio_lower_bound"] == pytest.approx(20.0)
    assert gates["protocol_complete"]
    assert not gates["matched_geometric_work_ratio_above_threshold"]
