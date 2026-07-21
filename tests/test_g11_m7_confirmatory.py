"""Protocol, selection, censoring, and aggregation tests for G11 M7."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from experiments.g11_m7_confirmatory import (
    _atomic_json,
    _load_config,
    _load_method_checkpoint,
    _load_method_result_cache,
    _load_regimes,
    _method_result,
    _save_method_checkpoint,
    _save_method_result_cache,
    _unlink_with_retry,
    _write_progress,
    cap_raw_allocation,
    expected_cell_count,
    summarize,
)
from src.path_integral.mlmc import (
    FixedFinestGridTarget,
    LevelAllocation,
    MLMCCheckpoint,
    MLMCHierarchy,
    MLMCPreparedRun,
    OnlineMoments,
    WorkLedger,
)
from src.path_integral.seed_ledger import SeedLedger

ROOT = Path(__file__).resolve().parents[1]


def _config(name: str):
    config, _ = _load_config(ROOT / "configs" / name)
    return config


def test_atomic_json_retries_a_transient_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.g11_m7_confirmatory as runner

    target = tmp_path / "artifact.json"
    target.write_text('{"old":true}', encoding="utf-8")
    real_replace = runner.os.replace
    attempts = 0

    def flaky_replace(source: Path, destination: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError("simulated Windows sharing violation")
        real_replace(source, destination)

    monkeypatch.setattr(runner.os, "replace", flaky_replace)
    _atomic_json(target, {"new": True}, initial_retry_seconds=0.0)
    assert attempts == 2
    assert target.read_text(encoding="utf-8") == '{"new":true}'
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_json_preserves_valid_temporary_after_exhausted_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.g11_m7_confirmatory as runner

    target = tmp_path / "artifact.json"

    def locked_replace(source: Path, destination: Path) -> None:
        raise PermissionError("simulated persistent sharing violation")

    monkeypatch.setattr(runner.os, "replace", locked_replace)
    with pytest.raises(PermissionError):
        _atomic_json(
            target,
            {"recoverable": True},
            replace_attempts=2,
            initial_retry_seconds=0.0,
        )
    temporaries = list(tmp_path.glob("*.tmp"))
    assert len(temporaries) == 1
    assert json.loads(temporaries[0].read_text(encoding="utf-8")) == {"recoverable": True}


def test_unlink_retries_a_transient_sharing_violation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "checkpoint.json"
    target.write_text("{}", encoding="utf-8")
    real_unlink = Path.unlink
    attempts = 0

    def flaky_unlink(path: Path, *, missing_ok: bool = False) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError("simulated Windows sharing violation")
        real_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", flaky_unlink)
    _unlink_with_retry(target, initial_retry_seconds=0.0)
    assert attempts == 2
    assert not target.exists()


def test_progress_counts_only_cells_with_complete_derived_fields(tmp_path: Path) -> None:
    path = tmp_path / "progress.json"
    methods = {"raw_defensive": {}, "dcs_mgi": {}}
    incomplete = {"methods": methods}
    complete = {
        "methods": methods,
        "allocated_work_ratio_raw_over_dcs": 2.0,
        "matched_work_ratio_raw_over_dcs": 2.0,
        "censored_work_ratio_lower_bound": None,
        "wall_ratio_raw_over_dcs": 2.0,
        "paired_estimate_difference": 0.0,
    }
    _write_progress(
        path,
        config_hash="a" * 64,
        cells=[incomplete, complete],
        failures=[],
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == "npi.g11.m7-progress.v2"
    assert payload["complete_method_count"] == 4
    assert payload["complete_cell_count"] == 1


def test_method_result_cache_rejects_a_different_cell_identity(tmp_path: Path) -> None:
    path = tmp_path / "method-result.json"
    _save_method_result_cache(
        path,
        config_hash="a" * 64,
        regime="base",
        task="terminal_1e4",
        replicate=3,
        method="dcs_mgi",
        result={"estimate": 1e-4},
    )
    restored = _load_method_result_cache(
        path,
        config_hash="a" * 64,
        regime="base",
        task="terminal_1e4",
        replicate=3,
        method="dcs_mgi",
    )
    assert restored == {"estimate": 1e-4}
    with pytest.raises(ValueError, match="does not match"):
        _load_method_result_cache(
            path,
            config_hash="a" * 64,
            regime="base",
            task="terminal_1e4",
            replicate=4,
            method="dcs_mgi",
        )


def test_frozen_protocol_selects_640_declared_cells_and_excludes_low_h_excursion() -> None:
    config = _config("g11_m7_confirmatory_v3.yaml")
    contexts, inputs = _load_regimes(config)
    task_counts = {context["name"]: len(context["tasks"]) for context in contexts}
    assert task_counts == {"low_h007": 8, "primary_h012": 12, "high_h030": 12}
    low_tasks = next(context["tasks"] for context in contexts if context["name"] == "low_h007")
    assert {task["name"] for task in low_tasks} == {"terminal", "barrier"}
    assert expected_cell_count(config) == 640
    assert len(inputs) == 6


def test_qualification_protocol_is_disjoint_and_contains_one_cell() -> None:
    qualification = _config("g11_m7_local_qualification.yaml")
    confirmatory = _config("g11_m7_confirmatory_v3.yaml")
    assert qualification["protocol_id"] != confirmatory["protocol_id"]
    assert not qualification["frozen"] and confirmatory["frozen"]
    assert expected_cell_count(qualification) == 1


def test_matrix_qualification_matches_all_32_frozen_task_cells() -> None:
    qualification = _config("g11_m7_matrix_qualification.yaml")
    confirmatory = _config("g11_m7_confirmatory_v3.yaml")
    qualification_contexts, _ = _load_regimes(qualification)
    confirmatory_contexts, _ = _load_regimes(confirmatory)
    assert expected_cell_count(qualification) == 32
    assert {
        context["name"]: [task["id"] for task in context["tasks"]]
        for context in qualification_contexts
    } == {
        context["name"]: [task["id"] for task in context["tasks"]]
        for context in confirmatory_contexts
    }


def test_qualification_preflight_allocates_no_random_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.g11_m7_confirmatory as runner

    monkeypatch.setattr(
        runner,
        "_verify_source",
        lambda config: [dict(item) for item in config["core_source_manifest"]],
    )
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(8)
        result = runner.preflight(ROOT / "configs" / "g11_m7_local_qualification.yaml")
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


def test_raw_pilot_cap_is_resource_censored_with_seed_and_work_evidence() -> None:
    config = _config("g11_m7_local_qualification.yaml")
    config["sampling"] = dict(config["sampling"])
    config["sampling"]["minimum_pilot_nonzero"] = 129
    config["sampling"]["maximum_pilot_samples"] = 128
    contexts, _ = _load_regimes(config)
    result = _method_result(
        config=config,
        context=contexts[0],
        task_item=contexts[0]["tasks"][0],
        replicate=0,
        method="raw_defensive",
    )
    assert result["pilot_resource_censored"]
    assert result["resource_censored"]
    assert result["censor_stage"] == "pilot"
    assert result["estimate"] is None
    assert result["total_work_units"] > 0.0
    assert result["seed_evidence_kind"] == "attempted_batch_seed_manifest"
    assert len(result["seed_evidence_sha256"]) == 64


def test_method_checkpoint_wrapper_preserves_resume_costs(tmp_path: Path) -> None:
    checkpoint = MLMCCheckpoint(
        schema="npi.g11.mlmc-checkpoint.v1",
        protocol="m7-wrapper-test",
        regime="oracle",
        task="linear",
        allocations=(4,),
        next_level=0,
        next_offset=2,
        moments=(OnlineMoments(2, 0.5, 0.25),),
        ledger_payload=SeedLedger().to_dict(),
        work_entries=(),
    )
    path = tmp_path / "method-checkpoint.json"
    _save_method_checkpoint(
        path,
        checkpoint,
        recovery_work_units=123.0,
        recovery_wall_seconds=4.5,
        cumulative_process_cpu_seconds=6.25,
        cumulative_orchestration_seconds=7.5,
    )
    restored, work, wall, process_cpu, orchestration = _load_method_checkpoint(path)
    assert restored == checkpoint
    assert work == 123.0
    assert wall == 4.5
    assert process_cpu == 6.25
    assert orchestration == 7.5


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
