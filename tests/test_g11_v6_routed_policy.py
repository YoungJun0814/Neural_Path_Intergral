"""End-to-end V6 routed-policy smoke test."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from experiments.g11_v6_rarity_calibration import run as run_calibration
from experiments.g11_v6_reference import run as run_reference
from experiments.g11_v6_result_audit import run as run_audit
from experiments.g11_v6_routed_policy import (
    _apportion_shared_training,
    _crude_work_interval,
    _load_config,
    _task_conditioned_training_source_audit,
)
from experiments.g11_v6_routed_policy import run as run_policy
from src.path_integral import (
    SeedKey,
    SeedLedger,
    V6WorkLedger,
    V6WorkRecord,
)

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / "configs" / "g11_v6" / "rarity_calibration_development.yaml"
REFERENCE = ROOT / "configs" / "g11_v6" / "reference_development.yaml"
POLICY = ROOT / "configs" / "g11_v6" / "routed_policy_development.yaml"
POLICY_V2 = ROOT / "configs" / "g11_v6" / "routed_policy_replicated_pilot_v3.yaml"
POLICY_V4 = ROOT / "configs" / "g11_v6" / "routed_policy_cem_anchored_pilot_v4.yaml"
POLICY_V7 = (
    ROOT / "configs" / "g11_v6" / "routed_policy_cem_anchored_development_v7.yaml"
)
POLICY_V8 = (
    ROOT / "configs" / "g11_v6" / "routed_policy_cem_anchored_development_v8.yaml"
)
AUDIT = ROOT / "configs" / "g11_v6" / "result_audit_development.yaml"


def test_v6_routed_policy_config_is_strict() -> None:
    config, digest = _load_config(POLICY)
    assert config["router"]["probability_cutoff"] == 0.05
    assert len(digest) == 64

    replicated, replicated_digest = _load_config(POLICY_V2)
    assert replicated["schema"] == "npi.g11.v6-routed-policy.config.v2"
    assert replicated["selector"]["decision_mode"] == "replicated_planning"
    assert replicated["sampling"]["allocation_safety_factor"] >= 1.0
    assert len(replicated_digest) == 64

    anchored, anchored_digest = _load_config(POLICY_V4)
    assert anchored["selector"]["planning_variance_statistic"] == (
        "mean_replicate_variance"
    )
    assert set(anchored["proposal"]["task_controls"]) == {
        "terminal_left_tail",
        "discrete_lower_barrier",
    }
    assert len(anchored_digest) == 64

    exact_bank, exact_bank_digest = _load_config(POLICY_V7)
    assert exact_bank["schema"] == "npi.g11.v6-routed-policy.config.v3"
    assert exact_bank["proposal"]["training_amortization_record_count"] == 16
    assert exact_bank["proposal"]["training_total_samples"] == 196608
    assert len(exact_bank_digest) == 64

    coverage, coverage_digest = _load_config(POLICY_V8)
    assert coverage["sampling"]["clusters"] == 24
    assert coverage["sampling"]["minimum_final_samples"] == 8192
    assert coverage["proposal"]["training_amortization_record_count"] == 48
    assert len(coverage_digest) == 64


def test_router_work_interval_charges_the_execution_floor_after_zero_hits() -> None:
    interval = _crude_work_interval(
        torch.zeros(256, dtype=torch.float64),
        cost=10.0,
        preprocessing_work=100.0,
        target=1e-8,
        confidence_level=0.99,
        minimum_final_samples=4096,
    )
    assert interval.lower == 100.0 + 4096 * 10.0
    assert interval.point == interval.lower
    assert interval.upper > interval.point


def test_v3_shared_training_apportionment_conserves_nondivisible_totals() -> None:
    proposal = {
        "training_amortization_record_count": 6,
        "training_total_samples": 20,
        "training_total_work_units": 41.0,
        "training_total_wall_seconds": 3.0,
        "training_total_cpu_seconds": 5.0,
    }
    allocations, contract = _apportion_shared_training(proposal, 6)
    assert [item["samples"] for item in allocations] == [4, 4, 3, 3, 3, 3]
    assert sum(int(item["samples"]) for item in allocations) == 20
    assert sum(float(item["work_units"]) for item in allocations) == pytest.approx(41.0)
    assert sum(float(item["wall_seconds"]) for item in allocations) == pytest.approx(3.0)
    assert sum(float(item["cpu_seconds"]) for item in allocations) == pytest.approx(5.0)
    assert contract["integer_sample_remainder"] == 2
    assert contract["rule"].startswith("manifest_order_then_cluster")


def test_v3_shared_training_apportionment_rejects_matrix_drift() -> None:
    proposal = {
        "training_amortization_record_count": 5,
        "training_total_samples": 20,
        "training_total_work_units": 41.0,
        "training_total_wall_seconds": 3.0,
        "training_total_cpu_seconds": 5.0,
    }
    with pytest.raises(ValueError, match="executed cell-cluster matrix"):
        _apportion_shared_training(proposal, 6)
    allocations, contract = _apportion_shared_training(
        proposal, 6, enforce_declared_count=False
    )
    assert len(allocations) == 6
    assert not contract["declared_count_enforced"]


def test_v3_proposal_bank_is_rederived_from_its_hashed_training_source(
    tmp_path: Path,
) -> None:
    controls = {
        "terminal_left_tail": [[1.0, -2.0], [3.0, -4.0]],
        "discrete_lower_barrier": [[2.0, -1.0], [4.0, -3.0]],
    }
    records = []
    seed_ledger = SeedLedger()
    work_ledger = V6WorkLedger()
    for index, (task, control) in enumerate(controls.items()):
        key = SeedKey(
            "g11-v6-test-routed-proposal-training",
            "proposal-training",
            f"cell-{index}:cluster-0",
            "pure_cem",
            0,
            0,
            "proposal",
        )
        seed = seed_ledger.allocate(key)
        work_record = V6WorkRecord(
            category="proposal_training",
            method="pure_cem",
            cell_id=f"cell-{index}",
            attempt=0,
            samples=10,
            work_units=20.0 + index,
            wall_seconds=1.0 + index,
            cpu_seconds=2.0 + index,
            peak_memory_bytes=0,
            successful=True,
        )
        work_ledger = work_ledger.append(work_record)
        records.append(
            {
                "cell_id": f"cell-{index}",
                "task": task,
                "cluster": 0,
                "method": "pure_cem",
                "seed_key": asdict(key),
                "seed": seed,
                "cem_fit": {"converged": True, "control": control},
                "training_work_record": asdict(work_record),
            }
        )
    source = {
        "schema": "npi.g11.v6-proposal-training.v1",
        "source_commit": "a" * 40,
        "dirty_worktree": False,
        "smoke": False,
        "records": records,
        "work_ledger": work_ledger.to_dict(),
        "work_ledger_sha256": work_ledger.sha256,
        "seed_ledger": seed_ledger.to_dict(),
        "seed_ledger_sha256": seed_ledger.sha256,
        "gates": {"complete": True},
        "formal_readiness": {"clean": True},
        "proposal_training_qualified": True,
    }
    source_path = tmp_path / "training.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    proposal = {
        "training_source_artifact_sha256": hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest(),
        "task_controls": {
            task: [
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.5 * value for value in segment] for segment in control],
                control,
            ]
            for task, control in controls.items()
        },
        "training_source_record_count": 2,
        "training_total_samples": 20,
        "training_total_work_units": 41.0,
        "training_total_wall_seconds": 3.0,
        "training_total_cpu_seconds": 5.0,
    }
    audit = _task_conditioned_training_source_audit(proposal, source_path)
    assert audit["verified"]
    assert audit["total_samples"] == 20
    assert audit["formal_training_source_readiness"]

    proposal["task_controls"]["terminal_left_tail"][2][0][0] += 0.01
    with pytest.raises(ValueError, match="declared derivation"):
        _task_conditioned_training_source_audit(proposal, source_path)


@pytest.mark.slow
def test_v6_routed_policy_smoke_is_auditable_and_unqualified(tmp_path: Path) -> None:
    calibration = run_calibration(CALIBRATION, smoke=True)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(calibration["candidate_manifest"]), encoding="utf-8")
    reference = run_reference(REFERENCE, manifest_path, smoke=True)
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(reference), encoding="utf-8")
    checkpoint_directory = tmp_path / "policy-progress"
    result = run_policy(
        POLICY,
        manifest_path,
        reference_path,
        smoke=True,
        checkpoint_directory=checkpoint_directory,
    )
    assert result["schema"] == "npi.g11.v6-routed-policy.v1"
    assert len(result["records"]) == 2
    assert result["gates"]["complete_matrix"]
    assert result["gates"]["all_routes_resolved"]
    assert result["gates"]["all_independent_audits"]
    assert not result["policy_qualified"]
    artifact_path = tmp_path / "routed.json"
    artifact_path.write_text(json.dumps(result), encoding="utf-8")
    offline = run_audit(AUDIT, artifact_path)
    assert offline["gates"]["all_records_pass"]
    assert not offline["qualification_audit_passed"]
    resumed = run_policy(
        POLICY,
        manifest_path,
        reference_path,
        smoke=True,
        checkpoint_directory=checkpoint_directory,
        resume=True,
    )
    assert [record["result"]["result_hash"] for record in resumed["records"]] == [
        record["result"]["result_hash"] for record in result["records"]
    ]
