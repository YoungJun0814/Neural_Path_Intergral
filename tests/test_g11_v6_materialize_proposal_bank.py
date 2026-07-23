"""Deterministic source-to-V3 proposal-bank materialization tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from experiments.g11_v6_materialize_proposal_bank import (
    materialize_proposal_policy,
)
from src.path_integral import (
    SeedKey,
    SeedLedger,
    V6WorkLedger,
    V6WorkRecord,
)


def _training_source(path: Path) -> None:
    records = []
    seed_ledger = SeedLedger()
    work_ledger = V6WorkLedger()
    for index, task in enumerate(
        ("terminal_left_tail", "discrete_lower_barrier")
    ):
        key = SeedKey(
            "g11-v6-test-proposal-training",
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
                "cem_fit": {
                    "converged": True,
                    "control": [
                        [1.0 + index, -2.0],
                        [3.0 + index, -4.0],
                    ]
                },
                "training_work_record": asdict(work_record),
            }
        )
    path.write_text(
        json.dumps(
            {
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
        ),
        encoding="utf-8",
    )


def test_materializer_derives_hash_controls_totals_and_exact_record_count(
    tmp_path: Path,
) -> None:
    template = {
        "schema": "npi.g11.v6-routed-policy.config.v2",
        "protocol_id": "development-template",
        "phase": "development",
        "frozen": False,
        "estimand": "fixed_finest_grid",
        "proposal": {"weights": [0.2, 0.3, 0.5]},
        "sampling": {"clusters": 2},
    }
    template_path = tmp_path / "template.yaml"
    template_path.write_text(yaml.safe_dump(template), encoding="utf-8")
    source_path = tmp_path / "training.json"
    _training_source(source_path)

    policy, receipt = materialize_proposal_policy(
        template_path,
        source_path,
        protocol_id="qualification-v1",
        phase="qualification",
        manifest_cell_count=18,
        clusters=24,
    )
    proposal = policy["proposal"]
    assert policy["schema"] == "npi.g11.v6-routed-policy.config.v3"
    assert policy["frozen"]
    assert policy["sampling"]["clusters"] == 24
    assert proposal["training_amortization_record_count"] == 432
    assert proposal["training_total_samples"] == 20
    assert proposal["training_source_artifact_sha256"] == hashlib.sha256(
        source_path.read_bytes()
    ).hexdigest()
    assert proposal["task_controls"]["terminal_left_tail"][1][0] == [
        0.5,
        -1.0,
    ]
    assert receipt["proposal_training_audit"]["verified"]
    assert receipt["proposal_training_audit"]["source_contract_verified"]
    assert isinstance(receipt["formal_readiness"]["clean_source"], bool)


def test_materializer_rejects_a_tampered_training_ledger(
    tmp_path: Path,
) -> None:
    template = {
        "schema": "npi.g11.v6-routed-policy.config.v2",
        "protocol_id": "development-template",
        "phase": "development",
        "frozen": False,
        "estimand": "fixed_finest_grid",
        "proposal": {"weights": [0.2, 0.3, 0.5]},
        "sampling": {"clusters": 2},
    }
    template_path = tmp_path / "template.yaml"
    template_path.write_text(yaml.safe_dump(template), encoding="utf-8")
    source_path = tmp_path / "training.json"
    _training_source(source_path)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["records"][0]["training_work_record"]["samples"] += 1
    source_path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="work does not match the ledger"):
        materialize_proposal_policy(
            template_path,
            source_path,
            protocol_id="qualification-v1",
            phase="qualification",
            manifest_cell_count=18,
            clusters=24,
        )


def test_legacy_combined_baseline_is_not_a_formal_training_source(
    tmp_path: Path,
) -> None:
    template = {
        "schema": "npi.g11.v6-routed-policy.config.v2",
        "protocol_id": "development-template",
        "phase": "development",
        "frozen": False,
        "estimand": "fixed_finest_grid",
        "proposal": {"weights": [0.2, 0.3, 0.5]},
        "sampling": {"clusters": 2},
    }
    template_path = tmp_path / "template.yaml"
    template_path.write_text(yaml.safe_dump(template), encoding="utf-8")
    source_path = tmp_path / "legacy.json"
    source_path.write_text(
        json.dumps(
            {
                "schema": "npi.g11.v6-baseline-qualification.v1",
                "source_commit": "a" * 40,
                "dirty_worktree": False,
                "smoke": False,
                "records": [
                    {
                        "method": "pure_cem",
                        "cem_fit": {"control": [[1.0, -1.0]]},
                        "preparation": {
                            "core": {"task": "terminal_left_tail"}
                        },
                        "result": {
                            "core": {"complete": True},
                            "total_work": {
                                "records": [
                                    {
                                        "category": "proposal_training",
                                        "samples": 10,
                                        "work_units": 10.0,
                                        "wall_seconds": 1.0,
                                        "cpu_seconds": 1.0,
                                    }
                                ]
                            },
                        },
                    },
                    {
                        "method": "pure_cem",
                        "cem_fit": {"control": [[1.0, -1.0]]},
                        "preparation": {
                            "core": {"task": "discrete_lower_barrier"}
                        },
                        "result": {
                            "core": {"complete": True},
                            "total_work": {
                                "records": [
                                    {
                                        "category": "proposal_training",
                                        "samples": 10,
                                        "work_units": 10.0,
                                        "wall_seconds": 1.0,
                                        "cpu_seconds": 1.0,
                                    }
                                ]
                            },
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="clean committed non-smoke training"):
        materialize_proposal_policy(
            template_path,
            source_path,
            protocol_id="qualification-v1",
            phase="qualification",
            manifest_cell_count=18,
            clusters=24,
        )
