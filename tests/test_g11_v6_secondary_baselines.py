"""V6 fixed DCS and smoothing-off baseline contract tests."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import yaml

from experiments.g11_v6_materialize_qualification_suite import (
    materialize_qualification_suite,
)
from experiments.g11_v6_routed_policy import _load_config as load_policy
from experiments.g11_v6_secondary_baselines import _load_config
from src.path_integral import (
    SeedKey,
    SeedLedger,
    V6WorkLedger,
    V6WorkRecord,
)

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (
    ROOT
    / "configs"
    / "g11_v6"
    / "routed_policy_cem_anchored_development_v8.yaml"
)


def _training_source(path: Path) -> None:
    records = []
    seed_ledger = SeedLedger()
    work_ledger = V6WorkLedger()
    for index, task in enumerate(
        ("terminal_left_tail", "discrete_lower_barrier")
    ):
        key = SeedKey(
            "g11-v6-test-secondary-proposal-training",
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


def _suite(tmp_path: Path) -> dict[str, Path]:
    source = tmp_path / "training.json"
    _training_source(source)
    payloads, receipt = materialize_qualification_suite(
        TEMPLATE,
        source,
        manifest_cell_count=18,
        clusters=24,
    )
    assert receipt["proposal_bank_receipt"]["proposal_training_audit"][
        "formal_training_source_readiness"
    ]
    paths = {}
    for name, payload in payloads.items():
        path = tmp_path / name
        path.write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )
        paths[name] = path
    return paths


def test_secondary_baseline_config_is_frozen_and_paired(
    tmp_path: Path,
) -> None:
    paths = _suite(tmp_path)
    config, digest = _load_config(
        paths["secondary_baselines_qualification_v1.yaml"]
    )
    assert config["phase"] == "qualification"
    assert config["frozen"]
    assert config["methods"] == ["fixed_dcs_slis", "fixed_raw_defensive"]
    assert config["proposal"]["training_amortization_record_count"] == 18 * 24
    assert len(digest) == 64


def test_selector_off_policy_preserves_router_but_precludes_profile_spend(
    tmp_path: Path,
) -> None:
    paths = _suite(tmp_path)
    config, _digest = load_policy(
        paths["routed_policy_selector_off_qualification_v1.yaml"]
    )
    assert config["router"]["initial_screening_trials"] > 0
    assert config["router"]["maximum_hybrid_profile_work"] == 1.0
    minimum_one_profile_batch = (
        config["selector"]["planning_replicates"]
        * config["selector"]["samples_per_replicate"]
        * config["hierarchy"]["coarsest_steps"]
    )
    assert config["router"]["maximum_hybrid_profile_work"] < minimum_one_profile_batch
