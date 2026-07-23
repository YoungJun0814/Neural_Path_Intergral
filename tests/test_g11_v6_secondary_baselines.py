"""V6 fixed DCS and smoothing-off baseline contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from experiments.g11_v6_materialize_qualification_suite import (
    materialize_qualification_suite,
)
from experiments.g11_v6_routed_policy import _load_config as load_policy
from experiments.g11_v6_secondary_baselines import _load_config

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (
    ROOT
    / "configs"
    / "g11_v6"
    / "routed_policy_cem_anchored_development_v8.yaml"
)


def _training_source(path: Path) -> None:
    records = []
    for index, task in enumerate(
        ("terminal_left_tail", "discrete_lower_barrier")
    ):
        records.append(
            {
                "method": "pure_cem",
                "cem_fit": {
                    "control": [
                        [1.0 + index, -2.0],
                        [3.0 + index, -4.0],
                    ]
                },
                "preparation": {"core": {"task": task}},
                "result": {
                    "core": {"complete": True},
                    "total_work": {
                        "records": [
                            {
                                "category": "proposal_training",
                                "samples": 10,
                                "work_units": 20.0 + index,
                                "wall_seconds": 1.0 + index,
                                "cpu_seconds": 2.0 + index,
                            }
                        ]
                    },
                },
            }
        )
    path.write_text(
        json.dumps(
            {
                "schema": "npi.g11.v6-baseline-qualification.v1",
                "source_commit": "a" * 40,
                "dirty_worktree": False,
                "smoke": False,
                "records": records,
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
