from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path

import yaml

from src.path_integral.dcs_proposal_bank import (
    DCSBankCell,
    DCSBankTrainingConfig,
    train_dcs_proposal_bank,
)
from src.path_integral.dcs_proposal_bank_audit import audit_dcs_proposal_bank
from src.path_integral.path_functionals import TerminalThresholdTask


def _fixture(tmp_path: Path):
    bank = train_dcs_proposal_bank(
        (DCSBankCell("cell", TerminalThresholdTask(90.0), 0.1),),
        spot=100.0,
        maturity=1.0,
        steps=8,
        eta=1.0,
        xi=0.04,
        rho=-0.7,
        base_seed=11,
        config=DCSBankTrainingConfig(
            segments=2,
            replicates=1,
            paths_per_iteration=32,
            maximum_iterations=1,
            elite_quantile=0.8,
            smoothing=0.5,
            minimum_elite_paths=4,
            control_bound=8.0,
            target_level_repetitions=1,
            minimum_price_driver_magnitude=0.05,
            initial_control=((0.0, -0.5), (0.0, -0.5)),
            mixture_scales=(0.0, 1.0),
            mixture_weights=(0.2, 0.8),
        ),
    )
    config = {
        "schema": "npi.g11.v8-dcs-proposal-bank.v1",
        "bindings": {},
        "base_seed": 11,
        "cells": [{"cell_id": "cell"}],
        "training": {
            "replicates": 1,
            "paths_per_iteration": 32,
            "mixture_scales": [0.0, 1.0],
            "mixture_weights": [0.2, 0.8],
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    result = {
        "schema": "npi.g11.v8-dcs-proposal-bank-result.v1",
        "config_sha256": __import__("hashlib").sha256(config_path.read_bytes()).hexdigest(),
        "entries": [asdict(entry) for entry in bank.entries],
        "seed_count": 1,
        "bank_sha256": bank.bank_sha256,
        "total_training_cost": asdict(bank.total_training_cost),
        "decision": {
            "bank_construction_complete": True,
            "stage_b_use_authorized": True,
            "p8_qualification_authorized": False,
            "performance_claim_authorized": False,
            "submission_authorized": False,
        },
    }
    return config_path, json.loads(json.dumps(result))


def test_dcs_proposal_bank_audit_passes_and_detects_mutation(tmp_path: Path) -> None:
    config_path, result = _fixture(tmp_path)
    audit = audit_dcs_proposal_bank(config_path=config_path, result=result, root=tmp_path)
    assert audit.passed, audit.failures
    mutated = copy.deepcopy(result)
    mutated["entries"][0]["schedules"][1][0][0] += 0.5
    failed = audit_dcs_proposal_bank(
        config_path=config_path, result=mutated, root=tmp_path
    )
    assert not failed.passed
    assert "mixture_rank_one" in failed.failures or "bank_hash" in failed.failures


def test_dcs_proposal_bank_audit_supports_v9_claim_locks(tmp_path: Path) -> None:
    config_path, result = _fixture(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    claim = tmp_path / "claim.yaml"
    claim.write_text("claim: true\n", encoding="utf-8")
    config["schema"] = "npi.g11.v9-terminal-proposal-bank.v1"
    config["claim_contract"] = {
        "path": "claim.yaml",
        "sha256": __import__("hashlib").sha256(claim.read_bytes()).hexdigest(),
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    result["schema"] = "npi.g11.v9-terminal-proposal-bank-result.v1"
    result["config_sha256"] = __import__("hashlib").sha256(config_path.read_bytes()).hexdigest()
    result["dirty_worktree"] = False
    result["decision"] = {
        "bank_construction_complete": True,
        "development_benchmark_authorized": True,
        "qualification_authorized": False,
        "performance_claim_authorized": False,
        "submission_authorized": False,
    }
    audit = audit_dcs_proposal_bank(config_path=config_path, result=result, root=tmp_path)
    assert audit.passed, audit.failures
