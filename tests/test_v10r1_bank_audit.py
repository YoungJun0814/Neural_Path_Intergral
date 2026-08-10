from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import yaml

import src.path_integral.v10r1_bank_audit as audit_module
from src.path_integral.baselines.cem import CEMTrainingConfig
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.v10r1_bank_audit import audit_v10r1_bank
from src.path_integral.v10r1_proposal_bank import (
    V10R1BankCell,
    bank_entry_to_dict,
    train_v10r1_proposal_bank,
)


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    cell = {
        "cell_id": "tiny",
        "task": "terminal_left_tail",
        "hurst": 0.12,
        "nominal_probability": 0.1,
        "threshold": 80.0,
    }
    model = {
        "spot": 100.0,
        "maturity": 1.0,
        "steps": 4,
        "eta": 1.1,
        "xi": 0.04,
        "rho": -0.7,
    }
    claim = {
        "schema": "npi.g11.v10r1-terminal-claim-contract.v1",
        "proposal_replicates": 2,
        "model": model,
        "cells": [cell],
    }
    claim_path = tmp_path / "claim.yaml"
    claim_path.write_text(yaml.safe_dump(claim), encoding="utf-8")
    training = {
        "iterations": 2,
        "samples_per_iteration": 32,
        "elite_fraction": 0.1,
        "smoothing": 0.7,
        "defensive_weight": 0.1,
        "max_mean_norm": 20.0,
        "time_bins": None,
    }
    config = {
        "schema": "npi.g11.v10r1-full-latent-proposal-bank.v1",
        "protocol_id": "test",
        "namespace": "test",
        "base_seed": 900,
        "replicates": 2,
        "current_namespace_outcomes_inspected_before_freeze": False,
        "claim_contract": {
            "path": "claim.yaml",
            "sha256": hashlib.sha256(claim_path.read_bytes()).hexdigest(),
        },
        "model": model,
        "training": training,
        "cells": [cell],
    }
    config_path = tmp_path / "bank.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    bank = train_v10r1_proposal_bank(
        (V10R1BankCell("tiny", TerminalThresholdTask(80.0), 0.12),),
        replicates=2,
        **model,
        base_seed=900,
        cem_config=CEMTrainingConfig(**training),
    )
    seeds = list(bank.training_seeds)
    result: dict[str, object] = {
        "schema": "npi.g11.v10r1-full-latent-proposal-bank-result.v1",
        "config_path": "bank.yaml",
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "source_commit": "test-commit",
        "dirty_worktree": False,
        "entries": [bank_entry_to_dict(entry) for entry in bank.entries],
        "entry_count": len(bank.entries),
        "training_seeds": seeds,
        "seed_set_sha256": hashlib.sha256(
            json.dumps(seeds, separators=(",", ":")).encode()
        ).hexdigest(),
        "bank_sha256": bank.bank_sha256,
        "total_training_cost": asdict(bank.total_training_cost),
        "decision": {
            "bank_construction_complete": True,
            "bank_audit_required": True,
            "development_authorized": False,
            "performance_claim_authorized": False,
            "submission_authorized": False,
        },
    }
    return config_path, result


def test_bank_audit_replays_and_rejects_mutated_proposal(
    tmp_path: Path, monkeypatch
) -> None:
    config, result = _fixture(tmp_path)
    monkeypatch.setattr(
        audit_module,
        "_git_blob",
        lambda _root, _commit, _path: config.read_bytes(),
    )
    audit = audit_v10r1_bank(
        config_path=config,
        result=result,
        root=tmp_path,
        replay_training=True,
    )
    assert audit.passed, audit.failures

    mutated = copy.deepcopy(result)
    proposal = mutated["entries"][0]["proposal"]  # type: ignore[index]
    means = [list(row) for row in proposal["component_means"]]  # type: ignore[index]
    means[1][1] += 0.25
    proposal["component_means"] = means  # type: ignore[index]
    failed = audit_v10r1_bank(
        config_path=config,
        result=mutated,
        root=tmp_path,
        replay_training=False,
    )
    assert not failed.passed
    assert "proposal_payloads" in failed.failures
