from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

from src.path_integral.v9_terminal_reference_audit import audit_v9_terminal_reference


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    claim = tmp_path / "claim.yaml"
    claim.write_text("claim: true\n", encoding="utf-8")
    config = {
        "schema": "npi.g11.v9-terminal-reference.v1",
        "claim_contract": {"path": "claim.yaml", "sha256": hashlib.sha256(claim.read_bytes()).hexdigest()},
        "model": {"steps": 2},
        "randomizations": 4,
        "points_per_randomization": 8,
        "maximum_relative_standard_error": 1.0,
        "cells": [
            {
                "cell_id": "cell",
                "task": "terminal_left_tail",
                "hurst": 0.1,
                "nominal_probability": 0.1,
                "threshold": 90.0,
            }
        ],
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    units = [0.1, 0.2, 0.3, 0.4]
    mean = 0.25
    standard_error = 0.06454972243679027
    seeds = [100, 200, 201, 202, 203]
    result: dict[str, object] = {
        "schema": "npi.g11.v9-terminal-reference-result.v1",
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "dirty_worktree": False,
        "seed_count": len(seeds),
        "seed_set_sha256": hashlib.sha256(
            json.dumps(sorted(seeds), separators=(",", ":")).encode()
        ).hexdigest(),
        "cells": [
            {
                **config["cells"][0],
                "method": "independent_smoothing_rqmc_reference",
                "proposal_seed": 100,
                "randomization_seeds": [200, 201, 202, 203],
                "randomizations": 4,
                "points_per_randomization": 8,
                "unit_estimates": units,
                "estimate": mean,
                "standard_error": standard_error,
                "relative_standard_error": standard_error / mean,
                "relative_standard_error_pass": True,
                "cost": {
                    "raw_samples": 32,
                    "cdf_calls": 32,
                    "algorithmic_work_units": 32.0 * 9.0,
                },
            }
        ],
        "decision": {
            "reference_complete": True,
            "proposal_bank_authorized": True,
            "benchmark_authorized": False,
            "performance_claim_authorized": False,
            "submission_authorized": False,
        },
    }
    return config_path, result


def test_v9_reference_audit_reconstructs_and_detects_mutation(tmp_path: Path) -> None:
    config, result = _fixture(tmp_path)
    audit = audit_v9_terminal_reference(config_path=config, result=result, root=tmp_path)
    assert audit.passed, audit.failures
    mutated = copy.deepcopy(result)
    mutated["cells"][0]["estimate"] = 0.5  # type: ignore[index]
    failed = audit_v9_terminal_reference(config_path=config, result=mutated, root=tmp_path)
    assert not failed.passed
    assert "randomization_reconstruction" in failed.failures
