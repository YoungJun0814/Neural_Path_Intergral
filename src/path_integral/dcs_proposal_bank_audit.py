"""Independent audit of replayable DCS proposal-bank construction."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DCSProposalBankAudit:
    checks: tuple[tuple[str, bool], ...]
    failures: tuple[str, ...]
    passed: bool


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bank_hash(entries: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        [
            {
                "cell_id": entry["cell_id"],
                "hurst": entry["hurst"],
                "schedules": entry["schedules"],
                "weights": entry["weights"],
            }
            for entry in entries
        ],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def audit_dcs_proposal_bank(
    *, config_path: Path, result: dict[str, Any], root: Path
) -> DCSProposalBankAudit:
    raw = config_path.read_bytes()
    config = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(config, dict):
        raise ValueError("DCS proposal-bank config must be a mapping")
    entries = result.get("entries")
    if not isinstance(entries, list):
        raise ValueError("DCS proposal-bank entries must be a list")
    training = config["training"]
    expected_cells = [str(cell["cell_id"]) for cell in config["cells"]]
    scales = [float(value) for value in training["mixture_scales"]]
    weights = [float(value) for value in training["mixture_weights"]]

    bindings_valid = True
    for binding in config.get("bindings", {}).values():
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            bindings_valid = False
            break
        path = root / str(binding["path"])
        if not path.is_file() or _sha256(path) != binding["sha256"]:
            bindings_valid = False
            break

    seeds = [int(rep["seed"]) for entry in entries for rep in entry["replicates"]]
    expected_seeds = list(range(int(config["base_seed"]), int(config["base_seed"]) + len(seeds)))
    structure = [entry["cell_id"] for entry in entries] == expected_cells
    mixture_valid = True
    costs_valid = True
    summed = {
        "training_samples": 0,
        "optimizer_steps": 0,
        "hyperparameter_trials": 0,
        "algorithmic_work_units": 0.0,
        "wall_seconds": 0.0,
        "cpu_seconds": 0.0,
        "peak_memory_bytes": 0,
    }
    for entry in entries:
        schedules = entry["schedules"]
        if len(schedules) != len(scales) or [float(value) for value in entry["weights"]] != weights:
            mixture_valid = False
            break
        base = schedules[scales.index(1.0)] if 1.0 in scales else None
        if base is None:
            mixture_valid = False
            break
        for scale, schedule in zip(scales, schedules, strict=True):
            for pair, base_pair in zip(schedule, base, strict=True):
                if not all(
                    math.isclose(float(pair[index]), scale * float(base_pair[index]), abs_tol=1e-12)
                    for index in range(2)
                ):
                    mixture_valid = False
                    break
        cost = entry["training_cost"]
        iterations = sum(int(rep["iterations"]) for rep in entry["replicates"])
        if (
            int(cost["optimizer_steps"]) != iterations
            or int(cost["training_samples"])
            != iterations * int(training["paths_per_iteration"])
            or float(cost["algorithmic_work_units"])
            > float(entry["training_budget_work_units"])
        ):
            costs_valid = False
        for key in summed:
            if key == "peak_memory_bytes":
                summed[key] = max(int(summed[key]), int(cost[key]))
            else:
                summed[key] += cost[key]

    total = result["total_training_cost"]
    total_cost_valid = costs_valid and all(
        math.isclose(float(total[key]), float(value), rel_tol=1e-12, abs_tol=1e-12)
        for key, value in summed.items()
    )
    config_schema = str(config.get("schema"))
    if config_schema == "npi.g11.v9-terminal-proposal-bank.v1":
        result_schema = "npi.g11.v9-terminal-proposal-bank-result.v1"
        expected_decision = {
            "bank_construction_complete": True,
            "development_benchmark_authorized": True,
            "qualification_authorized": False,
            "performance_claim_authorized": False,
            "submission_authorized": False,
        }
    else:
        result_schema = "npi.g11.v8-dcs-proposal-bank-result.v1"
        expected_decision = {
            "bank_construction_complete": True,
            "stage_b_use_authorized": True,
            "p8_qualification_authorized": False,
            "performance_claim_authorized": False,
            "submission_authorized": False,
        }
    decision = result.get("decision")
    claim_locks = isinstance(decision, dict) and decision == expected_decision
    checks = (
        ("schema", result.get("schema") == result_schema),
        ("config_hash", result.get("config_sha256") == hashlib.sha256(raw).hexdigest()),
        ("bindings", bindings_valid),
        ("entry_roster", structure),
        ("mixture_rank_one", mixture_valid),
        ("seed_uniqueness", len(seeds) == len(set(seeds))),
        ("seed_contiguity", sorted(seeds) == expected_seeds),
        ("seed_count", result.get("seed_count") == len(seeds)),
        ("entry_costs", costs_valid),
        ("total_cost", total_cost_valid),
        ("bank_hash", result.get("bank_sha256") == _bank_hash(entries)),
        ("claim_locks", claim_locks),
    )
    failures = tuple(name for name, passed in checks if not passed)
    return DCSProposalBankAudit(checks=checks, failures=failures, passed=not failures)
