"""Independent audit for V9 terminal development and qualification results."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.path_integral.v9_terminal_protocol import (
    aggregate_terminal_benchmark,
    attach_work_to_target,
)


@dataclass(frozen=True)
class V9TerminalBenchmarkAudit:
    checks: tuple[tuple[str, bool], ...]
    failures: tuple[str, ...]
    recomputed_aggregate: dict[str, Any]
    passed: bool


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expected_decision(stage: str, stage_pass: bool) -> dict[str, bool]:
    development = stage == "development"
    return {
        "development_complete": development,
        "qualification_authorized": development and stage_pass,
        "qualification_complete": not development,
        "regime_conditional_empirical_claim_authorized": not development and stage_pass,
        "broad_performance_claim_authorized": False,
        "top_journal_claim_authorized": False,
        "submission_authorized": False,
    }


def audit_v9_terminal_benchmark(
    *, config_path: Path, result: dict[str, Any], root: Path
) -> V9TerminalBenchmarkAudit:
    raw = config_path.read_bytes()
    config = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(config, dict):
        raise ValueError("V9 benchmark config must be a mapping")
    paired = result.get("paired_records")
    external = result.get("external_records")
    if not isinstance(paired, list) or not isinstance(external, list):
        raise ValueError("V9 benchmark records must be lists")
    bindings_valid = True
    for binding in config.get("bindings", {}).values():
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            bindings_valid = False
            break
        path = root / str(binding["path"])
        if not path.is_file() or _sha256(path) != str(binding["sha256"]):
            bindings_valid = False
            break
    bank = json.loads(
        (root / str(config["bindings"]["dcs_proposal_bank"]["path"])).read_text(
            encoding="utf-8"
        )
    )
    raw_paired = deepcopy(paired)
    raw_external = deepcopy(external)
    for record in raw_paired:
        record.pop("work_to_target", None)
        record.pop("comparison", None)
    for record in raw_external:
        record.pop("work_to_target", None)
    expected_paired_records, expected_external_records = attach_work_to_target(
        config=config,
        paired_records=raw_paired,
        external_records=raw_external,
        bank=bank,
    )
    work_reconstruction = paired == expected_paired_records and external == expected_external_records
    aggregate = aggregate_terminal_benchmark(
        config=config,
        paired_records=expected_paired_records,
        external_records=expected_external_records,
    )
    seeds: list[int] = []
    for record in paired:
        seeds.extend((int(record["path_seed"]), int(record["label_seed"])))
    for record in external:
        seeds.extend(int(value) for value in record["seeds"].values())
    expected_seed_range = list(
        range(int(config["base_seed"]), int(config["base_seed"]) + len(seeds))
    )
    seed_hash = hashlib.sha256(
        json.dumps(sorted(seeds), separators=(",", ":")).encode()
    ).hexdigest()
    bound_seeds: set[int] = set()
    reference = json.loads(
        (root / str(config["bindings"]["reference"]["path"])).read_text(encoding="utf-8")
    )
    for cell in reference["cells"]:
        bound_seeds.add(int(cell["proposal_seed"]))
        bound_seeds.update(int(value) for value in cell["randomization_seeds"])
    for entry in bank["entries"]:
        bound_seeds.update(int(rep["seed"]) for rep in entry["replicates"])
    cells = {str(record["cell_id"]) for record in paired}
    clusters = int(config["clusters"])
    methods = len(config["external_methods"]["primary"])
    expected_paired = len(cells) * clusters
    expected_external = expected_paired * methods
    bank_hash = bank["bank_sha256"]
    proposal_bindings = all(
        record["proposal_bank_sha256"] == bank_hash for record in paired
    )
    expected_decision = _expected_decision(str(config["stage"]), bool(aggregate["stage_pass"]))
    checks = (
        ("schema", result.get("schema") == "npi.g11.v9-terminal-benchmark-result.v1"),
        ("config_hash", result.get("config_sha256") == hashlib.sha256(raw).hexdigest()),
        ("bindings", bindings_valid),
        ("stage", result.get("stage") == config.get("stage")),
        ("paired_record_count", len(paired) == expected_paired),
        ("external_record_count", len(external) == expected_external),
        ("seed_uniqueness", len(seeds) == len(set(seeds))),
        ("seed_contiguity", sorted(seeds) == expected_seed_range),
        ("seed_disjoint_from_bound_artifacts", not (set(seeds) & bound_seeds)),
        ("seed_count", result.get("seed_count") == len(seeds)),
        ("seed_hash", result.get("seed_set_sha256") == seed_hash),
        ("proposal_bank_binding", proposal_bindings),
        ("work_reconstruction", work_reconstruction),
        ("aggregate_recomputation", result.get("aggregate") == aggregate),
        ("clean_source_generation", result.get("dirty_worktree") is False),
        ("decision_locks", result.get("decision") == expected_decision),
    )
    failures = tuple(name for name, passed in checks if not passed)
    return V9TerminalBenchmarkAudit(
        checks=checks,
        failures=failures,
        recomputed_aggregate=aggregate,
        passed=not failures,
    )
