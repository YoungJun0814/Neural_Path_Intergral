"""Independent structural and optional deterministic-replay audit for V10R1 banks."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from experiments.g11_v10r1_proposal_bank import (
    cells_from_config,
    cem_from_config,
    load_config,
    validate_config,
)
from src.path_integral.baseline_framework import BaselineCostLedger
from src.path_integral.v10r1_proposal_bank import (
    V10R1ProposalBankEntry,
    canonical_bank_sha256,
    canonical_proposal_law_sha256,
    proposal_from_dict,
    train_v10r1_proposal_bank,
)


@dataclass(frozen=True)
class V10R1BankAudit:
    checks: tuple[tuple[str, bool], ...]
    failures: tuple[str, ...]
    replay_performed: bool
    passed: bool


def _git_blob(root: Path, commit: str, path: str) -> bytes | None:
    try:
        return subprocess.check_output(
            ("git", "show", f"{commit}:{path}"), cwd=root
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def audit_v10r1_bank(
    *,
    config_path: Path,
    result: dict[str, Any],
    root: Path,
    replay_training: bool,
) -> V10R1BankAudit:
    config, config_sha = load_config(config_path)
    config["config_path"] = config_path.resolve().relative_to(root).as_posix()
    validate_config(config, root=root)
    source_commit = str(result.get("source_commit", ""))
    source_config = _git_blob(root, source_commit, str(result.get("config_path", "")))
    source_config_hash = (
        hashlib.sha256(source_config).hexdigest() if source_config is not None else ""
    )

    raw_entries = result.get("entries")
    rebuilt_entries: list[V10R1ProposalBankEntry] = []
    proposals_valid = isinstance(raw_entries, list)
    law_hashes_valid = isinstance(raw_entries, list)
    if isinstance(raw_entries, list):
        try:
            for raw in raw_entries:
                if not isinstance(raw, dict):
                    raise TypeError("bank entry is not a mapping")
                proposal = proposal_from_dict(raw["proposal"])
                if raw.get("proposal_law_sha256") != canonical_proposal_law_sha256(
                    proposal
                ):
                    law_hashes_valid = False
                rebuilt_entries.append(
                    V10R1ProposalBankEntry(
                        cell_id=str(raw["cell_id"]),
                        replicate=int(raw["replicate"]),
                        training_seed=int(raw["training_seed"]),
                        proposal=proposal,
                    )
                )
        except (KeyError, TypeError, ValueError):
            proposals_valid = False
            law_hashes_valid = False

    cells = cells_from_config(config)
    expected_pairs = [
        (cell.cell_id, replicate)
        for cell in cells
        for replicate in range(int(config["replicates"]))
    ]
    actual_pairs = [(entry.cell_id, entry.replicate) for entry in rebuilt_entries]
    expected_seeds = list(
        range(int(config["base_seed"]), int(config["base_seed"]) + len(expected_pairs))
    )
    actual_seeds = [entry.training_seed for entry in rebuilt_entries]
    seed_payload = json.dumps(actual_seeds, separators=(",", ":")).encode()
    bank_hash = canonical_bank_sha256(rebuilt_entries) if proposals_valid else ""
    total = BaselineCostLedger()
    for entry in rebuilt_entries:
        total = total.plus(entry.proposal.training_cost)

    replay_ok = True
    if replay_training and proposals_valid:
        model = config["model"]
        replay = train_v10r1_proposal_bank(
            cells,
            replicates=int(config["replicates"]),
            spot=float(model["spot"]),
            maturity=float(model["maturity"]),
            steps=int(model["steps"]),
            eta=float(model["eta"]),
            xi=float(model["xi"]),
            rho=float(model["rho"]),
            base_seed=int(config["base_seed"]),
            cem_config=cem_from_config(config),
        )
        replay_ok = [canonical_proposal_law_sha256(entry.proposal) for entry in replay.entries] == [
            canonical_proposal_law_sha256(entry.proposal) for entry in rebuilt_entries
        ]

    checks = (
        (
            "schema",
            result.get("schema")
            == "npi.g11.v10r1-full-latent-proposal-bank-result.v1",
        ),
        ("config_hash", result.get("config_sha256") == config_sha),
        ("clean_source", result.get("dirty_worktree") is False),
        ("source_contains_exact_config", source_config_hash == config_sha),
        ("proposal_payloads", proposals_valid),
        ("proposal_law_hashes", law_hashes_valid),
        ("entry_roster", actual_pairs == expected_pairs),
        ("seed_roster", actual_seeds == expected_seeds),
        ("seed_uniqueness", len(actual_seeds) == len(set(actual_seeds))),
        (
            "seed_hash",
            result.get("seed_set_sha256") == hashlib.sha256(seed_payload).hexdigest(),
        ),
        ("entry_count", result.get("entry_count") == len(expected_pairs)),
        ("full_3n_dimension", all(entry.proposal.dimension == 3 * int(config["model"]["steps"]) for entry in rebuilt_entries)),
        ("bank_hash", result.get("bank_sha256") == bank_hash),
        ("total_training_cost", result.get("total_training_cost") == asdict(total)),
        ("training_replay", replay_ok),
        (
            "decision_lock",
            result.get("decision")
            == {
                "bank_construction_complete": True,
                "bank_audit_required": True,
                "development_authorized": False,
                "performance_claim_authorized": False,
                "submission_authorized": False,
            },
        ),
    )
    failures = tuple(name for name, passed in checks if not passed)
    return V10R1BankAudit(
        checks=checks,
        failures=failures,
        replay_performed=replay_training,
        passed=not failures,
    )
