"""Replayable proposal bank for the corrected full-latent V10R1 protocol."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from src.path_integral.baseline_framework import (
    BaselineCostLedger,
    FrozenBaselineProposal,
    freeze_baseline_proposal,
)
from src.path_integral.baselines.cem import CEMTrainingConfig, train_cem_proposal
from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.path_functionals import TerminalThresholdTask


@dataclass(frozen=True)
class V10R1BankCell:
    cell_id: str
    task: TerminalThresholdTask
    hurst: float

    def __post_init__(self) -> None:
        if not self.cell_id.strip():
            raise ValueError("V10R1 bank cell id cannot be empty")
        if not math.isfinite(self.hurst) or not 0.0 < self.hurst < 0.5:
            raise ValueError("V10R1 Hurst exponent must lie in (0, 0.5)")


@dataclass(frozen=True)
class V10R1ProposalBankEntry:
    cell_id: str
    replicate: int
    training_seed: int
    proposal: FrozenBaselineProposal


@dataclass(frozen=True)
class V10R1ProposalBank:
    entries: tuple[V10R1ProposalBankEntry, ...]
    training_seeds: tuple[int, ...]
    total_training_cost: BaselineCostLedger
    bank_sha256: str


def _tuples_2d(value: Any) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError("expected a two-dimensional sequence")
    return tuple(tuple(float(item) for item in row) for row in value)


def proposal_from_dict(payload: dict[str, Any]) -> FrozenBaselineProposal:
    """Rebuild and rehash a stored defensive CEM proposal independently."""

    training_cost = payload.get("training_cost")
    if not isinstance(training_cost, dict):
        raise ValueError("stored proposal lacks a training cost ledger")
    rebuilt = freeze_baseline_proposal(
        method="defensive_cem",
        task_id=str(payload["task_id"]),
        dimension=int(payload["dimension"]),
        training_seed=int(payload["training_seed"]),
        training_cost=BaselineCostLedger(**training_cost),
        training_budget_work_units=float(payload["training_budget_work_units"]),
        component_means=_tuples_2d(payload["component_means"]),
        component_weights=tuple(float(x) for x in payload["component_weights"]),
    )
    if payload.get("sha256") != rebuilt.sha256:
        raise ValueError("stored proposal hash does not match its canonical payload")
    return rebuilt


def canonical_proposal_law_sha256(proposal: FrozenBaselineProposal) -> str:
    """Hash only the frozen probability law, excluding measured runtime fields."""

    payload = {
        "method": proposal.method,
        "task_id": proposal.task_id,
        "family": proposal.family,
        "dimension": proposal.dimension,
        "component_means": proposal.component_means,
        "component_weights": proposal.component_weights,
        "exact_likelihood": proposal.exact_likelihood,
        "self_normalized": proposal.self_normalized,
        "conditional_integral": proposal.conditional_integral,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_bank_sha256(entries: Sequence[V10R1ProposalBankEntry]) -> str:
    """Hash ordered cell/replicate identities and canonical proposal hashes."""

    payload = [
        {
            "cell_id": entry.cell_id,
            "replicate": entry.replicate,
            "training_seed": entry.training_seed,
            "proposal_sha256": entry.proposal.sha256,
        }
        for entry in entries
    ]
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def train_v10r1_proposal_bank(
    cells: Sequence[V10R1BankCell],
    *,
    replicates: int,
    spot: float,
    maturity: float,
    steps: int,
    eta: float,
    xi: float,
    rho: float,
    base_seed: int,
    cem_config: CEMTrainingConfig,
) -> V10R1ProposalBank:
    """Train independent full-``3N`` proposals without post-training projection."""

    if not cells or len({cell.cell_id for cell in cells}) != len(cells):
        raise ValueError("V10R1 bank cells must be nonempty and unique")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
        raise ValueError("proposal replicates must be a positive integer")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("proposal base seed must be a nonnegative integer")

    entries: list[V10R1ProposalBankEntry] = []
    total = BaselineCostLedger()
    seed = base_seed
    for cell in cells:
        problem = RBergomiBaselineProblem(
            task_id=cell.cell_id,
            task=cell.task,
            spot=spot,
            maturity=maturity,
            steps=steps,
            hurst=cell.hurst,
            eta=eta,
            xi=xi,
            rho=rho,
        )
        for replicate in range(replicates):
            proposal = train_cem_proposal(
                problem,
                method="defensive_cem",
                training_seed=seed,
                config=cem_config,
            )
            if proposal.dimension != 3 * steps:
                raise AssertionError("CEM training did not retain the complete latent dimension")
            entry = V10R1ProposalBankEntry(
                cell_id=cell.cell_id,
                replicate=replicate,
                training_seed=seed,
                proposal=proposal,
            )
            entries.append(entry)
            total = total.plus(proposal.training_cost)
            seed += 1
    frozen_entries = tuple(entries)
    return V10R1ProposalBank(
        entries=frozen_entries,
        training_seeds=tuple(entry.training_seed for entry in frozen_entries),
        total_training_cost=total,
        bank_sha256=canonical_bank_sha256(frozen_entries),
    )


def bank_entry_to_dict(entry: V10R1ProposalBankEntry) -> dict[str, Any]:
    """Return a JSON-ready entry while keeping the canonical proposal payload intact."""

    return {
        "cell_id": entry.cell_id,
        "replicate": entry.replicate,
        "training_seed": entry.training_seed,
        "proposal_law_sha256": canonical_proposal_law_sha256(entry.proposal),
        "proposal": asdict(entry.proposal),
    }
