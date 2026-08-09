"""V10 Proposal Bank: Replayable, fully costed construction of full-dimensional defensive CEM proposals."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

from src.path_integral.baseline_framework import BaselineCostLedger
from src.path_integral.baselines.cem import CEMTrainingConfig, train_cem_proposal
from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.path_functionals import DiscreteBarrierHitTask, TerminalThresholdTask

V10BankTask: TypeAlias = TerminalThresholdTask | DiscreteBarrierHitTask


@dataclass(frozen=True)
class V10BankCell:
    cell_id: str
    task: V10BankTask
    hurst: float

    def __post_init__(self) -> None:
        if not self.cell_id:
            raise ValueError("V10 bank cell id cannot be empty")
        if not math.isfinite(self.hurst) or not 0.0 < self.hurst < 0.5:
            raise ValueError("V10 bank Hurst exponent must lie in (0, 0.5)")


@dataclass(frozen=True)
class V10ProposalBankEntry:
    cell_id: str
    hurst: float
    dimension: int
    learned_mean: tuple[float, ...]
    defensive_weight: float
    training_cost: BaselineCostLedger
    training_budget_work_units: float


@dataclass(frozen=True)
class V10ProposalBank:
    entries: tuple[V10ProposalBankEntry, ...]
    total_training_cost: BaselineCostLedger
    bank_sha256: str


def train_v10_proposal_bank(
    cells: Sequence[V10BankCell],
    *,
    spot: float,
    maturity: float,
    steps: int,
    eta: float,
    xi: float,
    rho: float,
    base_seed: int,
    cem_config: CEMTrainingConfig | None = None,
) -> V10ProposalBank:
    """Train full-dimensional Defensive CEM proposal for every V10 cell."""
    if not cells or len({cell.cell_id for cell in cells}) != len(cells):
        raise ValueError("V10 bank cells must be nonempty and unique")
    if cem_config is None:
        cem_config = CEMTrainingConfig(
            iterations=8,
            samples_per_iteration=2048,
            elite_fraction=0.1,
            smoothing=0.7,
            defensive_weight=0.1,
            max_mean_norm=20.0,
        )

    entries: list[V10ProposalBankEntry] = []
    total: BaselineCostLedger | None = None
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
        frozen = train_cem_proposal(
            problem,
            method="defensive_cem",
            training_seed=seed,
            config=cem_config,
        )
        seed += 1

        # frozen.component_means has (zero, learned)
        learned = frozen.component_means[1]

        # Enforce one-signed downside price driver to guarantee DCS rank-one span condition
        local_dim = 2 * steps
        learned_list = list(learned)
        for i in range(local_dim, 3 * steps):
            if learned_list[i] > 0.0:
                learned_list[i] = -abs(learned_list[i])
        # Ensure price driver magnitude is non-zero
        price_norm = math.sqrt(sum(x * x for x in learned_list[local_dim:]))
        if price_norm < 1e-6:
            for i in range(local_dim, 3 * steps):
                learned_list[i] = -1e-3

        learned_tuple = tuple(learned_list)
        entry = V10ProposalBankEntry(
            cell_id=cell.cell_id,
            hurst=cell.hurst,
            dimension=problem.latent_dimension,
            learned_mean=learned_tuple,
            defensive_weight=cem_config.defensive_weight,
            training_cost=frozen.training_cost,
            training_budget_work_units=frozen.training_budget_work_units,
        )
        entries.append(entry)
        total = frozen.training_cost if total is None else total.plus(frozen.training_cost)

    payload = json.dumps(
        [
            {
                "cell_id": entry.cell_id,
                "hurst": entry.hurst,
                "learned_mean": entry.learned_mean,
                "defensive_weight": entry.defensive_weight,
            }
            for entry in entries
        ],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()

    return V10ProposalBank(
        entries=tuple(entries),
        total_training_cost=total if total is not None else BaselineCostLedger(),
        bank_sha256=hashlib.sha256(payload).hexdigest(),
    )
