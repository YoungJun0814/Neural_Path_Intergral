from __future__ import annotations

from dataclasses import asdict

from src.path_integral.baselines.cem import CEMTrainingConfig
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.v10r1_proposal_bank import (
    V10R1BankCell,
    bank_entry_to_dict,
    canonical_bank_sha256,
    proposal_from_dict,
    train_v10r1_proposal_bank,
)


def test_v10r1_bank_is_full_dimensional_replayable_and_unmodified() -> None:
    bank = train_v10r1_proposal_bank(
        (V10R1BankCell("tiny", TerminalThresholdTask(80.0), 0.12),),
        replicates=2,
        spot=100.0,
        maturity=1.0,
        steps=4,
        eta=1.1,
        xi=0.04,
        rho=-0.7,
        base_seed=700,
        cem_config=CEMTrainingConfig(iterations=2, samples_per_iteration=32),
    )

    assert bank.training_seeds == (700, 701)
    assert len(bank.entries) == 2
    assert len({entry.proposal.sha256 for entry in bank.entries}) == 2
    for entry in bank.entries:
        assert entry.proposal.dimension == 12
        assert len(entry.proposal.component_means[1]) == 12
        stored = bank_entry_to_dict(entry)
        rebuilt = proposal_from_dict(stored["proposal"])
        assert rebuilt == entry.proposal
        assert asdict(rebuilt) == stored["proposal"]
    assert canonical_bank_sha256(bank.entries) == bank.bank_sha256
