from __future__ import annotations

from src.path_integral.dcs_proposal_bank import (
    DCSBankCell,
    DCSBankTrainingConfig,
    train_dcs_proposal_bank,
)
from src.path_integral.path_functionals import TerminalThresholdTask


def _config() -> DCSBankTrainingConfig:
    return DCSBankTrainingConfig(
        segments=2,
        replicates=2,
        paths_per_iteration=64,
        maximum_iterations=2,
        elite_quantile=0.8,
        smoothing=0.5,
        minimum_elite_paths=8,
        control_bound=8.0,
        target_level_repetitions=1,
        minimum_price_driver_magnitude=0.05,
        initial_control=((0.0, -0.5), (0.0, -0.5)),
        mixture_scales=(0.0, 0.5, 1.0),
        mixture_weights=(0.2, 0.3, 0.5),
    )


def test_dcs_proposal_bank_is_replayable_rank_one_and_fully_costed() -> None:
    cell = DCSBankCell("terminal", TerminalThresholdTask(90.0), 0.1)
    kwargs = dict(
        cells=(cell,),
        spot=100.0,
        maturity=1.0,
        steps=8,
        eta=1.0,
        xi=0.04,
        rho=-0.7,
        base_seed=901,
        config=_config(),
    )
    first = train_dcs_proposal_bank(**kwargs)
    second = train_dcs_proposal_bank(**kwargs)
    assert first.bank_sha256 == second.bank_sha256
    assert first.entries[0].schedules == second.entries[0].schedules
    assert first.entries[0].weights == (0.2, 0.3, 0.5)
    assert all(pair == (0.0, 0.0) for pair in first.entries[0].schedules[0])
    base = first.entries[0].schedules[2]
    half = first.entries[0].schedules[1]
    assert half == tuple((0.5 * first_, 0.5 * second_) for first_, second_ in base)
    cost = first.total_training_cost
    assert 2 * 64 <= cost.training_samples <= 2 * 2 * 64
    assert cost.optimizer_steps * 64 == cost.training_samples
    assert cost.algorithmic_work_units > 0.0
    assert cost.wall_seconds > 0.0
    assert cost.measurement_mode == "standardized_hardware_wall"
    assert cost.algorithmic_work_units <= first.entries[0].training_budget_work_units
