from __future__ import annotations

import torch

from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.low_rank_residual_flow import LowRankResidualFlowTrainingConfig
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.rbergomi_structured_ecrpt import (
    StructuredECRPTTrainingConfig,
    evaluate_structured_ecrpt,
    train_structured_ecrpt,
)
from src.path_integral.residual_smc import AdaptiveResidualSMCConfig


def _problem() -> RBergomiBaselineProblem:
    return RBergomiBaselineProblem(
        task_id="v13-structured-test",
        task=TerminalThresholdTask(level=80.0),
        spot=100.0,
        maturity=1.0,
        steps=8,
        hurst=0.1,
        eta=1.2,
        xi=0.04,
        rho=-0.7,
    )


def _train(seed: int = 17):
    return train_structured_ecrpt(
        _problem(),
        training_seed=seed,
        config=StructuredECRPTTrainingConfig(
            direction_samples=64,
            direction_rates=(-1.0, 0.0, 1.0),
            smc=AdaptiveResidualSMCConfig(
                particles=128,
                target_ess_fraction=0.65,
                pcn_scale=0.3,
                pcn_sweeps_per_stage=1,
            ),
            flow=LowRankResidualFlowTrainingConfig(
                layers=2, rank=2, epochs=5, defensive_weight=0.2
            ),
        ),
    )


def test_training_boundary_schedule_seed_and_cost_contracts() -> None:
    result = _train()
    assert result.smc.final_beta == 1.0
    assert result.smc.final_particles_equally_weighted
    assert not result.smc.particles_are_final_inferential_units
    assert len(result.all_training_seeds) == len(set(result.all_training_seeds))
    assert result.proposal.training_cost.screening_samples == 64 * 3
    assert result.proposal.training_cost.cdf_calls > 64 * 3
    assert result.proposal.exact_likelihood
    assert not result.proposal.self_normalized
    assert result.proposal.frozen


def test_evaluation_is_exact_paired_bounded_and_reconstructs_paths() -> None:
    problem = _problem()
    result = _train()
    batch = evaluate_structured_ecrpt(
        problem,
        result.proposal,
        sample_count=12_000,
        gaussian_seed=101,
        label_seed=102,
        coordinate_seed=103,
        reconstruction_paths=256,
    )
    assert batch.inferential_unit_count == 12_000
    assert batch.raw_cost.final_samples == batch.ecrpt_cost.final_samples == 12_000
    assert batch.ecrpt_cost.cdf_calls == 12_000
    assert batch.raw_cost.cdf_calls == 0
    bound = 1.0 / result.proposal.defensive_weight
    assert float(torch.max(batch.likelihood_normalization)) <= bound * (1.0 + 1e-10)
    likelihood_se = float(torch.std(batch.likelihood_normalization, unbiased=True)) / (
        batch.inferential_unit_count**0.5
    )
    assert abs(float(torch.mean(batch.likelihood_normalization)) - 1.0) <= 5 * likelihood_se
    paired = batch.raw_contribution - batch.ecrpt_contribution
    paired_se = float(torch.std(paired, unbiased=True)) / (paired.numel() ** 0.5)
    assert abs(float(torch.mean(paired))) <= 5 * paired_se
    assert batch.maximum_residual_projection_error < 1e-9
    assert batch.maximum_path_reconstruction_error < 1e-9
    assert batch.maximum_full_path_reconstruction_error < 1e-9
    assert batch.maximum_likelihood_bound_violation < 1e-9
    assert batch.hard_threshold_mismatch_count == 0


def test_training_is_deterministic_except_measured_receipt() -> None:
    first = _train(991)
    second = _train(991)
    assert first.direction_selection == second.direction_selection
    assert first.smc.stages == second.smc.stages
    assert torch.equal(first.smc.residual_particles, second.smc.residual_particles)
    assert first.proposal.layers == second.proposal.layers
    assert first.proposal.sha256 != second.proposal.sha256
