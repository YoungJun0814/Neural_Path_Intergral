from __future__ import annotations

import torch

from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.low_rank_residual_flow import LowRankResidualFlowTrainingConfig
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.rbergomi_tempered_ecrpt import (
    TemperedECRPTTrainingConfig,
    evaluate_tempered_ecrpt,
    train_tempered_ecrpt,
)
from src.path_integral.residual_smc import AdaptiveResidualSMCConfig


def _problem() -> RBergomiBaselineProblem:
    return RBergomiBaselineProblem(
        task_id="v14-test",
        task=TerminalThresholdTask(80.0),
        spot=100.0,
        maturity=1.0,
        steps=8,
        hurst=0.1,
        eta=1.2,
        xi=0.04,
        rho=-0.7,
    )


def _training():
    return train_tempered_ecrpt(
        _problem(),
        training_seed=10,
        config=TemperedECRPTTrainingConfig(
            direction_samples=32,
            target_powers=(0.5, 1.0),
            mixture_weights=(0.5, 0.5),
            smc=AdaptiveResidualSMCConfig(
                particles=64, pcn_sweeps_per_stage=1, target_ess_fraction=0.65
            ),
            flow=LowRankResidualFlowTrainingConfig(
                layers=2, rank=2, epochs=3, defensive_weight=0.2
            ),
        ),
    )


def test_tempered_rbergomi_training_and_evaluation_contract() -> None:
    trained = _training()
    assert len(trained.smc_results) == 2
    assert all(result.final_beta == 1.0 for result in trained.smc_results)
    assert all(not result.particles_are_final_inferential_units for result in trained.smc_results)
    batch = evaluate_tempered_ecrpt(
        _problem(),
        trained.proposal,
        sample_count=10_000,
        proposal_seed=200,
        coordinate_seed=201,
    )
    likelihood_se = float(torch.std(batch.likelihood_normalization, unbiased=True)) / 100.0
    assert abs(float(torch.mean(batch.likelihood_normalization)) - 1.0) <= 5 * likelihood_se
    difference = batch.raw_contribution - batch.ecrpt_contribution
    difference_se = float(torch.std(difference, unbiased=True)) / 100.0
    assert abs(float(torch.mean(difference))) <= 5 * difference_se
    assert float(torch.max(batch.likelihood_normalization)) <= trained.proposal.likelihood_bound * (
        1 + 1e-10
    )
    assert batch.maximum_residual_projection_error < 1e-9
    assert batch.maximum_full_path_reconstruction_error < 1e-9
    assert batch.hard_threshold_mismatch_count == 0
