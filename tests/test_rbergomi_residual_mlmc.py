from __future__ import annotations

import math

import torch

from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.rbergomi_residual_mlmc import (
    RBergomiAdjacentResidualProblem,
    RBergomiResidualMLMCSampler,
    adjacent_positive_price_direction,
    evaluate_adjacent_conditional_residual,
    evaluate_adjacent_residual_transport,
    train_adjacent_residual_transport,
)
from src.path_integral.rbergomi_residual_transport import (
    RBergomiResidualTrainingConfig,
    train_rbergomi_residual_transport,
)
from src.path_integral.residual_transport import (
    ResidualTransportTrainingConfig,
    project_orthogonal,
)


def _problem() -> RBergomiAdjacentResidualProblem:
    return RBergomiAdjacentResidualProblem(
        task_id="adjacent-terminal",
        task=TerminalThresholdTask(80.0),
        spot=100.0,
        maturity=1.0,
        fine_steps=8,
        hurst=0.12,
        eta=1.1,
        xi=0.04,
        rho=-0.7,
    )


def test_adjacent_conditional_correction_is_pathwise_exact_and_signed() -> None:
    problem = _problem()
    direction = adjacent_positive_price_direction(problem)
    latent = torch.randn(
        (1024, problem.latent_dimension),
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(1),
    )
    residual = project_orthogonal(latent, direction)
    result = evaluate_adjacent_conditional_residual(problem, residual, direction)
    assert result.hard_correction_mismatch_count == 0
    assert result.maximum_coordinate_mismatch < 2e-13
    assert result.maximum_path_reconstruction_error < 3e-13
    assert torch.all(torch.isin(result.sign, torch.tensor([-1.0, 0.0, 1.0])))
    assert bool((result.sign > 0).any()) or bool((result.sign < 0).any())


def test_signed_residual_mlmc_is_unbiased_against_paired_raw_correction() -> None:
    problem = _problem()
    proposal = train_adjacent_residual_transport(
        problem,
        sample_count=1024,
        training_seed=10,
        config=ResidualTransportTrainingConfig(components=2, epochs=20),
    )
    batch = evaluate_adjacent_residual_transport(
        problem,
        proposal,
        sample_count=20_000,
        gaussian_seed=20,
        label_seed=21,
        coordinate_seed=22,
    )
    difference = batch.raw_correction - batch.residual_correction
    se = float(torch.std(difference, unbiased=True)) / math.sqrt(difference.numel())
    assert abs(float(torch.mean(difference))) <= 4.0 * se
    norm_se = float(torch.std(batch.likelihood_normalization, unbiased=True)) / math.sqrt(
        batch.likelihood_normalization.numel()
    )
    assert abs(float(torch.mean(batch.likelihood_normalization)) - 1.0) <= 4.0 * norm_se


def test_residual_mlmc_sampler_emits_level_zero_and_signed_level_batch() -> None:
    adjacent = _problem()
    zero_problem = RBergomiBaselineProblem(
        task_id="level-zero",
        task=adjacent.task,
        spot=adjacent.spot,
        maturity=adjacent.maturity,
        steps=adjacent.coarse_steps,
        hurst=adjacent.hurst,
        eta=adjacent.eta,
        xi=adjacent.xi,
        rho=adjacent.rho,
    )
    zero = train_rbergomi_residual_transport(
        zero_problem,
        training_seed=30,
        config=RBergomiResidualTrainingConfig(
            direction_samples=64,
            transport_samples=128,
            direction_rates=(0.0,),
            transport=ResidualTransportTrainingConfig(components=2, epochs=5),
        ),
    ).proposal
    correction = train_adjacent_residual_transport(
        adjacent,
        sample_count=128,
        training_seed=40,
        config=ResidualTransportTrainingConfig(components=2, epochs=5),
    )
    sampler = RBergomiResidualMLMCSampler(
        level_zero_problem=zero_problem,
        level_zero_proposal=zero,
        adjacent_levels={1: (adjacent, correction)},
    )
    streams = {"gaussian": 50, "labels": 51, "coordinate": 52}
    level_zero = sampler(0, "pilot", 64, streams)
    level_one = sampler(1, "final", 64, {"gaussian": 53, "labels": 54, "coordinate": 55})
    assert level_zero.values.shape == (64,)
    assert level_one.values.shape == (64,)
    assert level_zero.work_units > 0.0 and level_one.work_units > 0.0
