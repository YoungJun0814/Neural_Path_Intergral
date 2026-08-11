from __future__ import annotations

import math

import torch

from src.path_integral.baselines.conditional_rbergomi import (
    evaluate_conditional_terminal_units,
    freeze_conditional_rbergomi_proposal,
)
from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.path_functionals import DiscreteBarrierHitTask, TerminalThresholdTask
from src.path_integral.rbergomi_residual_transport import (
    RBergomiResidualTrainingConfig,
    embed_positive_price_direction,
    evaluate_rbergomi_conditional_residual,
    evaluate_rbergomi_residual_transport,
    train_rbergomi_residual_transport,
    validate_rbergomi_residual_direction,
)
from src.path_integral.residual_transport import (
    ResidualTransportTrainingConfig,
    project_orthogonal,
)


def _problem(*, barrier: bool = False) -> RBergomiBaselineProblem:
    task = DiscreteBarrierHitTask(75.0) if barrier else TerminalThresholdTask(70.0)
    return RBergomiBaselineProblem(
        task_id="rb-ecrpt-barrier" if barrier else "rb-ecrpt-terminal",
        task=task,
        spot=100.0,
        maturity=1.0,
        steps=8,
        hurst=0.12,
        eta=1.1,
        xi=0.04,
        rho=-0.7,
    )


def test_rbergomi_residual_threshold_is_pathwise_exact_for_supported_tasks() -> None:
    for problem in (_problem(), _problem(barrier=True)):
        direction = embed_positive_price_direction(
            problem, torch.ones(problem.steps, dtype=torch.float64)
        )
        target = torch.randn(
            (512, problem.latent_dimension),
            dtype=torch.float64,
            generator=torch.Generator().manual_seed(1),
        )
        residual = project_orthogonal(target, direction)
        result = evaluate_rbergomi_conditional_residual(problem, residual, direction)
        assert result.hard_threshold_mismatch_count == 0
        assert result.maximum_coordinate_error < 2e-14
        assert result.maximum_path_reconstruction_error < 2e-13
        assert torch.all((0.0 <= result.conditional_value) & (result.conditional_value <= 1.0))


def test_rbergomi_ecrpt_training_and_paired_evaluation() -> None:
    problem = _problem()
    training = train_rbergomi_residual_transport(
        problem,
        training_seed=100,
        config=RBergomiResidualTrainingConfig(
            direction_samples=256,
            transport_samples=768,
            direction_rates=(-1.0, 0.0, 1.0),
            transport=ResidualTransportTrainingConfig(
                epochs=30,
                learning_rate=0.04,
                components=2,
            ),
        ),
    )
    assert len(training.direction_selection.conditional_second_moments) == 3
    assert training.loss_history[-1] < training.loss_history[0]
    batch = evaluate_rbergomi_residual_transport(
        problem,
        training.proposal,
        sample_count=12_000,
        gaussian_seed=201,
        label_seed=202,
        coordinate_seed=203,
    )
    assert batch.maximum_residual_projection_error < 2e-13
    assert batch.maximum_path_reconstruction_error < 3e-13
    assert batch.maximum_full_path_reconstruction_error < 3e-13
    assert batch.maximum_likelihood_bound_violation < 1e-12
    difference = batch.raw_contribution - batch.ecrpt_contribution
    difference_se = float(torch.std(difference, unbiased=True)) / math.sqrt(difference.numel())
    assert abs(float(torch.mean(difference))) <= 4.0 * difference_se
    assert float(torch.var(batch.ecrpt_contribution, unbiased=True)) <= float(
        torch.var(batch.raw_contribution, unbiased=True)
    )

    # Independent target-law conditioning provides a separate reference estimator.
    natural = freeze_conditional_rbergomi_proposal(problem, training_seed=300)
    reference = evaluate_conditional_terminal_units(
        problem, natural, sample_count=30_000, seed=301
    ).unit_contributions
    estimate = float(torch.mean(batch.ecrpt_contribution))
    estimate_se = float(torch.std(batch.ecrpt_contribution, unbiased=True)) / math.sqrt(
        batch.ecrpt_contribution.numel()
    )
    reference_mean = float(torch.mean(reference))
    reference_se = float(torch.std(reference, unbiased=True)) / math.sqrt(reference.numel())
    assert abs(estimate - reference_mean) <= 4.0 * math.hypot(estimate_se, reference_se)


def test_direction_cannot_enter_volatility_block() -> None:
    problem = _problem()
    direction = embed_positive_price_direction(
        problem, torch.ones(problem.steps, dtype=torch.float64)
    )
    invalid = direction.clone()
    invalid[0] = 0.1
    invalid = invalid / torch.linalg.vector_norm(invalid)
    try:
        validate_rbergomi_residual_direction(problem, invalid)
    except ValueError as error:
        assert "volatility/local" in str(error)
    else:
        raise AssertionError("volatility-dependent integration direction was accepted")


def test_adaptive_training_uses_exact_importance_weights_and_freezes_final_law() -> None:
    problem = _problem()
    training = train_rbergomi_residual_transport(
        problem,
        training_seed=400,
        config=RBergomiResidualTrainingConfig(
            direction_samples=64,
            transport_samples=128,
            direction_rates=(0.0,),
            adaptive_rounds=2,
            adaptive_samples_per_round=128,
            transport=ResidualTransportTrainingConfig(
                epochs=8,
                learning_rate=0.04,
                components=2,
            ),
        ),
    )
    assert len(training.round_normalized_weight_ess) == 3
    assert len(training.round_finite_target_counts) == 3
    assert training.tempering_powers == (1.0 / 3.0, 2.0 / 3.0, 1.0)
    assert training.proposal.training_objective.endswith("cross_entropy")
    assert training.proposal.exact_likelihood
    assert not training.proposal.self_normalized
    batch = evaluate_rbergomi_residual_transport(
        problem,
        training.proposal,
        sample_count=4_000,
        gaussian_seed=501,
        label_seed=502,
        coordinate_seed=503,
    )
    difference = batch.raw_contribution - batch.ecrpt_contribution
    standard_error = float(torch.std(difference, unbiased=True)) / math.sqrt(
        difference.numel()
    )
    assert abs(float(torch.mean(difference))) <= 4.0 * standard_error
