from __future__ import annotations

import math
from typing import Literal, cast

import pytest
import torch

from src.path_integral.baseline_framework import (
    evaluate_baseline_log_q_over_p,
    sample_baseline_proposal,
)
from src.path_integral.baselines import (
    CEMTrainingConfig,
    FlowTrainingConfig,
    LargeDeviationTrainingConfig,
    RBergomiBaselineProblem,
    evaluate_conditional_terminal_units,
    evaluate_latent_is_units,
    evaluate_smoothing_rqmc_units,
    freeze_conditional_rbergomi_proposal,
    freeze_crude_or_antithetic_proposal,
    freeze_smoothing_rqmc_proposal,
    train_cem_proposal,
    train_coupling_flow_proposal,
    train_large_deviation_proposal,
)
from src.path_integral.benchmark_executor import (
    BaselineExecutionRequest,
    PilotSupportError,
    execute_baseline_lifecycle,
)
from src.path_integral.path_functionals import (
    DiscreteBarrierHitTask,
    TerminalThresholdTask,
)


def _terminal_problem(level: float = 90.0) -> RBergomiBaselineProblem:
    return RBergomiBaselineProblem(
        task_id=f"terminal-{level:g}",
        task=TerminalThresholdTask(level),
        spot=100.0,
        maturity=1.0,
        steps=8,
        hurst=0.1,
        eta=1.0,
        xi=0.04,
        rho=-0.7,
    )


def _barrier_problem() -> RBergomiBaselineProblem:
    return RBergomiBaselineProblem(
        task_id="barrier-85",
        task=DiscreteBarrierHitTask(85.0),
        spot=100.0,
        maturity=1.0,
        steps=8,
        hurst=0.1,
        eta=1.0,
        xi=0.04,
        rho=-0.7,
    )


def test_common_latent_contract_is_deterministic_and_log_canonical() -> None:
    problem = _terminal_problem()
    generator = torch.Generator(device="cpu").manual_seed(17)
    latent = torch.randn((32, problem.latent_dimension), generator=generator, dtype=torch.float64)
    first = problem.simulate_latent(latent)
    second = problem.simulate_latent(latent)
    assert torch.equal(first.log_spot, second.log_spot)
    assert torch.equal(first.spot, torch.exp(first.log_spot))
    assert first.proposal_brownian_increments is not None
    assert first.proposal_brownian_increments.shape == (32, problem.steps, 2)
    assert torch.equal(
        problem.hard_event(first),
        problem.task.hard_event_from_log_spot(first.log_spot, first.step_dt),
    )


def test_crude_and_antithetic_use_correct_independent_units() -> None:
    problem = _terminal_problem()
    crude = freeze_crude_or_antithetic_proposal(problem, method="crude_mc", training_seed=1)
    antithetic = freeze_crude_or_antithetic_proposal(
        problem, method="antithetic_mc", training_seed=1
    )
    crude_batch = evaluate_latent_is_units(problem, crude, sample_count=64, seed=2)
    pair_batch = evaluate_latent_is_units(problem, antithetic, sample_count=64, seed=2)
    assert crude_batch.unit_contributions.shape == (64,)
    assert pair_batch.unit_contributions.shape == (32,)
    assert crude_batch.raw_sample_count == pair_batch.raw_sample_count == 64
    assert crude_batch.likelihood_evaluations == pair_batch.likelihood_evaluations == 0
    latent = sample_baseline_proposal(antithetic, sample_count=64, seed=2)
    assert torch.equal(latent[0::2], -latent[1::2])
    with pytest.raises(ValueError, match="even"):
        evaluate_latent_is_units(problem, antithetic, sample_count=63, seed=2)


def test_conditional_terminal_matches_crude_with_combined_sampling_error() -> None:
    problem = _terminal_problem()
    crude = freeze_crude_or_antithetic_proposal(problem, method="crude_mc", training_seed=1)
    conditional = freeze_conditional_rbergomi_proposal(problem, training_seed=1)
    crude_batch = evaluate_latent_is_units(problem, crude, sample_count=8192, seed=21)
    conditional_batch = evaluate_conditional_terminal_units(
        problem, conditional, sample_count=8192, seed=22
    )
    difference = abs(
        float(torch.mean(crude_batch.unit_contributions))
        - float(torch.mean(conditional_batch.unit_contributions))
    )
    variance = float(torch.var(crude_batch.unit_contributions, unbiased=True)) / 8192
    variance += float(torch.var(conditional_batch.unit_contributions, unbiased=True)) / 8192
    assert difference <= 5.0 * math.sqrt(variance)
    assert conditional_batch.cdf_calls == 8192
    assert conditional_batch.likelihood_evaluations == 0


def test_conditional_terminal_rejects_path_dependent_event() -> None:
    with pytest.raises(ValueError, match="terminal"):
        freeze_conditional_rbergomi_proposal(_barrier_problem(), training_seed=1)


def test_smoothing_rqmc_is_reproducible_and_counts_scrambles_as_units() -> None:
    problem = _barrier_problem()
    proposal = freeze_smoothing_rqmc_proposal(problem, training_seed=1)
    first = evaluate_smoothing_rqmc_units(
        problem,
        proposal,
        randomizations=4,
        points_per_randomization=64,
        seed=33,
    )
    second = evaluate_smoothing_rqmc_units(
        problem,
        proposal,
        randomizations=4,
        points_per_randomization=64,
        seed=33,
    )
    assert torch.equal(first.unit_contributions, second.unit_contributions)
    assert first.unit_contributions.shape == (4,)
    assert first.raw_sample_count == first.cdf_calls == 256
    assert bool(((first.unit_contributions >= 0.0) & (first.unit_contributions <= 1.0)).all())


@pytest.mark.parametrize("method", ["pure_cem", "defensive_cem"])
def test_cem_freezes_deterministically_and_charges_training(method: str) -> None:
    problem = _terminal_problem()
    config = CEMTrainingConfig(iterations=2, samples_per_iteration=64)
    typed_method = cast(Literal["pure_cem", "defensive_cem"], method)
    first = train_cem_proposal(problem, method=typed_method, training_seed=41, config=config)
    second = train_cem_proposal(problem, method=typed_method, training_seed=41, config=config)
    assert first.location == second.location
    assert first.component_means == second.component_means
    assert 64 <= first.training_cost.training_samples <= 128
    assert first.training_cost.algorithmic_work_units > 64.0
    assert first.training_cost.algorithmic_work_units <= first.training_budget_work_units
    assert first.training_cost.wall_seconds > 0.0
    assert first.training_cost.cpu_seconds >= 0.0
    assert first.training_cost.peak_memory_bytes > 0
    if method == "defensive_cem":
        assert tuple(0.0 for _ in range(problem.latent_dimension)) in first.component_means
    batch = evaluate_latent_is_units(problem, first, sample_count=128, seed=42)
    assert torch.isfinite(batch.unit_contributions).all()
    assert batch.likelihood_evaluations == 128


def test_large_deviation_action_is_exactly_event_feasible_and_defensive() -> None:
    problem = _terminal_problem(level=99.0)
    proposal = train_large_deviation_proposal(
        problem,
        training_seed=51,
        config=LargeDeviationTrainingConfig(
            optimizer_steps=50,
            learning_rate=0.05,
            penalty=1000.0,
            restarts=2,
        ),
    )
    action = torch.tensor(proposal.component_means[1], dtype=torch.float64).unsqueeze(0)
    assert bool(problem.hard_event(problem.simulate_latent(action))[0])
    assert proposal.component_means[0] == tuple(0.0 for _ in range(problem.latent_dimension))
    assert proposal.training_cost.optimizer_steps == 100
    assert proposal.training_cost.screening_samples > 0


def test_coupling_flow_is_invertible_exact_likelihood_and_costed() -> None:
    problem = _terminal_problem()
    proposal = train_coupling_flow_proposal(
        problem,
        training_seed=61,
        config=FlowTrainingConfig(screening_samples=128, elite_fraction=0.25),
    )
    sample = sample_baseline_proposal(proposal, sample_count=128, seed=62)
    log_ratio = evaluate_baseline_log_q_over_p(sample, proposal)
    assert torch.isfinite(sample).all()
    assert torch.isfinite(log_ratio).all()
    assert proposal.exact_likelihood is True
    assert proposal.self_normalized is False
    assert proposal.conditional_integral == "baseline_only"
    assert proposal.training_cost.screening_samples == 128
    batch = evaluate_latent_is_units(problem, proposal, sample_count=128, seed=62)
    assert torch.isfinite(batch.unit_contributions).all()
    assert batch.likelihood_evaluations == 128


def _small_proposal(method: str):
    problem = _terminal_problem(level=99.0)
    if method in {"crude_mc", "antithetic_mc"}:
        proposal = freeze_crude_or_antithetic_proposal(
            problem,
            method=cast(Literal["crude_mc", "antithetic_mc"], method),
            training_seed=71,
        )
    elif method == "conditional_rbergomi":
        proposal = freeze_conditional_rbergomi_proposal(problem, training_seed=71)
    elif method == "smoothing_rqmc":
        proposal = freeze_smoothing_rqmc_proposal(problem, training_seed=71)
    elif method in {"pure_cem", "defensive_cem"}:
        proposal = train_cem_proposal(
            problem,
            method=cast(Literal["pure_cem", "defensive_cem"], method),
            training_seed=71,
            config=CEMTrainingConfig(iterations=1, samples_per_iteration=32),
        )
    elif method == "ld_subspace_is":
        proposal = train_large_deviation_proposal(
            problem,
            training_seed=71,
            config=LargeDeviationTrainingConfig(optimizer_steps=30, penalty=1000.0, restarts=1),
        )
    elif method == "flow_is":
        proposal = train_coupling_flow_proposal(
            problem,
            training_seed=71,
            config=FlowTrainingConfig(screening_samples=32, elite_fraction=0.25),
        )
    else:
        raise AssertionError(method)
    return problem, proposal


@pytest.mark.parametrize(
    "method",
    [
        "crude_mc",
        "antithetic_mc",
        "conditional_rbergomi",
        "pure_cem",
        "defensive_cem",
        "smoothing_rqmc",
        "ld_subspace_is",
        "flow_is",
    ],
)
def test_common_executor_passes_full_lifecycle_for_every_baseline(method: str) -> None:
    problem, proposal = _small_proposal(method)
    artifact = execute_baseline_lifecycle(
        problem,
        proposal,
        BaselineExecutionRequest(
            pilot_units=4,
            target_estimator_variance=1.0,
            pilot_seed=72,
            final_seed=73,
            minimum_final_units=2,
            maximum_final_units=4,
            rqmc_points_per_randomization=16 if method == "smoothing_rqmc" else 1,
        ),
    )
    assert artifact.audit.passed
    assert artifact.estimate.ordinary_mean
    assert not artifact.estimate.likelihood_clipped
    assert artifact.estimate.unit_count in {2, 3, 4}
    assert artifact.pilot_cost.planning_samples > 0
    assert artifact.estimate.final_cost.final_samples > 0
    assert artifact.estimate.final_cost.wall_seconds > 0.0
    assert artifact.estimate.final_cost.peak_memory_bytes > 0


def test_common_executor_rejects_training_pilot_seed_collision() -> None:
    problem, proposal = _small_proposal("crude_mc")
    with pytest.raises(ValueError, match="disjoint"):
        execute_baseline_lifecycle(
            problem,
            proposal,
            BaselineExecutionRequest(
                pilot_units=4,
                target_estimator_variance=1.0,
                pilot_seed=proposal.training_seed,
                final_seed=73,
            ),
        )


def test_cem_stops_at_the_declared_event_instead_of_chasing_a_deeper_tail() -> None:
    problem = _terminal_problem(level=99.0)
    config = CEMTrainingConfig(iterations=20, samples_per_iteration=512, elite_fraction=0.1)
    proposal = train_cem_proposal(problem, method="pure_cem", training_seed=741, config=config)
    assert proposal.training_cost.optimizer_steps < config.iterations
    assert proposal.training_cost.training_samples == (
        proposal.training_cost.optimizer_steps * config.samples_per_iteration
    )
    assert proposal.training_cost.algorithmic_work_units <= proposal.training_budget_work_units


def test_common_executor_fails_closed_when_rare_event_pilot_has_no_support() -> None:
    problem = _terminal_problem(level=1.0)
    proposal = freeze_crude_or_antithetic_proposal(
        problem, method="crude_mc", training_seed=751
    )
    request = BaselineExecutionRequest(
        pilot_units=16,
        target_estimator_variance=1.0,
        pilot_seed=752,
        final_seed=753,
        minimum_nonzero_pilot_units=1,
    )
    with pytest.raises(PilotSupportError, match="insufficient nonzero"):
        execute_baseline_lifecycle(problem, proposal, request)


def test_common_executor_uses_conservative_planning_variance() -> None:
    problem = _terminal_problem(level=99.0)
    proposal = freeze_crude_or_antithetic_proposal(
        problem, method="crude_mc", training_seed=761
    )
    artifact = execute_baseline_lifecycle(
        problem,
        proposal,
        BaselineExecutionRequest(
            pilot_units=256,
            target_estimator_variance=0.01,
            pilot_seed=762,
            final_seed=763,
            maximum_final_units=512,
            minimum_nonzero_pilot_units=1,
            pilot_variance_safety_factor=4.0,
        ),
    )
    assert artifact.pilot_nonzero_unit_count >= 1
    assert artifact.planning_variance == pytest.approx(4.0 * artifact.pilot_variance)
    assert artifact.plan.pilot_variance == artifact.planning_variance
