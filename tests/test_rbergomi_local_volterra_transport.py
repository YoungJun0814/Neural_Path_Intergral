from __future__ import annotations

import torch

from src.path_integral.baseline_framework import sample_baseline_proposal
from src.path_integral.baselines.conditional_rbergomi import (
    evaluate_conditional_terminal_units,
    freeze_conditional_rbergomi_proposal,
)
from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.rbergomi_local_volterra_transport import (
    LocalVolterraTransportTrainingConfig,
    evaluate_conditional_terminal_local,
    evaluate_local_volterra_transport,
    train_local_volterra_transport,
)
from src.path_integral.residual_smc import AdaptiveResidualSMCConfig


def _problem() -> RBergomiBaselineProblem:
    return RBergomiBaselineProblem(
        task_id="local-volterra-test",
        task=TerminalThresholdTask(80.0),
        spot=100.0,
        maturity=1.0,
        steps=8,
        hurst=0.1,
        eta=1.2,
        xi=0.04,
        rho=-0.7,
    )


def test_local_conditional_matches_natural_baseline_law() -> None:
    problem = _problem()
    local = torch.randn(
        (20_000, problem.local_dimension),
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(1),
    )
    conditional = evaluate_conditional_terminal_local(problem, local)
    assert bool((conditional.conditional_value >= 0.0).all())
    assert bool((conditional.conditional_value <= 1.0).all())


def test_local_conditional_is_pathwise_identical_to_existing_baseline() -> None:
    problem = _problem()
    proposal = freeze_conditional_rbergomi_proposal(problem, training_seed=90)
    local = sample_baseline_proposal(proposal, sample_count=512, seed=91)
    direct = evaluate_conditional_terminal_local(problem, local)
    baseline = evaluate_conditional_terminal_units(problem, proposal, sample_count=512, seed=91)
    torch.testing.assert_close(
        direct.conditional_value,
        baseline.unit_contributions,
        rtol=2e-14,
        atol=1e-15,
    )


def test_local_transport_exact_likelihood_pairing_and_bound() -> None:
    problem = _problem()
    trained = train_local_volterra_transport(
        problem,
        training_seed=10,
        config=LocalVolterraTransportTrainingConfig(
            target_powers=(0.2, 0.6),
            shifted_weights=(0.5, 0.5),
            defensive_weight=0.2,
            smc=AdaptiveResidualSMCConfig(
                particles=64, target_ess_fraction=0.65, pcn_sweeps_per_stage=1
            ),
        ),
    )
    batch = evaluate_local_volterra_transport(
        problem,
        trained.proposal,
        sample_count=20_000,
        proposal_seed=100,
        coordinate_seed=101,
    )
    likelihood_se = float(torch.std(batch.likelihood, unbiased=True)) / 20_000**0.5
    assert abs(float(torch.mean(batch.likelihood)) - 1.0) <= 5 * likelihood_se
    paired = batch.raw_contribution - batch.contribution
    paired_se = float(torch.std(paired, unbiased=True)) / 20_000**0.5
    assert abs(float(torch.mean(paired))) <= 5 * paired_se
    assert float(torch.max(batch.likelihood)) <= 5.0 * (1 + 1e-10)
    assert batch.maximum_likelihood_bound_violation < 1e-9
    assert all(not result.particles_are_final_inferential_units for result in trained.smc_results)


def test_local_transport_can_retain_replicate_centres_as_exact_mixture() -> None:
    problem = _problem()
    trained = train_local_volterra_transport(
        problem,
        training_seed=11,
        config=LocalVolterraTransportTrainingConfig(
            target_powers=(0.2, 0.6),
            shifted_weights=(0.5, 0.5),
            defensive_weight=0.2,
            replicates_per_power=2,
            replicate_aggregation="mixture",
            smc=AdaptiveResidualSMCConfig(
                particles=32,
                target_ess_fraction=0.65,
                pcn_sweeps_per_stage=1,
            ),
        ),
    )
    assert len(trained.proposal.component_means) == 5
    torch.testing.assert_close(
        torch.tensor(trained.proposal.component_weights),
        torch.tensor((0.2, 0.2, 0.2, 0.2, 0.2)),
    )
    assert len(set(trained.all_training_seeds)) == len(trained.all_training_seeds)
    batch = evaluate_local_volterra_transport(
        problem,
        trained.proposal,
        sample_count=10_000,
        proposal_seed=102,
        coordinate_seed=103,
    )
    likelihood_se = float(torch.std(batch.likelihood, unbiased=True)) / 10_000**0.5
    assert abs(float(torch.mean(batch.likelihood)) - 1.0) <= 5 * likelihood_se
    assert float(torch.max(batch.likelihood)) <= 5.0 * (1 + 1e-10)
