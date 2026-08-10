from __future__ import annotations

import math

import torch

from src.path_integral.baseline_framework import BaselineCostLedger, freeze_baseline_proposal
from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.v10r1_full_latent_dcs import (
    evaluate_full_latent_dcs,
    full_latent_integration_basis,
    positive_price_direction,
    proposal_spec,
)


def _problem(steps: int = 4) -> RBergomiBaselineProblem:
    return RBergomiBaselineProblem(
        task_id="oracle-terminal",
        task=TerminalThresholdTask(85.0),
        spot=100.0,
        maturity=1.0,
        steps=steps,
        hurst=0.12,
        eta=1.1,
        xi=0.04,
        rho=-0.7,
    )


def _proposal(problem: RBergomiBaselineProblem):
    mean = tuple(
        0.03 * math.sin(index + 0.4) for index in range(problem.latent_dimension)
    )
    return freeze_baseline_proposal(
        method="defensive_cem",
        task_id=problem.task_id,
        dimension=problem.latent_dimension,
        training_seed=11,
        training_cost=BaselineCostLedger(
            training_samples=8,
            optimizer_steps=1,
            hyperparameter_trials=1,
            algorithmic_work_units=100.0,
        ),
        training_budget_work_units=100.0,
        component_means=(tuple(0.0 for _ in mean), mean),
        component_weights=(0.1, 0.9),
    )


def test_full_latent_basis_retains_all_three_n_coordinates() -> None:
    problem = _problem()
    proposal = _proposal(problem)
    spec = proposal_spec(proposal)
    direction = positive_price_direction(proposal, steps=problem.steps)
    basis = full_latent_integration_basis(steps=problem.steps, direction=direction)

    assert spec.dimension == 3 * problem.steps
    assert basis.shape == (3 * problem.steps, 1)
    assert torch.count_nonzero(basis[: 2 * problem.steps]) == 0
    assert torch.count_nonzero(basis[2 * problem.steps :]) == problem.steps
    assert torch.all(direction > 0.0)
    assert math.isclose(float(torch.linalg.vector_norm(direction)), 1.0, abs_tol=1e-14)


def test_full_latent_dcs_is_exact_and_preserves_auxiliary_local_coordinates() -> None:
    problem = _problem(steps=8)
    proposal = _proposal(problem)
    batch = evaluate_full_latent_dcs(
        problem,
        proposal,
        sample_count=4096,
        gaussian_seed=101,
        label_seed=102,
    )

    assert batch.maximum_local_latent_reconstruction_error < 2e-14
    assert batch.maximum_price_latent_reconstruction_error < 2e-14
    assert batch.maximum_path_reconstruction_error < 2e-13
    assert batch.maximum_coordinate_error < 2e-14
    assert batch.maximum_component_density_error < 2e-13
    assert batch.maximum_mixture_density_error < 2e-13
    assert batch.maximum_full_likelihood_error < 2e-13
    assert batch.maximum_full_bound_violation < 2e-13
    assert batch.maximum_residual_bound_violation < 2e-13
    difference = batch.raw_contribution - batch.dcs_contribution
    difference_z = abs(float(torch.mean(difference))) / float(
        torch.std(difference, unbiased=True) / math.sqrt(difference.numel())
    )
    assert difference_z < 4.0
    assert batch.dcs_cost.likelihood_evaluations == 2 * difference.numel()
    assert batch.raw_cost.likelihood_evaluations == difference.numel()


def test_changing_only_previously_discarded_local_coordinate_changes_result() -> None:
    problem = _problem(steps=8)
    base = _proposal(problem)
    means = [list(row) for row in base.component_means]
    # Index one in each local pair was discarded by exploratory V10.
    for step in range(problem.steps):
        means[1][2 * step + 1] += 0.25
    changed = freeze_baseline_proposal(
        method="defensive_cem",
        task_id=problem.task_id,
        dimension=problem.latent_dimension,
        training_seed=12,
        training_cost=base.training_cost,
        training_budget_work_units=base.training_budget_work_units,
        component_means=tuple(tuple(row) for row in means),
        component_weights=base.component_weights,
    )
    original = evaluate_full_latent_dcs(
        problem, base, sample_count=512, gaussian_seed=201, label_seed=202
    )
    modified = evaluate_full_latent_dcs(
        problem, changed, sample_count=512, gaussian_seed=201, label_seed=202
    )

    assert not torch.equal(original.raw_contribution, modified.raw_contribution)
    assert not torch.equal(original.dcs_contribution, modified.dcs_contribution)
