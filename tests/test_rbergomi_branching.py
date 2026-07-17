"""Exact-law and variance tests for conditional Volterra bridge branching."""

from __future__ import annotations

import math

import pytest
import torch

from src.evaluation.volterra_branching import (
    BoundaryBranchingPolicy,
    boundary_branch_counts,
    calibrate_boundary_branching_policy,
    evaluate_adaptive_branched_correction,
)
from src.path_integral import (
    DownsideExcursionTask,
    TimePiecewiseTwoDriverControl,
    brownian_log_likelihood,
)
from src.path_integral.rbergomi_branching import (
    conditional_volterra_bridge_coefficients,
    refine_rbergomi_coarse_trunks,
    sample_conditional_volterra_fine_innovations,
    sample_rbergomi_coarse_trunks,
    simulate_branched_rbergomi_adjacent,
)
from src.path_integral.rbergomi_coupling import simulate_coupled_rbergomi_adjacent
from src.physics_engine import RBergomiSimulator


def _simulator() -> RBergomiSimulator:
    return RBergomiSimulator(H=0.1, eta=1.2, xi=0.04, rho=-0.7, device="cpu")


def _task() -> DownsideExcursionTask:
    return DownsideExcursionTask(
        hit_barrier=90.0,
        stress_level=95.0,
        minimum_occupation=0.05,
        hit_scale=3.0,
        occupation_scale=0.03,
    )


def _control() -> TimePiecewiseTwoDriverControl:
    return TimePiecewiseTwoDriverControl(((-0.5, 0.2), (-0.3, 0.1)), maturity=0.25)


def test_conditional_projection_annihilates_all_coarse_directions() -> None:
    simulator = _simulator()
    coefficients = conditional_volterra_bridge_coefficients(simulator, fine_dt=0.0125)
    identity = coefficients.coarse_projection @ coefficients.conditional_gain
    projected_covariance = coefficients.coarse_projection @ coefficients.conditional_covariance
    eigenvalues = torch.linalg.eigvalsh(coefficients.conditional_covariance)
    assert torch.allclose(identity, torch.eye(2, dtype=torch.float64), atol=3e-14)
    assert torch.max(torch.abs(projected_covariance)) <= 3e-14
    assert float(eigenvalues.min()) >= -3e-14
    assert int((eigenvalues > 1e-12).sum()) == 3
    coarse_chol, _weights, _variance, _drift = simulator._hybrid_coefficients(
        1, 0.025, H=simulator.H, dtype=torch.float64
    )
    assert torch.allclose(
        coefficients.coarse_covariance,
        coarse_chol @ coarse_chol.T,
        atol=3e-14,
        rtol=3e-14,
    )


def test_conditional_bridge_has_exact_constraint_mean_and_covariance() -> None:
    simulator = _simulator()
    coefficients = conditional_volterra_bridge_coefficients(simulator, fine_dt=0.0125)
    coarse = torch.tensor([[[0.08, -0.03]]], dtype=torch.float64)
    torch.manual_seed(8201)
    samples = sample_conditional_volterra_fine_innovations(coefficients, coarse, branches=40_000)[
        0, :, 0
    ]
    expected_mean = coarse[0, 0] @ coefficients.conditional_gain.T
    empirical_covariance = torch.cov(samples.T)
    reconstructed = samples @ coefficients.coarse_projection.T
    assert torch.max(torch.abs(reconstructed - coarse[0, 0])) <= 3e-14
    mean_standard_error = torch.sqrt(
        torch.diag(coefficients.conditional_covariance) / samples.shape[0]
    )
    mean_z = torch.abs(samples.mean(dim=0) - expected_mean) / torch.clamp(
        mean_standard_error, min=1e-14
    )
    assert torch.max(mean_z) < 4.0
    assert torch.allclose(
        empirical_covariance,
        coefficients.conditional_covariance,
        atol=4e-4,
        rtol=0.035,
    )


def test_branch_residuals_are_conditionally_independent() -> None:
    simulator = _simulator()
    coefficients = conditional_volterra_bridge_coefficients(simulator, fine_dt=0.0125)
    coarse = torch.zeros((30_000, 1, 2), dtype=torch.float64)
    torch.manual_seed(8202)
    samples = sample_conditional_volterra_fine_innovations(coefficients, coarse, branches=2)[
        :, :, 0
    ]
    cross = samples[:, 0].T @ samples[:, 1] / samples.shape[0]
    scale = torch.sqrt(
        torch.diag(coefficients.conditional_covariance)[:, None]
        * torch.diag(coefficients.conditional_covariance)[None, :]
    )
    standardized = torch.where(scale > 1e-14, cross / scale, torch.zeros_like(cross))
    assert torch.max(torch.abs(standardized)) < 0.025


def test_one_branch_matches_existing_adjacent_coupling_marginals() -> None:
    simulator = _simulator()
    paths = 15_000
    torch.manual_seed(8203)
    branched = simulate_branched_rbergomi_adjacent(
        simulator,
        S0=100.0,
        T=0.25,
        fine_steps=16,
        num_parents=paths,
        branches=1,
    )
    torch.manual_seed(8204)
    adjacent = simulate_coupled_rbergomi_adjacent(
        simulator, S0=100.0, T=0.25, fine_steps=16, num_paths=paths
    )

    def assert_mean_agreement(left: torch.Tensor, right: torch.Tensor) -> None:
        difference = float(left.mean() - right.mean())
        combined_se = math.sqrt(
            float(left.var(unbiased=True)) / paths + float(right.var(unbiased=True)) / paths
        )
        assert abs(difference) < 4.0 * combined_se + 3e-4

    assert_mean_agreement(branched.fine_spot[:, 0, -1], adjacent.fine.spot[:, -1])
    assert_mean_agreement(branched.trunks.spot[:, -1], adjacent.coarse.spot[:, -1])
    assert_mean_agreement(
        branched.fine_volterra[:, 0, -1].square(),
        adjacent.fine.volterra[:, -1].square(),
    )
    assert branched.conditional_constraint_error <= 3e-14


def test_deterministic_control_likelihood_and_aggregation_are_exact() -> None:
    simulator = _simulator()
    torch.manual_seed(8205)
    result = simulate_branched_rbergomi_adjacent(
        simulator,
        S0=100.0,
        T=0.25,
        fine_steps=16,
        num_parents=128,
        branches=4,
        control=_control(),
        record_augmented=True,
    )
    assert result.proposal_fine_brownian_increments is not None
    assert result.target_fine_brownian_increments is not None
    schedule = result.trunks.fine_control_schedule
    controls = schedule[None, None].expand_as(result.proposal_fine_brownian_increments)
    expected_likelihood = brownian_log_likelihood(
        controls,
        result.proposal_fine_brownian_increments,
        result.trunks.fine_dt,
    )
    fine_pair_sum = result.target_fine_brownian_increments.reshape(128, 4, 8, 2, 2).sum(dim=3)
    assert torch.allclose(result.log_likelihood, expected_likelihood, atol=3e-14)
    assert torch.allclose(
        fine_pair_sum,
        result.trunks.target_brownian_increments[:, None],
        atol=3e-14,
        rtol=0.0,
    )
    assert result.conditional_constraint_error <= 3e-14


def test_controlled_branched_hard_correction_is_unbiased() -> None:
    simulator = _simulator()
    task = _task()
    paths = 25_000
    torch.manual_seed(8206)
    natural = simulate_coupled_rbergomi_adjacent(
        simulator, S0=100.0, T=0.25, fine_steps=16, num_paths=paths
    )
    natural_fine = task.hard_event(natural.fine.spot, natural.fine.step_dt).double()
    natural_coarse = task.hard_event(natural.coarse.spot, natural.coarse.step_dt).double()
    natural_contribution = natural_fine - natural_coarse

    torch.manual_seed(8207)
    branched = simulate_branched_rbergomi_adjacent(
        simulator,
        S0=100.0,
        T=0.25,
        fine_steps=16,
        num_parents=paths,
        branches=4,
        control=_control(),
    )
    fine_event = (
        task.hard_event(branched.fine_spot.reshape(-1, 17), branched.trunks.fine_dt)
        .double()
        .reshape(paths, 4)
    )
    coarse_event = task.hard_event(branched.trunks.spot, 2.0 * branched.trunks.fine_dt).double()
    controlled_parent = torch.mean(
        (fine_event - coarse_event[:, None]) * torch.exp(branched.log_likelihood),
        dim=1,
    )
    difference = float(controlled_parent.mean() - natural_contribution.mean())
    combined_se = math.sqrt(
        float(controlled_parent.var(unbiased=True)) / paths
        + float(natural_contribution.var(unbiased=True)) / paths
    )
    assert abs(difference) < 4.0 * combined_se + 5e-4
    likelihood_parent = torch.exp(branched.log_likelihood).mean(dim=1)
    likelihood_se = float(likelihood_parent.std(unbiased=True)) / math.sqrt(paths)
    assert abs(float(likelihood_parent.mean()) - 1.0) < 4.0 * likelihood_se + 5e-4


def test_branch_averaging_obeys_conditional_variance_decomposition() -> None:
    simulator = _simulator()
    parents = 12_000
    branches = 8
    torch.manual_seed(8208)
    result = simulate_branched_rbergomi_adjacent(
        simulator,
        S0=100.0,
        T=0.25,
        fine_steps=16,
        num_parents=parents,
        branches=branches,
    )
    fine_payoff = torch.sigmoid((90.0 - result.fine_spot[:, :, -1]) / 5.0)
    coarse_payoff = torch.sigmoid((90.0 - result.trunks.spot[:, -1]) / 5.0)
    branch_contribution = fine_payoff - coarse_payoff[:, None]
    within = float(branch_contribution.var(dim=1, unbiased=True).mean())
    full_parent = branch_contribution.mean(dim=1)
    irreducible = max(float(full_parent.var(unbiased=True)) - within / branches, 0.0)
    observed: list[float] = []
    predicted: list[float] = []
    for count in (1, 2, 4, 8):
        observed.append(float(branch_contribution[:, :count].mean(dim=1).var(unbiased=True)))
        predicted.append(irreducible + within / count)
    assert observed == sorted(observed, reverse=True)
    for actual, target in zip(observed, predicted, strict=True):
        assert actual == pytest.approx(target, rel=0.08, abs=2e-6)


def test_boundary_policy_calibration_matches_its_declared_work_objective() -> None:
    simulator = _simulator()
    task = _task()
    torch.manual_seed(8209)
    trunks = sample_rbergomi_coarse_trunks(
        simulator,
        S0=100.0,
        T=0.25,
        fine_steps=16,
        num_parents=5_000,
        control=_control(),
    )
    refined = refine_rbergomi_coarse_trunks(simulator, trunks, branches=4)
    fine = (
        task.hard_event(refined.fine_spot.reshape(-1, 17), trunks.fine_dt)
        .double()
        .reshape(5_000, 4)
    )
    coarse = task.hard_event(trunks.spot, 2.0 * trunks.fine_dt).double()
    branch_values = (fine - coarse[:, None]) * torch.exp(refined.log_likelihood)
    calibration = calibrate_boundary_branching_policy(
        trunks,
        task,
        branch_values,
        hit_bands=(0.5, 1.0, 2.0),
        occupation_bands=(0.5, 1.0, 2.0),
        high_branch_candidates=(2, 4),
    )
    counts = boundary_branch_counts(trunks, task, calibration.policy)
    selected = counts > 1
    reproduced = branch_values[:, 0].clone()
    reproduced[selected] = branch_values[selected, : calibration.policy.high_branches].mean(dim=1)
    expected_work = float(reproduced.var(unbiased=True)) * (
        8.0 + 16.0 * float(counts.double().mean())
    )
    assert calibration.selected_work_proxy == pytest.approx(expected_work, rel=1e-14)
    assert calibration.mean_branches == pytest.approx(float(counts.double().mean()))


def test_coarse_measurable_adaptive_branching_remains_unbiased() -> None:
    simulator = _simulator()
    task = _task()
    parents = 20_000
    torch.manual_seed(8210)
    natural = simulate_coupled_rbergomi_adjacent(
        simulator, S0=100.0, T=0.25, fine_steps=16, num_paths=parents
    )
    natural_value = (
        task.hard_event(natural.fine.spot, natural.fine.step_dt).double()
        - task.hard_event(natural.coarse.spot, natural.coarse.step_dt).double()
    )

    torch.manual_seed(8211)
    trunks = sample_rbergomi_coarse_trunks(
        simulator,
        S0=100.0,
        T=0.25,
        fine_steps=16,
        num_parents=parents,
        control=_control(),
    )
    result = evaluate_adaptive_branched_correction(
        simulator,
        trunks,
        task,
        BoundaryBranchingPolicy(hit_band=1.5, occupation_band=1.5, high_branches=4),
    )
    difference = float(result.contributions.mean() - natural_value.mean())
    combined_se = math.sqrt(
        float(result.contributions.var(unbiased=True)) / parents
        + float(natural_value.var(unbiased=True)) / parents
    )
    assert abs(difference) < 4.0 * combined_se + 5e-4
    likelihood_se = float(result.likelihood_parent_means.std(unbiased=True)) / math.sqrt(parents)
    assert abs(float(result.likelihood_parent_means.mean()) - 1.0) < (4.0 * likelihood_se + 5e-4)
    assert set(result.branch_counts.tolist()) <= {1, 4}
    assert result.maximum_constraint_error <= 3e-14
