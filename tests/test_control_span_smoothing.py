"""Theory-contract tests for defensive control-span marginalization."""

from __future__ import annotations

import math

import pytest
import torch

from src.path_integral import (
    DownsideExcursionTask,
    TimePiecewiseTwoDriverControl,
    evaluate_control_span_marginalized_adjacent_mixture,
    evaluate_control_span_marginalized_mixture,
    evaluate_rank_two_control_span_marginalized_mixture,
    positive_rank_two_subspace,
    rank_one_price_control_span,
    simulate_coupled_rbergomi_mixture,
    simulate_rbergomi_fft,
    simulate_rbergomi_mixture,
)
from src.physics_engine import RBergomiSimulator


def _simulator() -> RBergomiSimulator:
    return RBergomiSimulator(H=0.1, eta=1.2, xi=0.04, rho=-0.55, device="cpu")


def _controls() -> tuple[TimePiecewiseTwoDriverControl, TimePiecewiseTwoDriverControl]:
    natural = TimePiecewiseTwoDriverControl(((0.0, 0.0), (0.0, 0.0)), maturity=0.25)
    tilted = TimePiecewiseTwoDriverControl(
        ((-0.35, -1.1), (-0.20, -0.7)),
        maturity=0.25,
    )
    return natural, tilted


def _task() -> DownsideExcursionTask:
    return DownsideExcursionTask(
        hit_barrier=92.0,
        stress_level=97.0,
        minimum_occupation=1.0 / 64.0,
        hit_scale=4.0,
        occupation_scale=0.02,
    )


def test_rank_one_span_is_positive_and_reconstructs_all_price_shifts() -> None:
    schedules = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[-0.2, -1.0], [-0.2, -0.8], [-0.1, -0.5], [-0.1, -0.2]],
        ],
        dtype=torch.float64,
    )
    span = rank_one_price_control_span(schedules, step_dt=0.025)
    assert torch.all(span.direction > 0.0)
    assert float(torch.linalg.vector_norm(span.direction)) == pytest.approx(1.0, abs=2e-15)
    assert span.maximum_span_residual <= 2e-16
    assert float(span.expert_coefficients[0]) == 0.0
    assert float(span.expert_coefficients[1]) < 0.0


def test_rank_one_span_rejects_mixed_sign_and_noncollinear_controls() -> None:
    mixed = torch.zeros((2, 4, 2), dtype=torch.float64)
    mixed[1, :, 1] = torch.tensor([-1.0, -0.5, 0.5, 1.0])
    with pytest.raises(ValueError, match="strictly one-signed"):
        rank_one_price_control_span(mixed, step_dt=0.1)

    noncollinear = torch.zeros((3, 4, 2), dtype=torch.float64)
    noncollinear[1, :, 1] = torch.tensor([-1.0, -0.8, -0.5, -0.2])
    noncollinear[2, :, 1] = torch.tensor([-0.2, -0.5, -0.8, -1.0])
    with pytest.raises(ValueError, match="common span"):
        rank_one_price_control_span(noncollinear, step_dt=0.1)


def test_positive_rank_two_oblique_coordinates_have_declared_covariance() -> None:
    control = torch.tensor([0.6, 0.5, 0.4, 0.3], dtype=torch.float64)
    control = control / torch.linalg.vector_norm(control)
    subspace = positive_rank_two_subspace(control)
    assert torch.all(subspace.basis > 0.0)
    assert subspace.minimum_gram_eigenvalue > 1e-3
    assert torch.allclose(
        subspace.coordinate_covariance @ subspace.gram,
        torch.eye(2, dtype=torch.float64),
        atol=2e-14,
        rtol=0.0,
    )


def test_fft_mixture_single_component_matches_direct_fft_pathwise() -> None:
    simulator = _simulator()
    _natural, tilted = _controls()
    torch.manual_seed(9811)
    direct = simulate_rbergomi_fft(
        simulator,
        S0=100.0,
        T=0.25,
        dt=1.0 / 64.0,
        num_paths=257,
        control_fn=tilted,
    )
    torch.manual_seed(9811)
    mixture = simulate_rbergomi_mixture(
        simulator,
        [tilted],
        torch.ones(1, dtype=torch.float64),
        spot=100.0,
        maturity=0.25,
        dt=1.0 / 64.0,
        num_paths=257,
        label_generator=torch.Generator().manual_seed(9812),
        engine="fft",
    )
    assert torch.equal(mixture.paths.spot, direct.spot)
    assert torch.equal(mixture.paths.variance, direct.variance)
    assert torch.allclose(
        mixture.mixture_log_likelihood,
        direct.log_likelihood,
        atol=3e-14,
        rtol=0.0,
    )


def test_defensive_control_span_estimator_replays_and_is_mean_consistent() -> None:
    simulator = _simulator()
    controls = _controls()
    alpha = 0.2
    torch.manual_seed(9901)
    sample = simulate_rbergomi_mixture(
        simulator,
        controls,
        torch.tensor([alpha, 1.0 - alpha], dtype=torch.float64),
        spot=100.0,
        maturity=0.25,
        dt=1.0 / 64.0,
        num_paths=20_000,
        label_generator=torch.Generator().manual_seed(9902),
        engine="fft",
    )
    estimate = evaluate_control_span_marginalized_mixture(
        sample,
        task=_task(),
        rho=simulator.rho,
    )
    assert torch.equal(estimate.hard_event, estimate.threshold_event)
    assert estimate.maximum_component_log_density_error <= 2e-13
    assert estimate.maximum_mixture_log_density_error <= 2e-13
    assert estimate.maximum_path_reconstruction_error <= 2e-13
    assert estimate.maximum_residual_projection <= 2e-13
    assert estimate.maximum_defensive_bound_violation <= 2e-13
    assert estimate.defensive_weight == pytest.approx(alpha, abs=1e-15)
    assert float(torch.max(estimate.outer_likelihood)) <= 1.0 / alpha + 2e-13

    difference = estimate.marginalized_contribution - estimate.raw_mixture_contribution
    paired_se = math.sqrt(float(difference.var(unbiased=True)) / difference.numel())
    assert abs(float(difference.mean())) <= 4.0 * paired_se
    assert float(estimate.marginalized_contribution.var(unbiased=True)) < float(
        estimate.raw_mixture_contribution.var(unbiased=True)
    )

    outer_se = float(estimate.outer_likelihood.std(unbiased=True)) / math.sqrt(
        estimate.outer_likelihood.numel()
    )
    assert abs(float(estimate.outer_likelihood.mean()) - 1.0) <= 4.0 * outer_se


def test_adjacent_control_span_correction_is_exact_and_mean_consistent() -> None:
    simulator = _simulator()
    controls = _controls()
    alpha = 0.2
    torch.manual_seed(9951)
    sample = simulate_coupled_rbergomi_mixture(
        simulator,
        controls,
        torch.tensor([alpha, 1.0 - alpha], dtype=torch.float64),
        spot=100.0,
        maturity=0.25,
        fine_steps=32,
        num_paths=20_000,
        label_generator=torch.Generator().manual_seed(9952),
        engine="fft",
    )
    estimate = evaluate_control_span_marginalized_adjacent_mixture(
        sample,
        task=_task(),
        rho=simulator.rho,
    )
    assert estimate.maximum_component_log_density_error <= 2e-13
    assert estimate.maximum_mixture_log_density_error <= 2e-13
    assert estimate.maximum_coordinate_mismatch <= 2e-15
    assert estimate.maximum_fine_path_reconstruction_error <= 2e-13
    assert estimate.maximum_coarse_path_reconstruction_error <= 2e-13
    assert estimate.maximum_residual_projection <= 2e-13
    assert estimate.maximum_defensive_bound_violation <= 2e-13
    assert float(torch.max(estimate.outer_likelihood)) <= 1.0 / alpha + 2e-13

    difference = estimate.marginalized_correction - estimate.raw_correction
    paired_se = math.sqrt(float(difference.var(unbiased=True)) / difference.numel())
    assert abs(float(difference.mean())) <= 4.0 * paired_se
    assert float(estimate.marginalized_correction.var(unbiased=True)) < float(
        estimate.raw_correction.var(unbiased=True)
    )


def test_rank_two_nested_estimator_is_unbiased_and_preserves_defensive_bound() -> None:
    simulator = _simulator()
    controls = _controls()
    alpha = 0.2
    torch.manual_seed(9971)
    sample = simulate_rbergomi_mixture(
        simulator,
        controls,
        torch.tensor([alpha, 1.0 - alpha], dtype=torch.float64),
        spot=100.0,
        maturity=0.25,
        dt=1.0 / 64.0,
        num_paths=12_000,
        label_generator=torch.Generator().manual_seed(9972),
        engine="fft",
    )
    estimate = evaluate_rank_two_control_span_marginalized_mixture(
        sample,
        task=_task(),
        rho=simulator.rho,
        inner_samples=4,
        inner_generator=torch.Generator().manual_seed(9973),
    )
    assert torch.all(estimate.conditional_event_probability >= 0.0)
    assert torch.all(estimate.conditional_event_probability <= 1.0)
    assert estimate.maximum_component_log_density_error <= 2e-13
    assert estimate.maximum_mixture_log_density_error <= 2e-13
    assert estimate.maximum_path_reconstruction_error <= 2e-13
    assert estimate.maximum_residual_projection <= 2e-13
    assert estimate.maximum_defensive_bound_violation <= 2e-13
    assert float(torch.max(estimate.outer_likelihood)) <= 1.0 / alpha + 2e-13
    difference = estimate.marginalized_contribution - estimate.raw_mixture_contribution
    paired_se = math.sqrt(float(difference.var(unbiased=True)) / difference.numel())
    assert abs(float(difference.mean())) <= 4.0 * paired_se
