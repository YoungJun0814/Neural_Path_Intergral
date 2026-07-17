"""Law, likelihood, and causality tests for adjacent-grid BLP coupling."""

from __future__ import annotations

import math

import pytest
import torch

from src.path_integral import brownian_log_likelihood
from src.path_integral.rbergomi_coupling import (
    adjacent_local_gaussian_coefficients,
    simulate_coupled_rbergomi_adjacent,
)
from src.path_integral.rbergomi_multilevel import simulate_coupled_rbergomi_mixture
from src.physics_engine import RBergomiSimulator


def _simulator() -> RBergomiSimulator:
    return RBergomiSimulator(H=0.1, eta=1.2, xi=0.04, rho=-0.7, device="cpu")


class _StateFeedback:
    def __call__(
        self,
        time: float,
        spot: torch.Tensor,
        variance: torch.Tensor,
        volterra: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack(
            (
                -0.35 + 0.002 * (spot - 100.0) + 0.1 * time,
                0.20 + 0.03 * (variance / 0.04 - 1.0) + 0.01 * volterra,
            ),
            dim=-1,
        )


def test_adjacent_first_cell_covariance_is_exact() -> None:
    simulator = _simulator()
    h = 0.0125
    coefficients = adjacent_local_gaussian_coefficients(simulator, fine_dt=h)
    covariance = coefficients.first_cell_cholesky @ coefficients.first_cell_cholesky.T
    alpha = simulator.H - 0.5
    expected_01 = h ** (alpha + 1.0) / (alpha + 1.0)
    expected_02 = ((2.0 * h) ** (alpha + 1.0) - h ** (alpha + 1.0)) / (alpha + 1.0)
    expected_22 = ((2.0 * h) ** (2.0 * alpha + 1.0) - h ** (2.0 * alpha + 1.0)) / (
        2.0 * alpha + 1.0
    )
    assert float(covariance[0, 0]) == pytest.approx(h, abs=2e-15)
    assert float(covariance[0, 1]) == pytest.approx(expected_01, abs=2e-15)
    assert float(covariance[0, 2]) == pytest.approx(expected_02, abs=2e-15)
    assert float(covariance[2, 2]) == pytest.approx(expected_22, abs=2e-15)


def test_null_coupling_realizes_both_blp_marginal_covariances() -> None:
    simulator = _simulator()
    torch.manual_seed(7201)
    result = simulate_coupled_rbergomi_adjacent(
        simulator,
        S0=100.0,
        T=0.25,
        fine_steps=16,
        num_paths=30_000,
        record_augmented=True,
    )
    assert result.target_fine_brownian_increments is not None
    assert result.target_fine_local_integrals is not None
    assert result.coarse.target_brownian_increments is not None
    assert result.target_coarse_local_integrals is not None
    fine_pair = torch.stack(
        (
            result.target_fine_brownian_increments[:, 0, 0],
            result.target_fine_local_integrals[:, 0],
        ),
        dim=1,
    )
    coarse_pair = torch.stack(
        (
            result.coarse.target_brownian_increments[:, 0, 0],
            result.target_coarse_local_integrals[:, 0],
        ),
        dim=1,
    )
    fine_empirical = torch.cov(fine_pair.T)
    coarse_empirical = torch.cov(coarse_pair.T)
    fine_chol, _fw, _fv, _fd = simulator._hybrid_coefficients(
        16, result.fine.step_dt, H=simulator.H, dtype=torch.float64
    )
    coarse_chol, _cw, _cv, _cd = simulator._hybrid_coefficients(
        8, result.coarse.step_dt, H=simulator.H, dtype=torch.float64
    )
    assert torch.allclose(fine_empirical, fine_chol @ fine_chol.T, rtol=0.035, atol=3e-4)
    assert torch.allclose(coarse_empirical, coarse_chol @ coarse_chol.T, rtol=0.035, atol=3e-4)
    assert torch.equal(result.log_likelihood, torch.zeros_like(result.log_likelihood))


def test_coupled_terminal_marginals_match_standalone_blp_simulators() -> None:
    simulator = _simulator()
    paths = 15_000
    torch.manual_seed(7210)
    coupled = simulate_coupled_rbergomi_adjacent(
        simulator, S0=100.0, T=0.25, fine_steps=16, num_paths=paths
    )
    torch.manual_seed(7211)
    fine = simulator.simulate_controlled_two_driver(
        S0=100.0, T=0.25, dt=0.25 / 16, num_paths=paths, control_fn=None
    )
    torch.manual_seed(7212)
    coarse = simulator.simulate_controlled_two_driver(
        S0=100.0, T=0.25, dt=0.25 / 8, num_paths=paths, control_fn=None
    )

    def assert_mean_agreement(left: torch.Tensor, right: torch.Tensor) -> None:
        difference = float(left.mean() - right.mean())
        combined_se = math.sqrt(
            float(left.var(unbiased=True)) / paths + float(right.var(unbiased=True)) / paths
        )
        assert abs(difference) < 4.0 * combined_se + 2e-4

    assert_mean_agreement(coupled.fine.spot[:, -1], fine.spot[:, -1])
    assert_mean_agreement(coupled.coarse.spot[:, -1], coarse.spot[:, -1])
    assert_mean_agreement(coupled.fine.variance[:, -1], fine.variance[:, -1])
    assert_mean_agreement(coupled.coarse.variance[:, -1], coarse.variance[:, -1])
    assert_mean_agreement(coupled.fine.volterra[:, -1].square(), fine.volterra[:, -1].square())
    assert_mean_agreement(coupled.coarse.volterra[:, -1].square(), coarse.volterra[:, -1].square())


def test_control_shifts_every_augmented_integral_and_likelihood_once() -> None:
    simulator = _simulator()
    torch.manual_seed(7202)
    result = simulate_coupled_rbergomi_adjacent(
        simulator,
        S0=100.0,
        T=0.2,
        fine_steps=8,
        num_paths=64,
        control_fn=_StateFeedback(),
        record_augmented=True,
    )
    assert result.proposal_fine_brownian_increments is not None
    assert result.target_fine_brownian_increments is not None
    assert result.proposal_fine_local_integrals is not None
    assert result.target_fine_local_integrals is not None
    assert result.proposal_coarse_local_integrals is not None
    assert result.target_coarse_local_integrals is not None
    assert result.fine_controls is not None
    coefficients = adjacent_local_gaussian_coefficients(simulator, fine_dt=result.fine.step_dt)
    expected_target_brownian = (
        result.proposal_fine_brownian_increments + result.fine_controls * result.fine.step_dt
    )
    expected_fine_local = (
        result.proposal_fine_local_integrals
        + result.fine_controls[:, :, 0] * coefficients.fine_drift_integral
    )
    first = result.fine_controls[:, 0::2, 0]
    second = result.fine_controls[:, 1::2, 0]
    expected_coarse_shift = (
        first * coefficients.coarse_first_drift_integral + second * coefficients.fine_drift_integral
    )
    assert torch.equal(result.target_fine_brownian_increments, expected_target_brownian)
    assert torch.allclose(
        result.target_fine_local_integrals, expected_fine_local, atol=2e-14, rtol=0.0
    )
    assert torch.allclose(
        result.target_coarse_local_integrals,
        result.proposal_coarse_local_integrals + expected_coarse_shift,
        atol=2e-14,
        rtol=0.0,
    )
    expected_log_likelihood = brownian_log_likelihood(
        result.fine_controls,
        result.proposal_fine_brownian_increments,
        result.fine.step_dt,
    )
    assert torch.allclose(result.log_likelihood, expected_log_likelihood, atol=2e-14)


def test_feedback_is_evaluated_on_the_preincrement_fine_state() -> None:
    simulator = _simulator()
    control = _StateFeedback()
    torch.manual_seed(7203)
    result = simulate_coupled_rbergomi_adjacent(
        simulator,
        S0=100.0,
        T=0.25,
        fine_steps=8,
        num_paths=32,
        control_fn=control,
        record_augmented=True,
    )
    assert result.fine_controls is not None
    expected = torch.stack(
        [
            control(
                step * result.fine.step_dt,
                result.fine.spot[:, step],
                result.fine.variance[:, step],
                result.fine.volterra[:, step],
            )
            for step in range(8)
        ],
        dim=1,
    )
    assert torch.equal(result.fine_controls, expected)


def test_controlled_correction_is_unbiased_for_a_bounded_payoff() -> None:
    simulator = _simulator()
    paths = 35_000
    torch.manual_seed(7204)
    natural = simulate_coupled_rbergomi_adjacent(
        simulator, S0=100.0, T=0.25, fine_steps=16, num_paths=paths
    )
    torch.manual_seed(7205)
    proposal = simulate_coupled_rbergomi_adjacent(
        simulator,
        S0=100.0,
        T=0.25,
        fine_steps=16,
        num_paths=paths,
        control_fn=_StateFeedback(),
    )
    natural_correction = torch.sigmoid((90.0 - natural.fine.spot[:, -1]) / 5.0) - torch.sigmoid(
        (90.0 - natural.coarse.spot[:, -1]) / 5.0
    )
    proposal_correction = torch.sigmoid((90.0 - proposal.fine.spot[:, -1]) / 5.0) - torch.sigmoid(
        (90.0 - proposal.coarse.spot[:, -1]) / 5.0
    )
    contribution = proposal_correction * torch.exp(proposal.log_likelihood)
    difference = float(contribution.mean() - natural_correction.mean())
    combined_se = math.sqrt(
        float(contribution.var(unbiased=True)) / paths
        + float(natural_correction.var(unbiased=True)) / paths
    )
    assert abs(difference) < 4.0 * combined_se + 5e-4


def test_single_component_coupled_mixture_matches_direct_sample_pathwise() -> None:
    simulator = _simulator()
    control = _StateFeedback()
    torch.manual_seed(7206)
    direct = simulate_coupled_rbergomi_adjacent(
        simulator,
        S0=100.0,
        T=0.25,
        fine_steps=8,
        num_paths=128,
        control_fn=control,
        record_augmented=True,
    )
    torch.manual_seed(7206)
    mixture = simulate_coupled_rbergomi_mixture(
        simulator,
        [control],
        torch.ones(1, dtype=torch.float64),
        spot=100.0,
        maturity=0.25,
        fine_steps=8,
        num_paths=128,
        label_generator=torch.Generator().manual_seed(55),
    )
    assert torch.equal(mixture.paths.fine.spot, direct.fine.spot)
    assert torch.equal(mixture.paths.coarse.spot, direct.coarse.spot)
    assert torch.allclose(
        mixture.mixture_log_likelihood, direct.log_likelihood, atol=2e-14, rtol=0.0
    )
    assert mixture.maximum_selected_replay_error <= 2e-14


def test_defensive_coupled_mixture_likelihood_normalizes() -> None:
    simulator = _simulator()

    class OppositeFeedback(_StateFeedback):
        def __call__(
            self,
            time: float,
            spot: torch.Tensor,
            variance: torch.Tensor,
            volterra: torch.Tensor,
        ) -> torch.Tensor:
            return -super().__call__(time, spot, variance, volterra)

    torch.manual_seed(7207)
    sample = simulate_coupled_rbergomi_mixture(
        simulator,
        [_StateFeedback(), OppositeFeedback()],
        torch.tensor([0.8, 0.2], dtype=torch.float64),
        spot=100.0,
        maturity=0.125,
        fine_steps=8,
        num_paths=30_000,
        label_generator=torch.Generator().manual_seed(7208),
    )
    likelihood = torch.exp(sample.mixture_log_likelihood)
    standard_error = float(likelihood.std(unbiased=True)) / math.sqrt(likelihood.numel())
    assert abs(float(likelihood.mean()) - 1.0) < 4.0 * standard_error + 5e-4
    assert sample.maximum_selected_replay_error <= 3e-13
