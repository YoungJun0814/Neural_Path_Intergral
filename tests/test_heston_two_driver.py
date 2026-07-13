"""Correctness gates for independent-basis two-driver Heston control."""

from __future__ import annotations

import math

import pytest
import torch

from src.path_integral import brownian_log_likelihood
from src.physics_engine import MarketSimulator, TwoDriverHestonPaths


@pytest.fixture()
def simulator() -> MarketSimulator:
    return MarketSimulator(
        mu=0.03,
        kappa=1.8,
        theta=0.04,
        xi=0.45,
        rho=-0.65,
        device="cpu",
    )


def _constant_two_driver(control_1: float, control_2: float):
    def control(
        _time: float,
        spot: torch.Tensor,
        _variance: torch.Tensor,
        _average: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack(
            (torch.full_like(spot, control_1), torch.full_like(spot, control_2)), dim=-1
        )

    return control


def test_two_driver_records_exact_coordinates_and_generic_likelihood(
    simulator: MarketSimulator,
) -> None:
    torch.manual_seed(101)
    result = simulator.simulate_controlled_two_driver(
        S0=100.0,
        v0=0.04,
        T=0.4,
        dt=0.06,
        num_paths=128,
        control_fn=_constant_two_driver(-0.55, 0.3),
        record_brownian=True,
        dtype=torch.float64,
    )

    assert isinstance(result, TwoDriverHestonPaths)
    assert result.proposal_brownian_increments is not None
    assert result.target_brownian_increments is not None
    assert result.controls is not None
    assert result.proposal_brownian_increments.shape == (128, 7, 2)
    expected_target = (
        result.proposal_brownian_increments + result.controls * result.step_dt
    )
    expected_log_likelihood = brownian_log_likelihood(
        result.controls, result.proposal_brownian_increments, result.step_dt
    )
    expected_energy = result.step_dt * torch.sum(result.controls.square(), dim=(1, 2))

    assert torch.equal(result.target_brownian_increments, expected_target)
    assert torch.allclose(result.log_likelihood, expected_log_likelihood, atol=1e-15, rtol=1e-15)
    assert torch.allclose(result.control_energy, expected_energy, atol=1e-15, rtol=1e-15)
    assert result.log_likelihood.dtype == torch.float64


def test_target_brownian_path_reconstructs_controlled_heston_path(
    simulator: MarketSimulator,
) -> None:
    def feedback(
        _time: float,
        spot: torch.Tensor,
        variance: torch.Tensor,
        _average: torch.Tensor,
    ) -> torch.Tensor:
        control_1 = -0.5 + 0.08 * torch.log(spot / 100.0)
        control_2 = 0.25 + 0.15 * (variance / 0.04 - 1.0)
        return torch.stack((control_1, control_2), dim=-1)

    torch.manual_seed(202)
    result = simulator.simulate_controlled_two_driver(
        S0=100.0,
        v0=0.04,
        T=0.3,
        dt=0.04,
        num_paths=96,
        control_fn=feedback,
        record_brownian=True,
        dtype=torch.float64,
    )
    assert result.target_brownian_increments is not None

    target = result.target_brownian_increments
    spot = torch.full((target.shape[0],), 100.0, dtype=torch.float64)
    variance_state = torch.full((target.shape[0],), 0.04, dtype=torch.float64)
    reconstructed_spot = [spot]
    reconstructed_variance = [variance_state]
    rho = simulator.rho
    rho_perp = math.sqrt(1.0 - rho * rho)

    for step in range(target.shape[1]):
        variance_plus = torch.clamp(variance_state, min=0.0)
        sqrt_variance = torch.sqrt(variance_plus)
        next_spot = spot * torch.exp(
            (simulator.mu - 0.5 * variance_plus) * result.step_dt
            + sqrt_variance * target[:, step, 0]
        )
        next_variance_state = (
            variance_state
            + simulator.kappa * (simulator.theta - variance_plus) * result.step_dt
            + simulator.xi
            * sqrt_variance
            * (rho * target[:, step, 0] + rho_perp * target[:, step, 1])
        )
        spot = next_spot
        variance_state = next_variance_state
        reconstructed_spot.append(spot)
        reconstructed_variance.append(torch.clamp(variance_state, min=0.0))

    assert torch.allclose(
        result.spot, torch.stack(reconstructed_spot, dim=1), atol=2e-12, rtol=2e-12
    )
    assert torch.allclose(
        result.variance,
        torch.stack(reconstructed_variance, dim=1),
        atol=2e-12,
        rtol=2e-12,
    )


def test_zero_second_control_matches_legacy_one_driver_simulator(
    simulator: MarketSimulator,
) -> None:
    def scalar_feedback(
        _time: float,
        spot: torch.Tensor,
        variance: torch.Tensor,
        _average: torch.Tensor,
    ) -> torch.Tensor:
        return -0.35 + 0.05 * torch.log(spot / 100.0) + 0.02 * (variance / 0.04 - 1.0)

    def vector_feedback(
        time: float,
        spot: torch.Tensor,
        variance: torch.Tensor,
        average: torch.Tensor,
    ) -> torch.Tensor:
        control_1 = scalar_feedback(time, spot, variance, average)
        return torch.stack((control_1, torch.zeros_like(control_1)), dim=-1)

    observed_legacy: list[torch.Tensor] = []
    torch.manual_seed(303)
    legacy_spot, legacy_variance, legacy_log_likelihood, _barrier, legacy_integral = (
        simulator.simulate_controlled(
            S0=100.0,
            v0=0.04,
            T=0.25,
            dt=0.03,
            num_paths=256,
            control_fn=scalar_feedback,
            brownian_observer=lambda _time, increment: observed_legacy.append(
                increment.detach()
            ),
        )
    )

    torch.manual_seed(303)
    two_driver = simulator.simulate_controlled_two_driver(
        S0=100.0,
        v0=0.04,
        T=0.25,
        dt=0.03,
        num_paths=256,
        control_fn=vector_feedback,
        record_brownian=True,
        dtype=torch.float32,
    )
    assert two_driver.proposal_brownian_increments is not None

    assert torch.allclose(two_driver.spot, legacy_spot, atol=2e-6, rtol=2e-6)
    assert torch.allclose(two_driver.variance, legacy_variance, atol=2e-6, rtol=2e-6)
    assert torch.allclose(
        two_driver.log_likelihood.float(), legacy_log_likelihood, atol=2e-6, rtol=2e-6
    )
    assert torch.allclose(
        two_driver.running_spot_integral, legacy_integral, atol=2e-6, rtol=2e-6
    )
    assert torch.equal(
        two_driver.proposal_brownian_increments[:, :, 0],
        torch.stack(observed_legacy, dim=1),
    )


def test_two_driver_likelihood_normalizes_without_recording_paths(
    simulator: MarketSimulator,
) -> None:
    torch.manual_seed(404)
    control_1 = -0.5
    control_2 = 0.35
    horizon = 0.6
    result = simulator.simulate_controlled_two_driver(
        S0=100.0,
        v0=0.04,
        T=horizon,
        dt=0.15,
        num_paths=120_000,
        control_fn=_constant_two_driver(control_1, control_2),
        record_brownian=False,
        dtype=torch.float64,
    )

    assert result.proposal_brownian_increments is None
    assert result.target_brownian_increments is None
    assert result.controls is None
    assert torch.exp(result.log_likelihood).mean() == pytest.approx(1.0, abs=0.006)
    assert result.control_energy.mean() == pytest.approx(
        (control_1**2 + control_2**2) * horizon, abs=1e-14
    )


def test_two_driver_fixed_control_heston_estimator_is_unbiased(
    simulator: MarketSimulator,
) -> None:
    paths = 30_000
    torch.manual_seed(505)
    natural_spot, _natural_variance = simulator.simulate(
        S0=100.0,
        v0=0.04,
        T=0.4,
        dt=1.0 / 64.0,
        num_paths=paths,
    )
    natural_payoff = torch.clamp(natural_spot[:, -1] - 100.0, min=0.0).double()

    torch.manual_seed(506)
    controlled = simulator.simulate_controlled_two_driver(
        S0=100.0,
        v0=0.04,
        T=0.4,
        dt=1.0 / 64.0,
        num_paths=paths,
        control_fn=_constant_two_driver(0.35, -0.3),
        record_brownian=False,
        dtype=torch.float32,
    )
    controlled_payoff = torch.clamp(controlled.spot[:, -1] - 100.0, min=0.0).double()
    contributions = controlled_payoff * torch.exp(controlled.log_likelihood)
    difference = float(contributions.mean() - natural_payoff.mean())
    combined_se = math.sqrt(
        float(contributions.var(unbiased=True)) / paths
        + float(natural_payoff.var(unbiased=True)) / paths
    )

    assert abs(difference) < 4.0 * combined_se + 0.03


def test_two_driver_rejects_wrong_control_shape(simulator: MarketSimulator) -> None:
    def invalid_control(
        _time: float,
        spot: torch.Tensor,
        _variance: torch.Tensor,
        _average: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros_like(spot)

    with pytest.raises(ValueError, match="shape"):
        simulator.simulate_controlled_two_driver(
            S0=100.0,
            v0=0.04,
            T=0.1,
            dt=0.05,
            num_paths=8,
            control_fn=invalid_control,
        )
