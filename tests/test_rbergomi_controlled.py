"""G2 law tests for the independent-basis controlled rBergomi BLP scheme."""

from __future__ import annotations

import math

import pytest
import torch

from src.path_integral import brownian_log_likelihood
from src.physics_engine import RBergomiSimulator, TwoDriverRBergomiPaths


def _simulator(*, rho: float = -0.7) -> RBergomiSimulator:
    return RBergomiSimulator(H=0.1, eta=1.2, xi=0.04, rho=rho, device="cpu")


def _constant_control(first: float, second: float):
    def control(
        _time: float,
        spot: torch.Tensor,
        _variance: torch.Tensor,
        _volterra: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack(
            (torch.full_like(spot, first), torch.full_like(spot, second)), dim=-1
        )

    return control


def test_null_control_matches_natural_entry_point_pathwise() -> None:
    simulator = _simulator()
    torch.manual_seed(2101)
    natural_spot, natural_variance = simulator.simulate(
        S0=100.0, T=0.3, dt=0.04, num_paths=128
    )
    torch.manual_seed(2101)
    controlled = simulator.simulate_controlled_two_driver(
        S0=100.0,
        T=0.3,
        dt=0.04,
        num_paths=128,
        control_fn=None,
        record_augmented=True,
    )
    assert torch.equal(controlled.spot, natural_spot)
    assert torch.equal(controlled.variance, natural_variance)
    assert torch.equal(controlled.log_likelihood, torch.zeros_like(controlled.log_likelihood))
    assert isinstance(controlled, TwoDriverRBergomiPaths)


def test_augmented_coordinates_and_likelihood_are_exact() -> None:
    simulator = _simulator()
    torch.manual_seed(2102)
    result = simulator.simulate_controlled_two_driver(
        S0=100.0,
        T=0.4,
        dt=0.06,
        num_paths=96,
        control_fn=_constant_control(-0.6, 0.35),
        record_augmented=True,
    )
    assert result.proposal_brownian_increments is not None
    assert result.target_brownian_increments is not None
    assert result.proposal_local_integrals is not None
    assert result.target_local_integrals is not None
    assert result.controls is not None
    expected_target = (
        result.proposal_brownian_increments + result.controls * result.step_dt
    )
    expected_log_likelihood = brownian_log_likelihood(
        result.controls, result.proposal_brownian_increments, result.step_dt
    )
    _local_cholesky, _weights, _variance, local_drift = simulator._hybrid_coefficients(
        result.controls.shape[1], result.step_dt, H=simulator.H, dtype=torch.float64
    )
    expected_local = (
        result.proposal_local_integrals + result.controls[:, :, 0] * local_drift
    )
    assert torch.equal(result.target_brownian_increments, expected_target)
    assert torch.equal(result.target_local_integrals, expected_local)
    assert torch.allclose(result.log_likelihood, expected_log_likelihood, atol=1e-14)


def test_local_blp_shift_has_no_extra_bridge_density() -> None:
    simulator = _simulator()
    dt = 0.025
    local_cholesky, _weights, _variance, local_drift = simulator._hybrid_coefficients(
        4, dt, H=simulator.H, dtype=torch.float64
    )
    covariance = local_cholesky @ local_cholesky.T
    control = 0.73
    mean_shift = torch.tensor([control * dt, control * local_drift], dtype=torch.float64)
    natural_parameter = torch.linalg.solve(covariance, mean_shift)
    assert natural_parameter[0] == pytest.approx(control, abs=1e-13)
    assert natural_parameter[1] == pytest.approx(0.0, abs=1e-13)
    assert float(mean_shift @ natural_parameter) == pytest.approx(
        control * control * dt, abs=1e-13
    )


def test_target_blp_path_equals_proposal_path_plus_all_cell_mean_shifts() -> None:
    simulator = _simulator()

    def feedback(
        time: float,
        spot: torch.Tensor,
        variance: torch.Tensor,
        volterra: torch.Tensor,
    ) -> torch.Tensor:
        first = -0.4 + 0.1 * torch.log(spot / 100.0) + 0.03 * time
        second = 0.2 + 0.05 * (variance / 0.04 - 1.0) + 0.01 * volterra
        return torch.stack((first, second), dim=-1)

    torch.manual_seed(2103)
    result = simulator.simulate_controlled_two_driver(
        S0=100.0,
        T=0.25,
        dt=0.04,
        num_paths=64,
        control_fn=feedback,
        record_augmented=True,
    )
    assert result.proposal_brownian_increments is not None
    assert result.proposal_local_integrals is not None
    assert result.controls is not None
    steps = result.controls.shape[1]
    _cholesky, weights, _variance, local_drift = simulator._hybrid_coefficients(
        steps, result.step_dt, H=simulator.H, dtype=torch.float64
    )
    scale = math.sqrt(2.0 * simulator.H)
    proposal_driver_one = result.proposal_brownian_increments[:, :, 0]
    proposal_volterra = scale * (
        proposal_driver_one @ weights.T + result.proposal_local_integrals
    )
    deterministic_shift = scale * (
        (result.controls[:, :, 0] * result.step_dt) @ weights.T
        + result.controls[:, :, 0] * local_drift
    )
    assert torch.allclose(
        result.volterra[:, 1:],
        proposal_volterra + deterministic_shift,
        atol=2e-12,
        rtol=2e-12,
    )


def test_target_coordinates_reconstruct_spot_variance_and_volterra() -> None:
    simulator = _simulator()
    torch.manual_seed(2104)
    result = simulator.simulate_controlled_two_driver(
        S0=100.0,
        T=0.2,
        dt=0.03,
        num_paths=48,
        control_fn=_constant_control(-0.5, 0.3),
        record_augmented=True,
    )
    assert result.target_brownian_increments is not None
    assert result.target_local_integrals is not None
    steps = result.target_brownian_increments.shape[1]
    _cholesky, weights, y_variance, _local_drift = simulator._hybrid_coefficients(
        steps, result.step_dt, H=simulator.H, dtype=torch.float64
    )
    rho_perpendicular = math.sqrt(1.0 - simulator.rho * simulator.rho)
    log_spot = torch.full((48,), math.log(100.0), dtype=torch.float64)
    driver_one_history: list[torch.Tensor] = []
    reconstructed_spot = [torch.exp(log_spot)]
    reconstructed_variance = [torch.full((48,), simulator.xi, dtype=torch.float64)]
    reconstructed_volterra = [torch.zeros(48, dtype=torch.float64)]
    for step in range(steps):
        current_variance = reconstructed_variance[-1]
        target = result.target_brownian_increments[:, step]
        log_spot = (
            log_spot
            - 0.5 * current_variance * result.step_dt
            + torch.sqrt(current_variance)
            * (simulator.rho * target[:, 0] + rho_perpendicular * target[:, 1])
        )
        driver_one_history.append(target[:, 0])
        earlier = torch.sum(
            torch.stack(driver_one_history, dim=1) * weights[step, : step + 1], dim=1
        )
        volterra = math.sqrt(2.0 * simulator.H) * (
            earlier + result.target_local_integrals[:, step]
        )
        variance = simulator.xi * torch.exp(
            simulator.eta * volterra
            - 0.5 * simulator.eta**2 * y_variance[step + 1]
        )
        reconstructed_spot.append(torch.exp(log_spot))
        reconstructed_volterra.append(volterra)
        reconstructed_variance.append(torch.clamp(variance, min=1e-10))
    assert torch.allclose(result.spot, torch.stack(reconstructed_spot, dim=1), atol=2e-12)
    assert torch.allclose(
        result.variance, torch.stack(reconstructed_variance, dim=1), atol=2e-12
    )
    assert torch.allclose(
        result.volterra, torch.stack(reconstructed_volterra, dim=1), atol=2e-12
    )


def test_constant_control_likelihood_normalizes_and_spot_is_unbiased() -> None:
    simulator = _simulator(rho=-0.7)
    torch.manual_seed(2105)
    result = simulator.simulate_controlled_two_driver(
        S0=100.0,
        T=0.25,
        dt=1.0 / 32.0,
        num_paths=80_000,
        control_fn=_constant_control(-0.45, 0.25),
        record_augmented=False,
    )
    likelihood = torch.exp(result.log_likelihood)
    assert float(likelihood.mean()) == pytest.approx(1.0, abs=0.008)
    weighted_terminal_spot = likelihood * result.spot[:, -1]
    standard_error = float(weighted_terminal_spot.std(unbiased=True)) / math.sqrt(
        weighted_terminal_spot.shape[0]
    )
    assert abs(float(weighted_terminal_spot.mean()) - 100.0) < 4.0 * standard_error + 0.15


def test_fixed_control_bounded_payoff_matches_natural_target() -> None:
    simulator = _simulator()
    paths = 40_000
    torch.manual_seed(2106)
    natural = simulator.simulate_controlled_two_driver(
        S0=100.0, T=0.25, dt=1.0 / 32.0, num_paths=paths, control_fn=None
    )
    torch.manual_seed(2107)
    proposal = simulator.simulate_controlled_two_driver(
        S0=100.0,
        T=0.25,
        dt=1.0 / 32.0,
        num_paths=paths,
        control_fn=_constant_control(-0.5, 0.3),
    )
    natural_payoff = torch.sigmoid((90.0 - natural.spot[:, -1]) / 5.0)
    proposal_payoff = torch.sigmoid((90.0 - proposal.spot[:, -1]) / 5.0)
    contribution = proposal_payoff * torch.exp(proposal.log_likelihood)
    difference = float(contribution.mean() - natural_payoff.mean())
    combined_se = math.sqrt(
        float(contribution.var(unbiased=True)) / paths
        + float(natural_payoff.var(unbiased=True)) / paths
    )
    assert abs(difference) < 4.0 * combined_se + 5e-4


def test_controlled_rbergomi_rejects_wrong_control_shape() -> None:
    def invalid(
        _time: float,
        spot: torch.Tensor,
        _variance: torch.Tensor,
        _volterra: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros_like(spot)

    with pytest.raises(ValueError, match="shape"):
        _simulator().simulate_controlled_two_driver(
            S0=100.0, T=0.1, dt=0.05, num_paths=8, control_fn=invalid
        )


def test_running_minimum_is_a_causal_path_state() -> None:
    torch.manual_seed(2108)
    result = _simulator().simulate_controlled_two_driver(
        S0=100.0,
        T=0.25,
        dt=1.0 / 16.0,
        num_paths=32,
        control_fn=_constant_control(-0.4, 0.2),
    )
    expected = torch.cummin(result.spot, dim=1).values
    assert torch.equal(result.running_minimum, expected)
