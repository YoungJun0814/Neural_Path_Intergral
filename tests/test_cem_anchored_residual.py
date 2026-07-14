"""Exact-reduction tests for the CEM-anchored residual feedback controller."""

from __future__ import annotations

import math

import pytest
import torch

from experiments.g5_cem_anchored_residual import _mean_summary
from src.path_integral import (
    CEMAnchoredResidualControl,
    ConstantTwoDriverControl,
    simulate_rbergomi_mixture,
)
from src.physics_engine import RBergomiSimulator
from src.training.path_mixture import lean_soft_pi_objective


def _anchored(
    mode: str,
    base: tuple[float, float],
) -> CEMAnchoredResidualControl:
    return CEMAnchoredResidualControl(
        spot=100.0,
        xi=0.04,
        maturity=0.5,
        lower_threshold=42.0,
        upper_threshold=139.0,
        mode=mode,
        hidden_dim=8,
        control_bound=(6.0, 6.0),
        base_control=base,
        residual_bound=(2.0, 2.0),
    ).double()


def _simulator() -> RBergomiSimulator:
    return RBergomiSimulator(H=0.1, eta=1.5, xi=0.04, rho=-0.7, device="cpu")


def test_zero_residual_equals_cem_control_for_arbitrary_states() -> None:
    base = (3.2, -1.4)
    control = _anchored("left", base)
    spot = torch.tensor([55.0, 100.0, 145.0], dtype=torch.float64)
    variance = torch.tensor([0.01, 0.04, 0.2], dtype=torch.float64)
    volterra = torch.tensor([-0.3, 0.0, 0.6], dtype=torch.float64)
    actual = control(0.2, spot, variance, volterra)
    expected = torch.tensor(base, dtype=torch.float64).expand(3, -1)
    assert torch.equal(actual, expected)


def test_zero_residual_matches_constant_cem_pathwise() -> None:
    simulator = _simulator()
    base = (3.0, -1.2)
    anchored_control = _anchored("left", base)
    torch.manual_seed(7101)
    constant = simulator.simulate_controlled_two_driver(
        S0=100.0,
        T=0.5,
        dt=1.0 / 32.0,
        num_paths=128,
        control_fn=ConstantTwoDriverControl(*base),
        record_augmented=True,
        dtype=torch.float64,
    )
    torch.manual_seed(7101)
    anchored = simulator.simulate_controlled_two_driver(
        S0=100.0,
        T=0.5,
        dt=1.0 / 32.0,
        num_paths=128,
        control_fn=anchored_control,
        record_augmented=True,
        dtype=torch.float64,
    )
    assert torch.equal(anchored.spot, constant.spot)
    assert torch.equal(anchored.variance, constant.variance)
    assert torch.equal(anchored.log_likelihood, constant.log_likelihood)


def test_zero_residual_mixture_density_matches_cem_mixture_pathwise() -> None:
    simulator = _simulator()
    bases = [(3.0, -1.2), (0.2, 2.8)]
    weights = torch.tensor([0.5, 0.5], dtype=torch.float64)
    residual_controls = [
        _anchored(mode, base) for mode, base in zip(("left", "right"), bases, strict=True)
    ]
    torch.manual_seed(7201)
    cem = simulate_rbergomi_mixture(
        simulator,
        [ConstantTwoDriverControl(*base) for base in bases],
        weights,
        spot=100.0,
        maturity=0.5,
        dt=1.0 / 32.0,
        num_paths=256,
        label_generator=torch.Generator().manual_seed(7202),
    )
    torch.manual_seed(7201)
    residual = simulate_rbergomi_mixture(
        simulator,
        residual_controls,
        weights,
        spot=100.0,
        maturity=0.5,
        dt=1.0 / 32.0,
        num_paths=256,
        label_generator=torch.Generator().manual_seed(7202),
    )
    assert torch.equal(residual.paths.spot, cem.paths.spot)
    assert torch.equal(residual.component_log_q_over_p, cem.component_log_q_over_p)
    assert torch.equal(residual.mixture_log_likelihood, cem.mixture_log_likelihood)


def test_residual_training_has_gradient_and_respects_global_bound() -> None:
    simulator = _simulator()
    control = _anchored("right", (0.2, 3.0))
    objective = lean_soft_pi_objective(
        simulator,
        control,
        spot=100.0,
        maturity=0.5,
        dt=1.0 / 16.0,
        num_paths=256,
        lower_threshold=48.0,
        upper_threshold=135.0,
        soft_scale=5.0,
        mode="right",
    )
    objective.loss.backward()
    assert any(
        parameter.grad is not None and bool(torch.any(parameter.grad != 0.0))
        for parameter in control.parameters()
    )
    with torch.no_grad():
        control.network[-1].bias.fill_(100.0)
    state = torch.full((4,), 100.0, dtype=torch.float64)
    output = control(0.0, state, torch.full_like(state, 0.04), torch.zeros_like(state))
    assert torch.all(torch.abs(output) <= 6.0)


def test_state_validation_rejects_nonfinite_volterra_values() -> None:
    control = _anchored("left", (3.0, -1.2))
    state = torch.full((2,), 100.0, dtype=torch.float64)
    volterra = torch.tensor([0.0, math.nan], dtype=torch.float64)
    with pytest.raises(ValueError, match="finite"):
        control(0.0, state, torch.full_like(state, 0.04), volterra)


def test_summary_uses_standard_error_of_the_cross_seed_mean() -> None:
    runs = [
        {
            "method": "candidate",
            "estimate": 0.1,
            "standard_error": 0.1,
            "single_path_variance": 1.0,
            "cost_per_path": 2.0,
            "online_work_proxy": 2.0,
            "left_contribution_share": 0.5,
            "right_contribution_share": 0.5,
            "contribution_ess_fraction": 0.1,
        },
        {
            "method": "candidate",
            "estimate": 0.2,
            "standard_error": 0.2,
            "single_path_variance": 2.0,
            "cost_per_path": 3.0,
            "online_work_proxy": 6.0,
            "left_contribution_share": 0.4,
            "right_contribution_share": 0.6,
            "contribution_ess_fraction": 0.2,
        },
    ]
    summary = _mean_summary(runs, "candidate")
    assert summary["mean_per_seed_standard_error"] == pytest.approx(0.15)
    assert summary["combined_standard_error"] == pytest.approx(math.sqrt(0.05) / 2.0)
