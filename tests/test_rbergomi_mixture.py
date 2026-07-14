"""Law and replay tests for exact mixtures of two-driver rBergomi controls."""

from __future__ import annotations

import math

import pytest
import torch

from src.path_integral import (
    LeanRBergomiControl,
    log_mixture_q_over_p,
    replay_rbergomi_control_on_target_paths,
    simulate_rbergomi_mixture,
)
from src.physics_engine import RBergomiSimulator


class _ConstantControl:
    def __init__(self, first: float, second: float) -> None:
        self.first = first
        self.second = second

    def __call__(
        self,
        _time: float,
        spot: torch.Tensor,
        _variance: torch.Tensor,
        _volterra: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack(
            (
                torch.full_like(spot, self.first),
                torch.full_like(spot, self.second),
            ),
            dim=-1,
        )


def _simulator() -> RBergomiSimulator:
    return RBergomiSimulator(H=0.1, eta=1.2, xi=0.04, rho=-0.7, device="cpu")


def _lean(mode: str = "union") -> LeanRBergomiControl:
    return LeanRBergomiControl(
        spot=100.0,
        xi=0.04,
        maturity=0.25,
        lower_threshold=80.0,
        upper_threshold=125.0,
        mode=mode,  # type: ignore[arg-type]
        hidden_dim=12,
        control_bound=(5.0, 5.0),
    ).double()


def test_lean_controller_starts_at_exact_null_control() -> None:
    control = _lean()
    batch = 7
    output = control(
        0.0,
        torch.full((batch,), 100.0, dtype=torch.float64),
        torch.full((batch,), 0.04, dtype=torch.float64),
        torch.zeros(batch, dtype=torch.float64),
    )
    assert torch.equal(output, torch.zeros_like(output))


def test_single_component_mixture_matches_controlled_entry_point_pathwise() -> None:
    simulator = _simulator()
    control = _ConstantControl(-0.4, 0.25)
    torch.manual_seed(310)
    direct = simulator.simulate_controlled_two_driver(
        S0=100.0,
        T=0.25,
        dt=1.0 / 32.0,
        num_paths=128,
        control_fn=control,
        record_augmented=True,
        dtype=torch.float64,
    )
    torch.manual_seed(310)
    labels = torch.Generator().manual_seed(999)
    mixture = simulate_rbergomi_mixture(
        simulator,
        [control],
        torch.ones(1, dtype=torch.float64),
        spot=100.0,
        maturity=0.25,
        dt=1.0 / 32.0,
        num_paths=128,
        label_generator=labels,
    )
    assert torch.equal(mixture.paths.spot, direct.spot)
    assert torch.equal(mixture.paths.variance, direct.variance)
    assert torch.allclose(
        mixture.mixture_log_likelihood, direct.log_likelihood, atol=2e-14, rtol=0.0
    )
    assert mixture.maximum_selected_replay_error <= 2e-14


def test_all_expert_replay_matches_constant_control_analytic_density() -> None:
    simulator = _simulator()
    controls = [_ConstantControl(-0.6, 0.3), _ConstantControl(0.45, -0.2)]
    torch.manual_seed(811)
    sample = simulate_rbergomi_mixture(
        simulator,
        controls,
        torch.tensor([0.4, 0.6], dtype=torch.float64),
        spot=100.0,
        maturity=0.2,
        dt=0.025,
        num_paths=257,
        label_generator=torch.Generator().manual_seed(812),
    )
    target = sample.paths.target_brownian_increments
    assert target is not None
    expected = []
    for control in controls:
        value = torch.tensor([control.first, control.second], dtype=torch.float64)
        stochastic = torch.sum(target * value, dim=(1, 2))
        energy = 0.2 * torch.sum(value.square())
        expected.append(stochastic - 0.5 * energy)
    expected_tensor = torch.stack(expected, dim=-1)
    assert torch.allclose(
        sample.component_log_q_over_p, expected_tensor, atol=3e-14, rtol=3e-14
    )
    assert sample.paths.controls is not None
    selected = sample.all_expert_controls[
        torch.arange(sample.labels.numel()), sample.labels
    ]
    assert torch.equal(selected, sample.paths.controls)
    assert sample.maximum_selected_replay_error <= 3e-14


def test_expert_label_permutation_leaves_marginal_density_invariant() -> None:
    component = torch.tensor(
        [[-1.2, 0.4], [2.1, -0.7], [0.0, 0.3]], dtype=torch.float64
    )
    weights = torch.tensor([0.35, 0.65], dtype=torch.float64)
    original = log_mixture_q_over_p(component, weights)
    permuted = log_mixture_q_over_p(component[:, [1, 0]], weights[[1, 0]])
    assert torch.equal(original, permuted)


def test_replay_rejects_unrecorded_target_path() -> None:
    simulator = _simulator()
    paths = simulator.simulate_controlled_two_driver(
        S0=100.0,
        T=0.1,
        dt=0.05,
        num_paths=8,
        control_fn=None,
        record_augmented=False,
    )
    with pytest.raises(ValueError, match="must be recorded"):
        replay_rbergomi_control_on_target_paths(_ConstantControl(0.0, 0.0), paths)


def test_two_component_mixture_likelihood_normalizes() -> None:
    simulator = _simulator()
    torch.manual_seed(1901)
    sample = simulate_rbergomi_mixture(
        simulator,
        [_ConstantControl(-0.55, 0.25), _ConstantControl(0.45, -0.15)],
        torch.tensor([0.45, 0.55], dtype=torch.float64),
        spot=100.0,
        maturity=0.125,
        dt=1.0 / 32.0,
        num_paths=40_000,
        label_generator=torch.Generator().manual_seed(1902),
    )
    likelihood = torch.exp(sample.mixture_log_likelihood)
    standard_error = float(likelihood.std(unbiased=True) / math.sqrt(likelihood.numel()))
    assert abs(float(likelihood.mean()) - 1.0) <= 3.0 * standard_error
    component_likelihood = torch.exp(sample.selected_component_log_likelihood)
    component_se = float(
        component_likelihood.std(unbiased=True) / math.sqrt(component_likelihood.numel())
    )
    assert abs(float(component_likelihood.mean()) - 1.0) <= 3.0 * component_se
