"""Architecture, adaptedness, and exact-density tests for SDV."""

from __future__ import annotations

from dataclasses import replace

import torch

from src.path_integral import (
    DownsideExcursionTask,
    SpectralDoobVolterraControl,
    TimePiecewiseTwoDriverControl,
    simulate_rbergomi_mixture,
)
from src.physics_engine import RBergomiSimulator
from src.training import replay_sdv_outputs_on_target_paths, sdv_regression_objective


def _simulator() -> RBergomiSimulator:
    return RBergomiSimulator(H=0.1, eta=1.4, xi=0.04, rho=-0.7, device="cpu")


def _task() -> DownsideExcursionTask:
    return DownsideExcursionTask(
        hit_barrier=75.0,
        stress_level=90.0,
        minimum_occupation=0.10,
        hit_scale=4.0,
        occupation_scale=0.04,
    )


def _control() -> SpectralDoobVolterraControl:
    return SpectralDoobVolterraControl(
        H=0.1,
        spot=100.0,
        xi=0.04,
        maturity=0.25,
        hit_barrier=75.0,
        stress_level=90.0,
        minimum_occupation=0.10,
        minimum_dt=1.0 / 32.0,
        anchor_values=((1.2, -0.7), (0.4, -0.2)),
        soe_terms=4,
        hidden_dim=12,
        control_bound=(6.0, 6.0),
        residual_bound=(1.0, 1.0),
    ).double()


def test_sdv_zero_residual_initialization_equals_piecewise_anchor() -> None:
    control = _control()
    batch = 4
    control.reset_for_simulation(
        batch_size=batch, device=torch.device("cpu"), dtype=torch.float64
    )
    spot = torch.full((batch,), 100.0, dtype=torch.float64)
    first = control(0.0, spot, torch.full_like(spot, 0.04), torch.zeros_like(spot), spot)
    control.observe_target_increment(torch.zeros(batch, dtype=torch.float64), 0.125)
    second = control(
        0.125, spot, torch.full_like(spot, 0.04), torch.zeros_like(spot), spot
    )
    assert torch.allclose(
        first,
        torch.tensor([[1.2, -0.7]], dtype=torch.float64).expand(batch, -1),
        atol=1e-14,
    )
    assert torch.allclose(
        second,
        torch.tensor([[0.4, -0.2]], dtype=torch.float64).expand(batch, -1),
        atol=1e-14,
    )
    assert torch.allclose(
        control.last_desirability,
        torch.full((batch,), 0.05, dtype=torch.float64),
        atol=1e-14,
    )


def test_sdv_occupation_state_uses_observed_right_endpoints_only() -> None:
    control = _control()
    control.reset_for_simulation(
        batch_size=2, device=torch.device("cpu"), dtype=torch.float64
    )
    initial = torch.full((2,), 100.0, dtype=torch.float64)
    control(0.0, initial, torch.full_like(initial, 0.04), torch.zeros_like(initial), initial)
    current = torch.tensor([89.0, 91.0], dtype=torch.float64)
    control(
        0.05,
        current,
        torch.full_like(current, 0.04),
        torch.zeros_like(current),
        current,
    )
    assert control._occupation is not None
    assert torch.equal(control._occupation, torch.tensor([0.05, 0.0], dtype=torch.float64))


def test_sdv_replay_is_unchanged_before_a_suffix_perturbation() -> None:
    torch.manual_seed(2801)
    paths = _simulator().simulate_controlled_two_driver(
        S0=100.0,
        T=0.25,
        dt=1.0 / 32.0,
        num_paths=32,
        control_fn=None,
        record_augmented=True,
        dtype=torch.float64,
    )
    assert paths.target_brownian_increments is not None
    changed_target = paths.target_brownian_increments.clone()
    changed_target[:, 4:] += 3.0
    changed = replace(
        paths,
        spot=torch.cat((paths.spot[:, :5], paths.spot[:, 5:] * 1.2), dim=1),
        variance=torch.cat((paths.variance[:, :5], paths.variance[:, 5:] * 1.3), dim=1),
        volterra=torch.cat((paths.volterra[:, :5], paths.volterra[:, 5:] + 2.0), dim=1),
        running_minimum=torch.cat(
            (paths.running_minimum[:, :5], paths.running_minimum[:, 5:] * 0.5), dim=1
        ),
        target_brownian_increments=changed_target,
    )
    control = _control()
    output = control.residual_network[-1]
    assert isinstance(output, torch.nn.Linear)
    torch.nn.init.normal_(output.weight, std=0.1)
    original_h, original_u = replay_sdv_outputs_on_target_paths(control, paths)
    changed_h, changed_u = replay_sdv_outputs_on_target_paths(control, changed)
    assert torch.equal(original_h[:, :5], changed_h[:, :5])
    assert torch.equal(original_u[:, :5], changed_u[:, :5])


def test_stateful_sdv_has_exact_all_expert_replay_density() -> None:
    control = _control()
    anchor = TimePiecewiseTwoDriverControl(
        ((1.2, -0.7), (0.4, -0.2)), maturity=0.25
    )
    torch.manual_seed(2802)
    sample = simulate_rbergomi_mixture(
        _simulator(),
        (anchor, control),
        torch.tensor([0.4, 0.6], dtype=torch.float64),
        spot=100.0,
        maturity=0.25,
        dt=1.0 / 32.0,
        num_paths=2_000,
        label_generator=torch.Generator().manual_seed(2803),
    )
    assert sample.maximum_selected_replay_error < 2e-13
    assert torch.isfinite(sample.mixture_log_likelihood).all()


def test_sdv_conditional_moment_objective_is_finite_and_differentiable() -> None:
    control = _control()
    torch.manual_seed(2804)
    result = sdv_regression_objective(
        _simulator(),
        control,
        _task(),
        spot=100.0,
        maturity=0.25,
        dt=1.0 / 16.0,
        num_paths=192,
        natural_behavior_mass=0.25,
        label_seed=2805,
    )
    result.loss.backward()
    assert torch.isfinite(result.loss)
    assert 0.0 < float(result.behavior_ess_fraction) <= 1.0
    assert result.maximum_selected_replay_error < 2e-13
    assert any(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in control.parameters()
    )
