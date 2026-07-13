"""Unit tests for the discrete path-integral action convention."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.path_integral import (
    brownian_log_likelihood,
    log_tilted_weight,
    path_action,
    terminal_left_tail_potential,
)


def test_two_driver_action_matches_manual_likelihood() -> None:
    controls = torch.tensor(
        [
            [[0.2, -0.1], [0.3, 0.4], [-0.2, 0.1]],
            [[-0.4, 0.2], [0.1, -0.3], [0.2, 0.5]],
        ],
        dtype=torch.float64,
    )
    increments = torch.tensor(
        [
            [[0.05, -0.02], [-0.08, 0.04], [0.01, 0.03]],
            [[-0.03, 0.07], [0.02, -0.04], [0.06, -0.01]],
        ],
        dtype=torch.float64,
    )
    potential = torch.tensor([0.7, 1.1], dtype=torch.float64)
    dt = 0.125

    manual_log_likelihood = -torch.sum(controls * increments, dim=(1, 2))
    manual_log_likelihood -= 0.5 * dt * torch.sum(controls.square(), dim=(1, 2))

    actual_log_likelihood = brownian_log_likelihood(controls, increments, dt)
    actual_action = path_action(potential, controls, increments, dt)
    actual_log_tilted = log_tilted_weight(potential, controls, increments, dt)

    assert torch.equal(actual_log_likelihood, manual_log_likelihood)
    assert torch.equal(actual_action, potential - manual_log_likelihood)
    assert torch.equal(actual_log_tilted, -potential + manual_log_likelihood)


def test_path_action_gradient_recovers_exponential_tilt_direction() -> None:
    torch.manual_seed(19)
    half_increments = torch.randn(256, 5, 1, dtype=torch.float64) * (0.2**0.5)
    increments = torch.cat((half_increments, -half_increments), dim=0)
    horizon = 1.0
    dt = horizon / increments.shape[1]
    tilt = -0.8
    control_value = 0.25
    control = torch.tensor(control_value, dtype=torch.float64, requires_grad=True)
    controls = control.expand_as(increments)

    # Under Q_u, B_T^M = B_T^Q + uT and Phi=-a B_T^M.
    target_terminal = torch.sum(increments, dim=(1, 2)) + control * horizon
    potential = -tilt * target_terminal
    objective = path_action(potential, controls, increments, dt).mean()
    objective.backward()

    assert control.grad is not None
    assert float(control.grad) == pytest.approx((control_value - tilt) * horizon, abs=1e-12)


def test_left_tail_potential_is_stable_negative_log_sigmoid() -> None:
    terminal = torch.tensor([1.0, 80.0, 100.0, 120.0, 10_000.0], dtype=torch.float64)
    barrier = 100.0
    temperature = 0.05
    potential = terminal_left_tail_potential(terminal, barrier, temperature)
    expected = -F.logsigmoid((barrier - terminal) / (temperature * barrier))

    assert torch.allclose(potential, expected)
    assert torch.isfinite(potential).all()
    assert torch.all(potential >= 0.0)
    assert potential[0] < potential[-1]


@pytest.mark.parametrize("dt", [0.0, -0.1, float("nan")])
def test_action_rejects_invalid_time_step(dt: float) -> None:
    values = torch.zeros(2, 3, 1)
    with pytest.raises(ValueError, match="dt"):
        brownian_log_likelihood(values, values, dt)


def test_action_rejects_shape_mismatch() -> None:
    controls = torch.zeros(2, 3, 1)
    increments = torch.zeros(2, 3, 2)
    with pytest.raises(ValueError, match="identical shapes"):
        brownian_log_likelihood(controls, increments, 0.1)
