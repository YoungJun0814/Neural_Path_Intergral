"""Tests for the two-driver rBergomi constant-mixture CEM baseline."""

from __future__ import annotations

import math

import torch

from src.path_integral import ConstantTwoDriverControl
from src.physics_engine import RBergomiSimulator
from src.training.rbergomi_cem import fit_rbergomi_two_driver_cem


def _simulator() -> RBergomiSimulator:
    return RBergomiSimulator(H=0.1, eta=1.2, xi=0.04, rho=-0.7, device="cpu")


def test_constant_two_driver_control_preserves_dtype_and_shape() -> None:
    control = ConstantTwoDriverControl(0.4, -0.2)
    spot = torch.full((5,), 100.0, dtype=torch.float32)
    result = control(0.0, spot, spot, spot)
    assert result.shape == (5, 2)
    assert result.dtype == torch.float32
    assert torch.equal(result[0], torch.tensor([0.4, -0.2]))


def test_cem_learns_opposite_spot_drift_directions_for_two_tails() -> None:
    simulator = _simulator()
    common = {
        "simulator": simulator,
        "spot": 100.0,
        "maturity": 0.25,
        "dt": 1.0 / 16.0,
        "num_paths": 2_000,
        "max_iterations": 5,
        "elite_quantile": 0.85,
        "smoothing": 0.6,
        "min_elite_paths": 32,
        "control_bound": 6.0,
        "target_level_repetitions": 1,
    }
    left = fit_rbergomi_two_driver_cem(
        **common,
        threshold=82.0,
        mode="left",
        initial_control=(0.8, -0.8),
        seed=601,
    )
    right = fit_rbergomi_two_driver_cem(
        **common,
        threshold=118.0,
        mode="right",
        initial_control=(-0.8, 0.8),
        seed=602,
    )
    perpendicular = math.sqrt(1.0 - simulator.rho**2)
    left_spot_drift = simulator.rho * left.control[0] + perpendicular * left.control[1]
    right_spot_drift = simulator.rho * right.control[0] + perpendicular * right.control[1]
    assert left_spot_drift < 0.0
    assert right_spot_drift > 0.0
    assert left.history and right.history
    assert all(math.isfinite(item.hard_probability_estimate) for item in left.history)
    assert all(math.isfinite(item.hard_probability_estimate) for item in right.history)
