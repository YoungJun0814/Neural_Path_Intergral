"""Statistical correctness tests for the normalized rBergomi hybrid scheme."""

from __future__ import annotations

import math

import pytest
import torch

from src.physics_engine import RBergomiSimulator
from src.utils import set_seed


@pytest.mark.parametrize("H", [0.05, 0.10, 0.30])
def test_hybrid_volterra_variance_matches_discrete_covariance(H: float) -> None:
    """The sampler must realize its deterministic Gaussian covariance."""
    set_seed(100 + int(100 * H))
    sim = RBergomiSimulator(H=H, eta=1.0, xi=0.04, rho=-0.7, device="cpu")
    _, Y, variance = sim._hybrid_scheme_volterra(num_paths=8_000, n_steps=16, dt=1.0 / 64.0)

    assert torch.isfinite(Y).all()
    assert torch.isfinite(variance).all()
    for index in (1, 4, 16):
        empirical = Y[:, index].var(unbiased=True)
        relative_error = abs(float(empirical / variance[index]) - 1.0)
        assert relative_error < 0.06, (
            f"H={H} index={index}: empirical={empirical} "
            f"target={variance[index]} rel_error={relative_error}"
        )


def test_hybrid_normalization_and_refinement() -> None:
    """The normalized process has Var(W_dt^H)=dt^(2H) exactly at step one."""
    H = 0.1
    dt = 1.0 / 128.0
    sim = RBergomiSimulator(H=H, eta=1.0, xi=0.04, rho=0.0, device="cpu")
    _, _, variance = sim._hybrid_scheme_volterra(num_paths=2, n_steps=64, dt=dt)

    assert math.isclose(float(variance[1]), dt ** (2.0 * H), rel_tol=1e-12)
    terminal_exact = (64 * dt) ** (2.0 * H)
    assert abs(float(variance[-1]) / terminal_exact - 1.0) < 0.03


def test_discrete_wick_correction_preserves_forward_variance_mean() -> None:
    """Using the hybrid variance in the Wick term should preserve E[V_t]=xi."""
    set_seed(211)
    xi = 0.04
    sim = RBergomiSimulator(H=0.1, eta=1.2, xi=xi, rho=-0.7, device="cpu")
    _, variance_paths = sim.simulate(S0=100.0, T=0.25, dt=1.0 / 64.0, num_paths=20_000)

    mean_terminal = float(variance_paths[:, -1].mean())
    standard_error = float(variance_paths[:, -1].std(unbiased=True)) / math.sqrt(
        variance_paths.shape[0]
    )
    assert abs(mean_terminal - xi) < 4.0 * standard_error + 5e-4


def test_noninteger_requested_step_hits_exact_maturity() -> None:
    """The simulator uses ceil(T/dt) equal steps whose total is exactly T."""
    sim = RBergomiSimulator(H=0.1, eta=1.0, xi=0.04, rho=0.0, device="cpu")
    S, V = sim.simulate(S0=100.0, T=0.3, dt=0.07, num_paths=8)
    assert S.shape == V.shape == (8, math.ceil(0.3 / 0.07) + 1)
    assert torch.isfinite(S).all() and torch.isfinite(V).all()


def test_only_implemented_hybrid_kappa_is_accepted() -> None:
    with pytest.raises(ValueError, match="kappa=1"):
        RBergomiSimulator(kappa_hybrid=2, device="cpu")


def test_parameter_override_does_not_mutate_simulator() -> None:
    sim = RBergomiSimulator(H=0.1, eta=1.0, xi=0.04, rho=-0.7, device="cpu")
    sim.simulate(
        S0=100.0,
        T=0.05,
        dt=1.0 / 64.0,
        num_paths=4,
        override_params={"H": 0.2},
    )
    assert sim.H == 0.1
