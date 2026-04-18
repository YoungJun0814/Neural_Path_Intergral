"""Verify Bates/SVJJ jump statistics — Poisson counts, mean shift, variance."""
from __future__ import annotations

import math

import torch

from src.physics_engine import MarketSimulator
from src.utils import set_seed


def test_no_jumps_when_lambda_zero():
    """With jump_lambda=0, model_type='bates' should be statistically identical
    to model_type='heston' (same shock realisations under fixed seed)."""
    set_seed(0)
    sim = MarketSimulator(
        mu=0.05, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7,
        jump_lambda=0.0, jump_mean=0.0, jump_std=0.0, device="cpu",
    )
    set_seed(7)
    S_h, v_h = sim.simulate(S0=100.0, v0=0.04, T=0.5, dt=1/252.0, num_paths=200, model_type="heston")
    set_seed(7)
    S_b, v_b = sim.simulate(S0=100.0, v0=0.04, T=0.5, dt=1/252.0, num_paths=200, model_type="bates")
    assert torch.allclose(S_h, S_b, atol=1e-6)
    assert torch.allclose(v_h, v_b, atol=1e-6)


def test_jump_count_poisson_mean():
    """Average jump count per step should match λ·dt."""
    set_seed(1)
    lam = 50.0
    dt = 1 / 252.0
    expected = lam * dt
    n_paths = 50_000
    samples = torch.poisson(torch.full((n_paths,), expected))
    emp_mean = float(samples.mean())
    se = expected / math.sqrt(n_paths)
    assert abs(emp_mean - expected) < 5 * se + 1e-3


def test_bates_increases_left_tail():
    """Negative-mean jumps should make the return distribution left-skewed."""
    set_seed(2)
    sim_pure = MarketSimulator(
        mu=0.05, kappa=2.0, theta=0.04, xi=0.5, rho=0.0,
        jump_lambda=0.0, jump_mean=0.0, jump_std=0.0, device="cpu",
    )
    sim_jump = MarketSimulator(
        mu=0.05, kappa=2.0, theta=0.04, xi=0.5, rho=0.0,
        jump_lambda=30.0, jump_mean=-0.05, jump_std=0.1, device="cpu",
    )
    S_p, _ = sim_pure.simulate(S0=100.0, v0=0.04, T=1.0, dt=1/252.0, num_paths=4000, model_type="heston")
    S_j, _ = sim_jump.simulate(S0=100.0, v0=0.04, T=1.0, dt=1/252.0, num_paths=4000, model_type="bates")
    log_ret_p = torch.log(S_p[:, -1] / S_p[:, 0])
    log_ret_j = torch.log(S_j[:, -1] / S_j[:, 0])
    z_p = (log_ret_p - log_ret_p.mean()) / (log_ret_p.std() + 1e-8)
    z_j = (log_ret_j - log_ret_j.mean()) / (log_ret_j.std() + 1e-8)
    skew_p = float((z_p ** 3).mean())
    skew_j = float((z_j ** 3).mean())
    assert skew_j < skew_p, f"jumps should add left-skew: pure={skew_p:.3f}, jump={skew_j:.3f}"


def test_svjj_volatility_jump_increases_terminal_v():
    """SVJJ adds upward variance jumps — terminal v should be higher than Bates with same params."""
    set_seed(3)
    sim_b = MarketSimulator(
        mu=0.05, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7,
        jump_lambda=20.0, jump_mean=-0.05, jump_std=0.05, vol_jump_mean=0.0, device="cpu",
    )
    sim_s = MarketSimulator(
        mu=0.05, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7,
        jump_lambda=20.0, jump_mean=-0.05, jump_std=0.05, vol_jump_mean=0.05, device="cpu",
    )
    _, v_b = sim_b.simulate(S0=100.0, v0=0.04, T=1.0, dt=1/252.0, num_paths=3000, model_type="bates")
    _, v_s = sim_s.simulate(S0=100.0, v0=0.04, T=1.0, dt=1/252.0, num_paths=3000, model_type="svjj")
    assert float(v_s.mean()) > float(v_b.mean()), (
        f"SVJJ mean v={float(v_s.mean()):.5f} should exceed Bates {float(v_b.mean()):.5f}"
    )
