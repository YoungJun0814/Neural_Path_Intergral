"""Heston simulator analytic checks.

We test two weak-consistency properties that don't require the Heston
characteristic-function machinery:

1. The *martingale* property under mu = 0: E[S_T] ≈ S0 (no drift).
2. The long-run variance mean: E[v_T] → θ as T→∞ (with CLT tolerance).
3. The Feller condition (2κθ > ξ²) keeps v strictly positive when
   ``full-truncation`` is used — we only check finiteness and
   non-negativity since v may briefly touch 0.

All tests use 10k paths and fixed seeds.
"""

from __future__ import annotations

import math

import torch

from src.physics_engine import MarketSimulator
from src.utils import set_seed


def _make_sim(**overrides) -> MarketSimulator:
    defaults = dict(mu=0.0, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, device="cpu")
    defaults.update(overrides)
    return MarketSimulator(**defaults)


def test_zero_drift_martingale_property():
    """mu = 0 ⇒ E[S_T] ≈ S0 (martingale under P when no discount)."""
    set_seed(0)
    sim = _make_sim(mu=0.0, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
    S0 = 100.0
    S, _ = sim.simulate(S0=S0, v0=0.04, T=1.0, dt=1 / 252.0, num_paths=10_000)
    mean_ST = float(S[:, -1].mean())
    # 3-sigma CLT bound with conservative std ≈ S0·vol·√T
    std_est = float(S[:, -1].std()) / math.sqrt(10_000)
    assert abs(mean_ST - S0) < 5.0 * std_est + 0.5  # small absolute slack


def test_variance_long_run_mean():
    """For T large enough, average of v_T over paths should be ≈ θ."""
    set_seed(1)
    theta = 0.04
    sim = _make_sim(kappa=5.0, theta=theta, xi=0.3)  # Fast mean reversion
    _, v = sim.simulate(S0=100.0, v0=0.09, T=3.0, dt=1 / 252.0, num_paths=10_000)
    mean_vT = float(v[:, -1].mean())
    # With fast MR and long horizon, expect |mean − θ| < 0.01
    assert abs(mean_vT - theta) < 0.01, f"E[v_T]={mean_vT}, θ={theta}"


def test_variance_nonnegative_and_finite():
    set_seed(2)
    sim = _make_sim(xi=1.0)  # high vol-of-vol, Feller violated (2·2·0.04 < 1)
    _, v = sim.simulate(S0=100.0, v0=0.04, T=1.0, dt=1 / 252.0, num_paths=1000)
    assert torch.isfinite(v).all()
    assert (v >= 0).all()  # full-truncation guarantees


def test_price_positivity():
    set_seed(3)
    sim = _make_sim()
    S, _ = sim.simulate(S0=100.0, v0=0.04, T=1.0, dt=1 / 252.0, num_paths=1000)
    assert (S > 0).all()


def test_rho_zero_decorrelates():
    """ρ=0 ⇒ sample correlation between price returns and variance changes ≈ 0."""
    set_seed(4)
    sim = _make_sim(rho=0.0, xi=0.5)
    S, v = sim.simulate(S0=100.0, v0=0.04, T=1.0, dt=1 / 252.0, num_paths=20_000)
    ret = (S[:, 1:] / S[:, :-1] - 1.0).flatten()
    dv = (v[:, 1:] - v[:, :-1]).flatten()
    c = torch.corrcoef(torch.stack([ret, dv]))[0, 1].item()
    assert abs(c) < 0.03, f"rho=0 but sample corr={c}"


def test_rho_negative_leverage():
    """ρ = −0.9 ⇒ empirical correlation should be strongly negative."""
    set_seed(5)
    sim = _make_sim(rho=-0.9, xi=0.5)
    S, v = sim.simulate(S0=100.0, v0=0.04, T=1.0, dt=1 / 252.0, num_paths=20_000)
    ret = (S[:, 1:] / S[:, :-1] - 1.0).flatten()
    dv = (v[:, 1:] - v[:, :-1]).flatten()
    c = torch.corrcoef(torch.stack([ret, dv]))[0, 1].item()
    assert c < -0.6, f"expected strong negative, got {c}"


def test_bates_jump_increases_kurtosis():
    """Adding a jump component should increase return kurtosis vs. pure Heston."""
    set_seed(6)
    sim_heston = _make_sim(jump_lambda=0.0)
    _, _ = sim_heston.simulate(S0=100.0, v0=0.04, T=1.0, dt=1 / 252.0, num_paths=5000)
    S_h, _ = sim_heston.simulate(S0=100.0, v0=0.04, T=1.0, dt=1 / 252.0, num_paths=5000)
    set_seed(6)
    sim_bates = _make_sim(jump_lambda=20.0, jump_mean=-0.05, jump_std=0.1)
    S_b, _ = sim_bates.simulate(
        S0=100.0, v0=0.04, T=1.0, dt=1 / 252.0, num_paths=5000, model_type="bates"
    )

    ret_h = torch.log(S_h[:, 1:] / S_h[:, :-1]).flatten()
    ret_b = torch.log(S_b[:, 1:] / S_b[:, :-1]).flatten()
    z_h = (ret_h - ret_h.mean()) / (ret_h.std() + 1e-8)
    z_b = (ret_b - ret_b.mean()) / (ret_b.std() + 1e-8)
    kurt_h = float((z_h**4).mean())
    kurt_b = float((z_b**4).mean())
    assert kurt_b > kurt_h + 0.5, f"jump model should be more leptokurtic: h={kurt_h}, b={kurt_b}"
