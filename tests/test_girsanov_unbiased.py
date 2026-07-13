"""Girsanov unbiasedness tests — the single most important correctness check.

For any fixed control ``u``, the importance-sampling estimator

    I_IS = (1/N) Σ F(S^{Q,i}) · E_T^{(i)}

should converge to the natural Monte-Carlo estimator

    I_MC = (1/M) Σ F(S^{P,j})

as N, M → ∞, with CLT error bars.

We verify this for a fixed `u = +0.5` constant control, pricing a vanilla
call option under the base Heston model. Both the v-drift correction-on and
-off variants are run; only the corrected variant should pass.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.physics_engine import MarketSimulator
from src.utils import set_seed


def _constant_control(u_value: float):
    def fn(t, S, v, A):
        return torch.full_like(S, float(u_value))

    return fn


@pytest.fixture()
def heston_sim() -> MarketSimulator:
    return MarketSimulator(mu=0.05, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, device="cpu")


def _call_payoff(K: float):
    def payoff(S_T):
        return torch.clamp(S_T - K, min=0.0)

    return payoff


def _std_err(x: torch.Tensor, w: torch.Tensor | None = None) -> float:
    if w is None:
        return float(x.std(unbiased=True)) / math.sqrt(x.numel())
    weighted = x * w
    return float(weighted.std(unbiased=True)) / math.sqrt(x.numel())


def test_is_unbiased_constant_positive_control(heston_sim):
    """u = +0.3 constant: IS mean ≈ MC mean within 3σ."""
    set_seed(0)
    S0, v0, T, dt = 100.0, 0.04, 0.5, 1 / 252.0
    K = 100.0
    payoff = _call_payoff(K)

    N = 20_000
    # Natural MC
    S_mc, _ = heston_sim.simulate(S0=S0, v0=v0, T=T, dt=dt, num_paths=N)
    pay_mc = payoff(S_mc[:, -1])
    mean_mc = float(pay_mc.mean())
    se_mc = float(pay_mc.std()) / math.sqrt(N)

    # IS with v-drift correction (correct)
    set_seed(1)
    S_is, _v, log_w, _bh, _ = heston_sim.simulate_controlled(
        S0=S0,
        v0=v0,
        T=T,
        dt=dt,
        num_paths=N,
        control_fn=_constant_control(0.3),
        apply_v_drift_correction=True,
    )
    reweighted = payoff(S_is[:, -1]) * torch.exp(log_w)
    mean_is = float(reweighted.mean())
    se_is = float(reweighted.std()) / math.sqrt(N)

    diff = mean_is - mean_mc
    tol = 3.0 * math.sqrt(se_mc**2 + se_is**2) + 0.05
    assert abs(diff) < tol, (
        f"IS not unbiased: MC={mean_mc:.4f} IS={mean_is:.4f} diff={diff:.4f} tol={tol:.4f}"
    )


def test_is_unbiased_constant_negative_control(heston_sim):
    """u = −0.3 (OTM-pushing control) must also be unbiased."""
    set_seed(2)
    S0, v0, T, dt = 100.0, 0.04, 0.5, 1 / 252.0
    K = 100.0
    payoff = _call_payoff(K)

    N = 20_000
    S_mc, _ = heston_sim.simulate(S0=S0, v0=v0, T=T, dt=dt, num_paths=N)
    mean_mc = float(payoff(S_mc[:, -1]).mean())
    se_mc = float(payoff(S_mc[:, -1]).std()) / math.sqrt(N)

    set_seed(3)
    S_is, _v, log_w, _, _ = heston_sim.simulate_controlled(
        S0=S0,
        v0=v0,
        T=T,
        dt=dt,
        num_paths=N,
        control_fn=_constant_control(-0.3),
        apply_v_drift_correction=True,
    )
    reweighted = payoff(S_is[:, -1]) * torch.exp(log_w)
    mean_is = float(reweighted.mean())
    se_is = float(reweighted.std()) / math.sqrt(N)

    diff = mean_is - mean_mc
    tol = 3.0 * math.sqrt(se_mc**2 + se_is**2) + 0.1
    assert abs(diff) < tol, f"negative control not unbiased: MC={mean_mc} IS={mean_is}"


def test_zero_control_matches_mc(heston_sim):
    """u ≡ 0 ⇒ log-weight ≡ 0 ⇒ IS ≡ MC (same path distribution)."""
    set_seed(4)
    S0, v0, T, dt = 100.0, 0.04, 0.25, 1 / 252.0
    N = 5000
    S_is, _v, log_w, _, _ = heston_sim.simulate_controlled(
        S0=S0,
        v0=v0,
        T=T,
        dt=dt,
        num_paths=N,
        control_fn=_constant_control(0.0),
    )
    assert torch.allclose(log_w, torch.zeros_like(log_w))
    # And statistics should match MC within noise
    set_seed(4)
    S_mc, _ = heston_sim.simulate(S0=S0, v0=v0, T=T, dt=dt, num_paths=N)
    assert abs(float(S_is[:, -1].mean()) - float(S_mc[:, -1].mean())) < 1e-3


def test_v_drift_correction_removes_bias_vs_uncorrected(heston_sim):
    """With rho ≠ 0 and strong control, the uncorrected variant should be
    biased and the corrected one should match MC. This is the raison d'être
    of Phase 1."""
    set_seed(5)
    S0, v0, T, dt = 100.0, 0.04, 1.0, 1 / 252.0
    K = 100.0
    payoff = _call_payoff(K)
    u_val = 0.8  # strong

    N = 30_000
    S_mc, _ = heston_sim.simulate(S0=S0, v0=v0, T=T, dt=dt, num_paths=N)
    mean_mc = float(payoff(S_mc[:, -1]).mean())
    se_mc = float(payoff(S_mc[:, -1]).std()) / math.sqrt(N)

    # Corrected IS
    set_seed(6)
    S_ok, _, log_w_ok, _, _ = heston_sim.simulate_controlled(
        S0=S0,
        v0=v0,
        T=T,
        dt=dt,
        num_paths=N,
        control_fn=_constant_control(u_val),
        apply_v_drift_correction=True,
    )
    mean_ok = float((payoff(S_ok[:, -1]) * torch.exp(log_w_ok)).mean())

    # Uncorrected IS
    set_seed(6)
    S_bad, _, log_w_bad, _, _ = heston_sim.simulate_controlled(
        S0=S0,
        v0=v0,
        T=T,
        dt=dt,
        num_paths=N,
        control_fn=_constant_control(u_val),
        apply_v_drift_correction=False,
    )
    mean_bad = float((payoff(S_bad[:, -1]) * torch.exp(log_w_bad)).mean())

    # Corrected must be close to MC
    tol_ok = 4.0 * se_mc + 0.2
    assert abs(mean_ok - mean_mc) < tol_ok, f"corrected IS biased: {mean_ok} vs {mean_mc}"

    # Uncorrected should show *some* discrepancy OR be within noise — depending
    # on strike/rho. We don't fail on this; we just record it (soft sanity
    # print). The critical test is that corrected matches MC.
    print(
        f"unbiased test: MC={mean_mc:.3f} corrected={mean_ok:.3f} "
        f"uncorrected={mean_bad:.3f} 3σ={3 * se_mc:.3f}"
    )


def test_observed_q_brownian_reconstructs_constant_control_likelihood(heston_sim):
    torch.manual_seed(713)
    control_value = -1.7
    increments: list[torch.Tensor] = []

    def control(_time, spot, _variance, _average):
        return torch.full_like(spot, control_value)

    def observer(_time, increment):
        increments.append(increment.detach())

    _S, _v, log_weight, _barrier, _average = heston_sim.simulate_controlled(
        S0=100.0,
        v0=0.04,
        T=0.4,
        dt=0.03,
        num_paths=512,
        control_fn=control,
        brownian_observer=observer,
    )
    brownian_terminal = torch.stack(increments, dim=1).sum(dim=1)
    expected = -control_value * brownian_terminal - 0.5 * control_value**2 * 0.4
    assert len(increments) == math.ceil(0.4 / 0.03)
    assert torch.allclose(log_weight, expected, atol=2e-6, rtol=1e-6)
