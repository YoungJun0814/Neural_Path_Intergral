"""rBergomi sanity checks: shape, finiteness, ATM skew sign.

We don't test the *magnitude* of the ATM skew (which depends on the mid-T
asymptotics ~ T^{H-1/2}); only sign and ordering. A full BLP scheme test
would require a 2-dim covariance check at the kernel level.
"""

from __future__ import annotations

import pytest
import torch

from src.physics_engine import RBergomiSimulator
from src.utils import set_seed


def test_rbergomi_shapes():
    set_seed(0)
    sim = RBergomiSimulator(H=0.1, eta=1.5, xi=0.04, rho=-0.7, device="cpu")
    S, V = sim.simulate(S0=100.0, T=0.5, dt=1 / 252.0, num_paths=64)
    assert S.shape == V.shape == (64, 127)
    assert torch.isfinite(S).all() and torch.isfinite(V).all()
    assert (S > 0).all() and (V > 0).all()


def test_rbergomi_rejects_H_outside_zero_half():
    with pytest.raises(ValueError):
        RBergomiSimulator(H=0.5, device="cpu")
    with pytest.raises(ValueError):
        RBergomiSimulator(H=0.0, device="cpu")


def test_rbergomi_lower_H_more_persistent():
    """Lower Hurst → vol-of-vol is more persistent → terminal V variance is
    larger across paths (compared to higher H, all else equal)."""
    set_seed(1)
    sim_low = RBergomiSimulator(H=0.05, eta=1.0, xi=0.04, rho=0.0, device="cpu")
    sim_high = RBergomiSimulator(H=0.45, eta=1.0, xi=0.04, rho=0.0, device="cpu")
    _, V_low = sim_low.simulate(S0=100.0, T=0.25, dt=1 / 252.0, num_paths=2000)
    _, V_high = sim_high.simulate(S0=100.0, T=0.25, dt=1 / 252.0, num_paths=2000)
    # Compare cross-path variance of log V at terminal time
    var_low = float(torch.log(V_low[:, -1]).var())
    var_high = float(torch.log(V_high[:, -1]).var())
    assert var_low > var_high, f"H=0.05 var={var_low}, H=0.45 var={var_high}"


def test_rbergomi_negative_rho_atm_skew_negative():
    """With ρ<0 the implied vol vs strike skew is negative — a coarse proxy:
    log returns should be left-skewed.

    Allow a sign-only test (sample size limits magnitude reliability).
    """
    set_seed(2)
    sim = RBergomiSimulator(H=0.1, eta=2.0, xi=0.04, rho=-0.9, device="cpu")
    S, _ = sim.simulate(S0=100.0, T=0.5, dt=1 / 252.0, num_paths=5000)
    log_ret_total = torch.log(S[:, -1] / S[:, 0])
    z = (log_ret_total - log_ret_total.mean()) / (log_ret_total.std() + 1e-8)
    skew = float((z**3).mean())
    assert skew < 0.0, f"ρ=−0.9 should give negative terminal skew; got {skew}"


def test_rbergomi_xi_scales_variance_level():
    """V(0) = ξ; doubling ξ should roughly double early-time variance level."""
    set_seed(3)
    sim_lo = RBergomiSimulator(H=0.1, eta=1.0, xi=0.02, rho=0.0, device="cpu")
    sim_hi = RBergomiSimulator(H=0.1, eta=1.0, xi=0.04, rho=0.0, device="cpu")
    _, V_lo = sim_lo.simulate(S0=100.0, T=0.05, dt=1 / 252.0, num_paths=2000)
    _, V_hi = sim_hi.simulate(S0=100.0, T=0.05, dt=1 / 252.0, num_paths=2000)
    ratio = float(V_hi.mean() / V_lo.mean())
    assert 1.7 < ratio < 2.3, f"V mean ratio={ratio}, expected ≈ 2"
