"""Fractional Brownian motion — empirical covariance matches analytic kernel."""
from __future__ import annotations

import math

import pytest
import torch

from src.physics_engine import FractionalBrownianMotion
from src.utils import set_seed


def fbm_cov_analytic(s: float, t: float, H: float) -> float:
    return 0.5 * (abs(s) ** (2 * H) + abs(t) ** (2 * H) - abs(s - t) ** (2 * H))


@pytest.mark.parametrize("H", [0.1, 0.3, 0.5, 0.7])
def test_fbm_empirical_covariance(H):
    set_seed(0)
    n_paths = 5000
    n_steps = 30
    dt = 0.02
    fbm = FractionalBrownianMotion(H=H, device="cpu")
    W = fbm.generate(n_paths, n_steps, dt)  # (paths, n_steps+1), W[:,0]=0

    # Sample empirical covariance at time t_i and t_j
    i, j = 10, 25
    t_i = (i) * dt
    t_j = (j) * dt
    cov_empirical = float((W[:, i] * W[:, j]).mean())
    cov_theory = fbm_cov_analytic(t_i, t_j, H)
    # 4σ CLT tolerance: cov of product terms is bounded by max cov
    tol = 5.0 * cov_theory / math.sqrt(n_paths) + 1e-3
    assert abs(cov_empirical - cov_theory) < tol + 0.01, (
        f"H={H}: emp={cov_empirical} theory={cov_theory}"
    )


def test_fbm_variance_scaling():
    """Var[W_H(t)] = t^{2H} — the simplest invariant."""
    set_seed(1)
    H = 0.3
    fbm = FractionalBrownianMotion(H=H, device="cpu")
    n_steps = 50
    dt = 1.0 / n_steps
    W = fbm.generate(3000, n_steps, dt)
    for i in (10, 25, 50):
        t_i = i * dt
        var_emp = float((W[:, i] ** 2).mean())
        var_th = t_i ** (2 * H)
        assert abs(var_emp - var_th) / (var_th + 1e-6) < 0.15


def test_fbm_H_rejects_extremes():
    with pytest.raises(ValueError):
        FractionalBrownianMotion(H=0.0, device="cpu")
    with pytest.raises(ValueError):
        FractionalBrownianMotion(H=1.0, device="cpu")
