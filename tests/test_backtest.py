"""Tests for src.evaluation.backtest (Kupiec POF, Christoffersen independence, VRF)."""
from __future__ import annotations

import math

import numpy as np

from src.evaluation.backtest import (
    christoffersen_independence,
    compute_var_series,
    efficiency_metrics,
    kupiec_pof_test,
    var_exceptions,
)


def test_kupiec_correct_rate_high_pvalue():
    """When observed rate ≈ α, p-value should be high."""
    rng = np.random.default_rng(0)
    T = 5000
    exceptions = rng.random(T) < 0.01
    r = kupiec_pof_test(exceptions, alpha=0.01)
    assert r.p_value > 0.1, f"p={r.p_value}"


def test_kupiec_wrong_rate_low_pvalue():
    """Observed rate far from α should reject."""
    rng = np.random.default_rng(1)
    T = 5000
    exceptions = rng.random(T) < 0.05  # 5× the nominal
    r = kupiec_pof_test(exceptions, alpha=0.01)
    assert r.p_value < 0.01, f"p={r.p_value}"


def test_christoffersen_iid_passes():
    rng = np.random.default_rng(2)
    exceptions = rng.random(5000) < 0.02
    r = christoffersen_independence(exceptions)
    assert r.p_value > 0.01, f"p={r.p_value}"


def test_christoffersen_clustered_rejects():
    """Cluster all exceptions at the start — should reject independence."""
    T = 2000
    ex = np.zeros(T, dtype=bool)
    ex[:40] = True  # all 40 exceptions in a row
    r = christoffersen_independence(ex)
    assert r.p_value < 0.01, f"p={r.p_value}"


def test_var_computation():
    rng = np.random.default_rng(3)
    mat = rng.standard_normal((50, 5000)) * 0.01
    vars_ = compute_var_series(mat, alpha=0.01)
    assert vars_.shape == (50,)
    assert (vars_ > 0).all()  # 1% VaR on centered returns is positive


def test_var_exceptions_counts():
    mat = np.zeros((100, 1000))
    mat[10, :] = -0.05  # heavy loss day 10
    vars_ = compute_var_series(mat, alpha=0.01)
    # realized uses mean of sims minus a fixed daily realization; test by construction
    realized = np.full(100, 0.0)
    realized[10] = -0.10
    ex = var_exceptions(realized, vars_)
    assert ex[10]  # day 10 breached


def test_efficiency_metrics_basic():
    rng = np.random.default_rng(4)
    est_mc = rng.standard_normal(1000) * 1.0
    est_is = rng.standard_normal(1000) * 0.3
    w = np.exp(rng.standard_normal(1000) * 0.3)
    eff = efficiency_metrics(estimates_mc=est_mc, estimates_is=est_is, weights_is=w)
    assert eff.var_mc > eff.var_is
    assert eff.vrf > 1.0
    assert 0 < eff.ess_is <= 1000
