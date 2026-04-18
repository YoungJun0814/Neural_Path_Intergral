"""VaR backtesting and efficiency metrics.

References:
* Kupiec (1995) "Techniques for verifying the accuracy of risk measurement models."
* Christoffersen (1998) "Evaluating interval forecasts."
* L'Ecuyer (1994) "Efficiency improvement and variance reduction."

All functions are pure-numpy so they can be imported without a torch
dependency.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.stats import chi2


# -----------------------------------------------------------------------------
# 1.  VaR computation
# -----------------------------------------------------------------------------

def compute_var_series(
    returns_matrix: np.ndarray,
    alpha: float = 0.01,
) -> np.ndarray:
    """Given a (T, N) matrix of scenario returns per forecast date, return the
    1-step ahead VaR (negative of the α-quantile) of length T.
    """
    if returns_matrix.ndim != 2:
        raise ValueError("returns_matrix must be 2-dim (T, N)")
    q = np.quantile(returns_matrix, alpha, axis=1)
    return -q  # VaR is positive loss magnitude


def var_exceptions(realized: np.ndarray, var_series: np.ndarray) -> np.ndarray:
    """Return a boolean array where the realized return breaches the VaR."""
    if realized.shape != var_series.shape:
        raise ValueError(f"shape mismatch: realized {realized.shape} vs var {var_series.shape}")
    return -realized > var_series  # loss > VaR


# -----------------------------------------------------------------------------
# 2.  Coverage tests (Kupiec)
# -----------------------------------------------------------------------------

@dataclass
class CoverageTestResult:
    lr: float
    p_value: float
    x: int  # number of exceptions
    T: int  # total observations
    observed_rate: float
    expected_rate: float


def kupiec_pof_test(exceptions: np.ndarray, alpha: float = 0.01) -> CoverageTestResult:
    """Kupiec proportion-of-failures likelihood ratio test.

    H0: P(exception) = α. Rejects if the observed rate is materially different.
    Under H0, the LR statistic is χ²(1) distributed asymptotically.
    """
    T = int(exceptions.size)
    x = int(exceptions.sum())
    p_obs = x / T if T > 0 else 0.0
    # Avoid log(0) — add tiny epsilon
    eps = 1e-12
    log_lik_h1 = x * math.log(max(p_obs, eps)) + (T - x) * math.log(max(1 - p_obs, eps))
    log_lik_h0 = x * math.log(max(alpha, eps)) + (T - x) * math.log(max(1 - alpha, eps))
    lr = -2.0 * (log_lik_h0 - log_lik_h1)
    p_value = float(1.0 - chi2.cdf(lr, df=1))
    return CoverageTestResult(
        lr=lr, p_value=p_value, x=x, T=T, observed_rate=p_obs, expected_rate=float(alpha),
    )


# -----------------------------------------------------------------------------
# 3.  Independence test (Christoffersen)
# -----------------------------------------------------------------------------

@dataclass
class IndependenceTestResult:
    lr: float
    p_value: float
    n00: int
    n01: int
    n10: int
    n11: int


def christoffersen_independence(exceptions: np.ndarray) -> IndependenceTestResult:
    """Christoffersen (1998) Markov independence test.

    Tests whether exceptions cluster in time. LR ~ χ²(1) under H0 (iid).
    """
    x = exceptions.astype(int)
    n00 = int(((x[:-1] == 0) & (x[1:] == 0)).sum())
    n01 = int(((x[:-1] == 0) & (x[1:] == 1)).sum())
    n10 = int(((x[:-1] == 1) & (x[1:] == 0)).sum())
    n11 = int(((x[:-1] == 1) & (x[1:] == 1)).sum())
    eps = 1e-12
    # Transition probabilities
    p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    p_star = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    log_lik_h1 = (
        (n00) * math.log(max(1 - p01, eps))
        + n01 * math.log(max(p01, eps))
        + n10 * math.log(max(1 - p11, eps))
        + n11 * math.log(max(p11, eps))
    )
    log_lik_h0 = (
        (n00 + n10) * math.log(max(1 - p_star, eps))
        + (n01 + n11) * math.log(max(p_star, eps))
    )
    lr = -2.0 * (log_lik_h0 - log_lik_h1)
    p_value = float(1.0 - chi2.cdf(lr, df=1))
    return IndependenceTestResult(lr=lr, p_value=p_value, n00=n00, n01=n01, n10=n10, n11=n11)


# -----------------------------------------------------------------------------
# 4.  Efficiency metrics: VRF + ESS
# -----------------------------------------------------------------------------

@dataclass
class EfficiencyReport:
    var_mc: float
    var_is: float
    cost_mc: float
    cost_is: float
    vrf: float
    ess_is: float
    paths_is: int


def efficiency_metrics(
    *,
    estimates_mc: np.ndarray,
    estimates_is: np.ndarray,
    weights_is: np.ndarray,
    cost_mc: float = 1.0,
    cost_is: float = 1.0,
) -> EfficiencyReport:
    """Compute work-normalized Variance Reduction Factor (VRF) and ESS.

    ``estimates_mc`` : shape (N_mc,) sample of the raw MC estimator.
    ``estimates_is`` : shape (N_is,) sample of the IS re-weighted estimator.
    ``weights_is``  : shape (N_is,) the likelihood ratios E_T^{(i)}.
    """
    var_mc = float(np.var(estimates_mc, ddof=1))
    var_is = float(np.var(estimates_is, ddof=1))
    vrf = (var_mc / max(cost_mc, 1e-12)) / max(var_is / max(cost_is, 1e-12), 1e-12)
    ess = float(weights_is.sum() ** 2 / max((weights_is ** 2).sum(), 1e-12))
    return EfficiencyReport(
        var_mc=var_mc, var_is=var_is, cost_mc=cost_mc, cost_is=cost_is,
        vrf=vrf, ess_is=ess, paths_is=int(weights_is.size),
    )
