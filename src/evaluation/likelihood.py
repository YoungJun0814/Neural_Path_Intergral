"""Numerically stable likelihood and estimator-contribution diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


@dataclass(frozen=True)
class LikelihoodDiagnostics:
    paths: int
    finite_fraction: float
    log_mean_likelihood: float
    mean_likelihood: float
    normalization_z_score: float
    likelihood_ess: float
    likelihood_ess_fraction: float
    max_normalized_weight: float
    top_one_percent_weight_share: float
    estimate: float | None
    contribution_ess: float | None
    contribution_ess_fraction: float | None
    max_contribution_share: float | None
    top_one_percent_contribution_share: float | None


def _share_of_largest(values: np.ndarray, fraction: float) -> float:
    if values.size == 0:
        return math.nan
    count = max(1, int(math.ceil(fraction * values.size)))
    if count == values.size:
        return 1.0
    partition = np.partition(values, values.size - count)
    return float(partition[-count:].sum() / values.sum())


def likelihood_diagnostics(
    log_weights: np.ndarray,
    payoff: np.ndarray | None = None,
) -> LikelihoodDiagnostics:
    """Diagnose ``dP/dQ`` and optional nonnegative estimator contributions.

    All normalization, ESS, and concentration calculations are performed after
    a log-domain shift.  ``normalization_z_score`` tests ``E_Q[dP/dQ] = 1``;
    contribution diagnostics reveal whether a small number of paths dominate
    a rare-event estimate even when weight-only ESS appears acceptable.
    """
    logs = np.asarray(log_weights, dtype=np.float64)
    if logs.ndim != 1 or logs.size < 2:
        raise ValueError("log_weights must be a one-dimensional array with at least two paths")
    finite = np.isfinite(logs)
    finite_fraction = float(finite.mean())
    if not finite.all():
        raise ValueError("all log_weights must be finite")

    n = logs.size
    shift = float(logs.max())
    scaled = np.exp(logs - shift)
    scaled_sum = float(scaled.sum())
    normalized = scaled / scaled_sum
    log_mean = float(logsumexp(logs) - math.log(n))
    if log_mean > math.log(np.finfo(np.float64).max):
        mean_likelihood = math.inf
    elif log_mean < math.log(np.nextafter(0.0, 1.0)):
        mean_likelihood = 0.0
    else:
        mean_likelihood = math.exp(log_mean)

    scaled_mean = float(scaled.mean())
    scaled_se = float(scaled.std(ddof=1) / math.sqrt(n))
    if shift >= 710.0:
        target_in_scaled_units = 0.0
    elif shift <= -710.0:
        target_in_scaled_units = math.inf
    else:
        target_in_scaled_units = math.exp(-shift)
    normalization_z = (
        (scaled_mean - target_in_scaled_units) / scaled_se
        if scaled_se > 0.0
        else (
            0.0
            if scaled_mean == target_in_scaled_units
            else math.copysign(math.inf, scaled_mean - target_in_scaled_units)
        )
    )
    likelihood_ess = float(1.0 / np.sum(normalized**2))

    estimate: float | None = None
    contribution_ess: float | None = None
    contribution_ess_fraction: float | None = None
    max_contribution_share: float | None = None
    top_contribution_share: float | None = None
    if payoff is not None:
        payoff_array = np.asarray(payoff, dtype=np.float64)
        if payoff_array.shape != logs.shape:
            raise ValueError("payoff must match log_weights shape")
        if not np.isfinite(payoff_array).all() or np.any(payoff_array < 0.0):
            raise ValueError("payoff must be finite and nonnegative")

        positive = payoff_array > 0.0
        if not positive.any():
            estimate = 0.0
            contribution_ess = 0.0
            contribution_ess_fraction = 0.0
            max_contribution_share = 0.0
            top_contribution_share = 0.0
        else:
            log_contributions = logs[positive] + np.log(payoff_array[positive])
            contribution_shift = float(log_contributions.max())
            scaled_contributions = np.exp(log_contributions - contribution_shift)
            normalized_contributions = scaled_contributions / scaled_contributions.sum()
            log_estimate = float(logsumexp(log_contributions) - math.log(n))
            estimate = math.exp(log_estimate) if log_estimate < 710.0 else math.inf
            contribution_ess = float(1.0 / np.sum(normalized_contributions**2))
            contribution_ess_fraction = contribution_ess / n
            max_contribution_share = float(normalized_contributions.max())
            top_contribution_share = _share_of_largest(normalized_contributions, 0.01)

    return LikelihoodDiagnostics(
        paths=n,
        finite_fraction=finite_fraction,
        log_mean_likelihood=log_mean,
        mean_likelihood=mean_likelihood,
        normalization_z_score=float(normalization_z),
        likelihood_ess=likelihood_ess,
        likelihood_ess_fraction=likelihood_ess / n,
        max_normalized_weight=float(normalized.max()),
        top_one_percent_weight_share=_share_of_largest(normalized, 0.01),
        estimate=estimate,
        contribution_ess=contribution_ess,
        contribution_ess_fraction=contribution_ess_fraction,
        max_contribution_share=max_contribution_share,
        top_one_percent_contribution_share=top_contribution_share,
    )
