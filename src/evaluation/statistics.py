"""Repeated-run uncertainty summaries for frozen rare-event estimators."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class RepeatedEstimateReport:
    runs: int
    truth: float
    mean_estimate: float
    bias: float
    bias_z_score: float
    empirical_standard_deviation: float
    mean_reported_standard_error: float
    relative_bias: float
    relative_rmse: float
    confidence_level: float
    ci_coverage: float


def repeated_estimate_report(
    estimates: np.ndarray,
    standard_errors: np.ndarray,
    *,
    truth: float,
    confidence_level: float = 0.95,
) -> RepeatedEstimateReport:
    """Summarize independent estimator runs against a known reference value."""
    values = np.asarray(estimates, dtype=np.float64)
    errors = np.asarray(standard_errors, dtype=np.float64)
    if values.ndim != 1 or errors.shape != values.shape or values.size < 2:
        raise ValueError("estimates and standard_errors must be matching 1D arrays of length >= 2")
    if not np.isfinite(values).all() or not np.isfinite(errors).all():
        raise ValueError("estimates and standard_errors must be finite")
    if np.any(errors < 0.0):
        raise ValueError("standard_errors must be nonnegative")
    if truth <= 0.0:
        raise ValueError("truth must be positive for relative rare-event diagnostics")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")

    runs = values.size
    mean = float(values.mean())
    bias = mean - truth
    empirical_std = float(values.std(ddof=1))
    mean_se = float(errors.mean())
    standard_error_of_mean = empirical_std / math.sqrt(runs)
    if standard_error_of_mean > 0.0:
        bias_z = bias / standard_error_of_mean
    else:
        bias_z = 0.0 if bias == 0.0 else math.copysign(math.inf, bias)

    critical = float(norm.ppf(0.5 + confidence_level / 2.0))
    covered = (values - critical * errors <= truth) & (truth <= values + critical * errors)
    relative_rmse = float(np.sqrt(np.mean((values - truth) ** 2)) / truth)
    return RepeatedEstimateReport(
        runs=runs,
        truth=float(truth),
        mean_estimate=mean,
        bias=bias,
        bias_z_score=float(bias_z),
        empirical_standard_deviation=empirical_std,
        mean_reported_standard_error=mean_se,
        relative_bias=bias / truth,
        relative_rmse=relative_rmse,
        confidence_level=confidence_level,
        ci_coverage=float(covered.mean()),
    )
