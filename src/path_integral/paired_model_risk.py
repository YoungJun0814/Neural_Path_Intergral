"""Paired finite-grid M2 diagnostics on independent fresh auxiliary draws."""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any

import torch


def paired_risk_reduction(
    log_baseline: torch.Tensor,
    log_candidate: torch.Tensor,
    *,
    comparisons: int = 60,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Estimate relative M2 reduction, retaining shared-draw covariance.

    Both vectors must use the same independent fresh draws from a frozen G:
    ``log H_a = 2 log g - log(G/p) - log(q_a/p)``. Their expectations are
    M2(q_a). Pairing is across candidate proposals, not across independent
    physical draws. The caller must establish freshness and independence;
    tensor validation cannot establish either property.

    The Bonferroni-adjusted interval is an asymptotic delta-method development
    diagnostic, not a finite-sample tail, accuracy, or total-cost certificate.
    """
    vectors = (log_baseline, log_candidate)
    if any(
        not isinstance(value, torch.Tensor)
        or value.ndim != 1
        or value.numel() < 2
        or value.dtype != torch.float64
        or value.device.type != "cpu"
        or value.layout != torch.strided
        or value.requires_grad
        or not bool(torch.isfinite(value).all())
        for value in vectors
    ):
        raise ValueError("finite one-dimensional CPU float64 no-autograd vectors required")
    if log_baseline.shape != log_candidate.shape:
        raise ValueError("paired vectors must have the same sample count")
    if (
        isinstance(comparisons, bool)
        or not isinstance(comparisons, int)
        or comparisons < 1
        or isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(alpha)
        or not 0 < alpha < 1
    ):
        raise ValueError("positive comparison count and finite alpha in (0, 1) required")

    count = log_baseline.numel()
    scale = torch.maximum(log_baseline.max(), log_candidate.max())
    baseline = torch.exp(log_baseline - scale)
    candidate = torch.exp(log_candidate - scale)
    denominator = float(baseline.mean())
    if not math.isfinite(denominator) or denominator <= 0:
        raise FloatingPointError("unresolved paired baseline mean")

    difference = baseline - candidate
    relative = float(difference.mean()) / denominator
    influence = difference - relative * baseline
    standard_error = float(influence.std(unbiased=True)) / math.sqrt(count) / denominator

    # Evaluate the lower normal tail to avoid rounding 1 - tiny_tail to one.
    tail = math.exp(math.log(alpha) - math.log(2) - math.log(comparisons))
    if not 0 < tail < 0.5:
        raise FloatingPointError("unresolved Bonferroni normal quantile")
    confidence_z = -NormalDist().inv_cdf(tail)
    lower = relative - confidence_z * standard_error
    upper = relative + confidence_z * standard_error
    absolute_mean = float(difference.mean())
    absolute_se = float(difference.std(unbiased=True))/math.sqrt(count)
    absolute_lower = absolute_mean-confidence_z*absolute_se
    absolute_upper = absolute_mean+confidence_z*absolute_se
    if any(not math.isfinite(value) for value in (relative, standard_error, confidence_z, lower, upper,
            absolute_mean, absolute_se, absolute_lower, absolute_upper)):
        raise FloatingPointError("unresolved paired risk diagnostic numerical range")

    return {
        "sample_count": count,
        "relative_m2_reduction": relative,
        "paired_relative_se": standard_error,
        "confidence_z": confidence_z,
        "lower_relative_reduction": lower,
        "upper_relative_reduction": upper,
        "candidate_lower_m2_signal": lower > 0,
        "log_scale": float(scale),
        "scaled_mean_difference": absolute_mean,
        "scaled_difference_se": absolute_se,
        "scaled_difference_lower": absolute_lower,
        "scaled_difference_upper": absolute_upper,
        "candidate_lower_absolute_signal": absolute_lower > 0,
        "same_observed_values": torch.equal(log_baseline, log_candidate),
        "se_unit": "independent_fresh_G_draw",
        "status": "development_delta_method_not_tail_or_cost_certificate",
    }
