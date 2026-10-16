"""Ordinary full-mixture IS with fixed known Gaussian component strata.

This is not self-normalization or empirical normalization. SE uses independent
stratum sample variances, not a fictitious IID pooled-draw sample variance.
"""

from __future__ import annotations

import math
from typing import Any

import torch


def proportional_guide_counts(weights: torch.Tensor, *, multiplier: int) -> tuple[int, ...]:
    """Exact allocation for .1 natural + uniform .9 shifted components.

For K shifted components, n0=K*m and nj=9*m, total=10*K*m.
"""
    if (weights.ndim != 1 or len(weights) < 2 or weights.dtype != torch.float64
            or weights.device.type != "cpu" or not torch.isfinite(weights).all()
            or isinstance(multiplier, bool) or not isinstance(multiplier, int) or multiplier < 1):
        raise ValueError("declared static-guide weights and positive multiplier required")
    k = len(weights)-1
    expected = torch.full_like(weights, .9/k)
    expected[0] = .1
    if not torch.allclose(weights, expected, rtol=0, atol=1e-13):
        raise ValueError("requires the frozen .1 / uniform .9 guide")
    return (k*multiplier,)+(9*multiplier,)*k


def stratified_summary(strata: list[dict[str, Any]], log_weights: torch.Tensor) -> dict[str, Any]:
    """Combine unbiased stratum means and their independently estimated SEs."""
    if (log_weights.shape != (len(strata),) or not strata or log_weights.dtype != torch.float64
            or log_weights.device.type != "cpu" or not torch.isfinite(log_weights).all()
            or abs(float(torch.logsumexp(log_weights, 0))) > 1e-12
            or any(s["count"] < 2 or not math.isfinite(s["log_mean"])
                   or not math.isfinite(s["relative_se"]) or s["relative_se"] < 0
                   or not 0 < s["maximum_fraction"] <= 1+1e-12 for s in strata)):
        raise ValueError("complete positive stratum moments and fixed normalized weights required")
    logs = torch.tensor([s["log_mean"] for s in strata], dtype=torch.float64)
    log_mean = torch.logsumexp(log_weights+logs, 0)
    fractions = (log_weights+logs-log_mean).exp()
    variance = sum(float(fractions[j])**2*s["relative_se"]**2 for j, s in enumerate(strata))
    maximum = max(float(fractions[j])*s["maximum_fraction"] for j, s in enumerate(strata))
    return {"count": sum(s["count"] for s in strata), "log_mean": float(log_mean),
            "relative_se": math.sqrt(variance), "maximum_fraction": maximum,
            "se_contract": "independent_fixed_stratum_variances_not_pooled_iid_variance"}
