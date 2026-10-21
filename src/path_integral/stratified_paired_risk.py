"""Log-stable positive cross-moments with correct known-stratum covariance.

Intervals are asymptotic development diagnostics, never tail certificates.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any

import torch


def sufficient(log_values: torch.Tensor) -> dict[str, Any]:
    if (log_values.ndim != 2 or len(log_values) < 2 or log_values.shape[1] < 1
            or log_values.dtype != torch.float64 or log_values.device.type != "cpu"
            or log_values.requires_grad or not torch.isfinite(log_values).all()):
        raise ValueError("finite detached CPU float64 n-by-target log values required")
    # Pairwise logsumexp avoids underflow of cross-products with disjoint maxima.
    k = log_values.shape[1]
    cross = torch.empty((k, k), dtype=torch.float64)
    for a in range(k):
        for b in range(a, k):
            cross[a, b] = cross[b, a] = torch.logsumexp(log_values[:, a]+log_values[:, b], 0)
    return {"count": len(log_values), "log_sum": torch.logsumexp(log_values, 0).tolist(),
        "log_cross_sum": cross.tolist(), "log_maximum": log_values.max(0).values.tolist()}


def merge_sufficient(parts: list[dict[str, Any]]) -> dict[str, Any]:
    if not parts:
        raise ValueError("complete moment parts required")
    for p in parts:
        validate_sufficient(p)
    return {"count": sum(p["count"] for p in parts),
        "log_sum": torch.logsumexp(torch.tensor([p["log_sum"] for p in parts], dtype=torch.float64), 0).tolist(),
        "log_cross_sum": torch.logsumexp(torch.tensor([p["log_cross_sum"] for p in parts], dtype=torch.float64), 0).tolist(),
        "log_maximum": torch.tensor([p["log_maximum"] for p in parts], dtype=torch.float64).max(0).values.tolist()}


def validate_sufficient(part: dict[str, Any]) -> None:
    sums, cross, maximum = [torch.tensor(part[k], dtype=torch.float64) for k in ("log_sum", "log_cross_sum", "log_maximum")]
    if (type(part["count"]) is not int or part["count"] < 2 or sums.ndim != 1 or len(sums) < 1
            or cross.shape != (len(sums), len(sums)) or maximum.shape != sums.shape
            or not all(torch.isfinite(x).all() for x in (sums, cross, maximum))
            or not torch.equal(cross, cross.T) or (maximum > sums+1e-12).any()):
        raise ValueError("invalid finite positive sufficient moments")


def estimate(strata: list[dict[str, Any]], masses: torch.Tensor) -> dict[str, Any]:
    if (not strata or masses.shape != (len(strata),) or masses.dtype != torch.float64
            or masses.device.type != "cpu" or not torch.isfinite(masses).all()
            or (masses <= 0).any() or abs(float(masses.sum())-1) > 1e-12):
        raise ValueError("complete fixed normalized stratum masses required")
    for s in strata:
        validate_sufficient(s)
    scales = torch.tensor([s["log_maximum"] for s in strata], dtype=torch.float64).max(0).values
    means, covariances, max_terms = [], [], []
    for s, mass in zip(strata, masses, strict=True):
        n = s["count"]
        mean = (torch.tensor(s["log_sum"], dtype=torch.float64)-scales).exp()/n
        second = (torch.tensor(s["log_cross_sum"], dtype=torch.float64)-scales[:, None]-scales[None, :]).exp()/n
        covariance = (second-torch.outer(mean, mean))/(n-1)
        if float(torch.linalg.eigvalsh(covariance).min()) < -1e-11:
            raise FloatingPointError("saved covariance is not PSD beyond roundoff")
        means.append(mean)
        covariances.append(covariance)
        max_terms.append(float(mass)*(torch.tensor(s["log_maximum"], dtype=torch.float64)-scales).exp()/n)
    m = masses @ torch.stack(means)
    covariance = (masses.square()[:, None, None]*torch.stack(covariances)).sum(0)
    if not torch.isfinite(m).all() or (m <= 0).any() or not torch.isfinite(covariance).all():
        raise FloatingPointError("unresolved positive estimator range")
    diagonal = covariance.diag().clamp_min(0)
    return {"count": sum(s["count"] for s in strata), "stratum_count": len(strata),
        "log_mean": (scales+m.log()).tolist(), "relative_se": (diagonal.sqrt()/m).tolist(),
        "maximum_fraction": (torch.stack(max_terms).max(0).values/m).tolist(),
        "log_scale": scales.tolist(), "scaled_mean": m.tolist(), "scaled_covariance_of_mean": covariance.tolist(),
        "stratum_scaled_means": torch.stack(means).tolist(),
        "se_contract": "known_mass_independent_stratum_covariance_not_pooled_iid"}


def paired(summary: dict[str, Any], baseline: int, candidate: int, *, comparisons: int) -> dict[str, Any]:
    if type(comparisons) is not int or comparisons < 1 or baseline == candidate:
        raise ValueError("distinct paired endpoints and fixed interval family required")
    means = torch.tensor(summary["scaled_mean"], dtype=torch.float64)
    cov = torch.tensor(summary["scaled_covariance_of_mean"], dtype=torch.float64)
    scales = summary["log_scale"]
    scale = max(scales[baseline], scales[candidate])
    factor = torch.tensor([math.exp(scales[i]-scale) for i in (baseline, candidate)], dtype=torch.float64)
    m = means[[baseline, candidate]]*factor
    c = cov[[baseline, candidate]][:, [baseline, candidate]]*torch.outer(factor, factor)
    if m[0] <= 0:
        raise FloatingPointError("unresolved paired baseline")
    diff = float(m[0]-m[1])
    relative = diff/float(m[0])
    direction = torch.tensor([1., -1.], dtype=torch.float64)
    influence = torch.tensor([1-relative, -1.], dtype=torch.float64)
    absolute_se = math.sqrt(max(0., float(direction @ c @ direction)))
    relative_se = math.sqrt(max(0., float(influence @ c @ influence)))/float(m[0])
    z = -NormalDist().inv_cdf(.05/(2*comparisons))
    return {"relative_reduction": relative, "relative_se": relative_se,
        "relative_lower": relative-z*relative_se, "relative_upper": relative+z*relative_se,
        "scaled_difference": diff, "scaled_difference_se": absolute_se,
        "absolute_lower": diff-z*absolute_se, "absolute_upper": diff+z*absolute_se,
        "log_scale": scale, "confidence_z": z, "interval_family": comparisons,
        "positive_signal": relative-z*relative_se > 0 and diff-z*absolute_se > 0,
        "negative_signal": relative+z*relative_se < 0 and diff+z*absolute_se < 0,
        "status": "asymptotic_development_not_finite_sample_coverage"}


def independent_contrast(a: dict[str, Any], b: dict[str, Any], *, target: int,
                         comparisons: int = 24) -> dict[str, Any]:
    la, lb = a["log_mean"][target], b["log_mean"][target]
    scale = max(la, lb)
    x, y = math.exp(la-scale), math.exp(lb-scale)
    d = 2*(x-y)/(x+y)
    se = 4*x*y/(x+y)**2*math.hypot(a["relative_se"][target], b["relative_se"][target])
    z = -NormalDist().inv_cdf(.05/(2*comparisons))
    lo, hi = d-z*se, d+z*se
    return {"symmetric_difference": d, "se": se, "lower": lo, "upper": hi,
        "contradiction": lo > .2 or hi < -.2, "equivalence_claim": False,
        "status": "independent_design_asymptotic_development"}
