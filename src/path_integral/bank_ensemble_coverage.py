"""Fixed bank ensembles and all-sample finite-grid physical-feature ledgers.

Observed feature ratios and empty bins are not unseen-tail/coverage certificates.
Probability estimates remain ordinary full-mixture importance-sampling means.
"""

from __future__ import annotations

import math
from typing import Any

import torch

FEATURES = ("log_integrated_variance", "log_point_concentration", "maximum_window_T32_energy_fraction", "peak_time_fraction")
BIN_COUNT = 96


def mean_defensive_weights(weights: list[torch.Tensor]) -> torch.Tensor:
    if len(weights) < 2:
        raise ValueError("at least two fixed independent-bank weight vectors required")
    for w in weights:
        if (w.ndim != 1 or len(w) < 2 or w.shape != weights[0].shape or w.dtype != torch.float64
                or w.device.type != "cpu" or w.requires_grad or not torch.isfinite(w).all()
                or abs(float(w.sum())-1) > 1e-12 or abs(float(w[0])-.1) > 1e-12
                or (w[1:] < .1/(len(w)-1)-1e-14).any()):
            raise ValueError("same strictly defensive fixed dictionary weights required")
    return torch.stack(weights).mean(0)


def physical_path_features(left_variance: torch.Tensor, *, maturity: float) -> torch.Tensor:
    if (left_variance.ndim != 2 or len(left_variance) < 1 or left_variance.shape[1] < 32
            or left_variance.shape[1] % 32 or left_variance.dtype != torch.float64
            or left_variance.device.type != "cpu" or left_variance.requires_grad
            or not torch.isfinite(left_variance).all() or (left_variance <= 0).any()
            or not math.isfinite(maturity) or maturity <= 0):
        raise ValueError("positive finite CPU float64 left-variance paths and maturity required")
    n = left_variance.shape[1]
    total = left_variance.sum(1)
    peak, index = left_variance.max(1)
    prefix = torch.cat((torch.zeros((len(left_variance), 1), dtype=torch.float64), left_variance.cumsum(1)), 1)
    width = n//32
    maximum_window = (prefix[:, width:]-prefix[:, :-width]).max(1).values
    concentration, fraction = n*peak/total, maximum_window/total
    features = torch.stack(((total*(maturity/n)).log(), concentration.log(), fraction, index.to(torch.float64)/n), 1)
    if (not torch.isfinite(features).all() or (concentration < 1-1e-12).any()
            or (fraction < 0).any() or (fraction > 1+1e-12).any()):
        raise FloatingPointError("physical path-feature identity/range failed")
    return features


def path_bins(features: torch.Tensor) -> torch.Tensor:
    if (features.ndim != 2 or features.shape[1] != 4 or len(features) < 1
            or features.dtype != torch.float64 or features.device.type != "cpu"
            or features.requires_grad or not torch.isfinite(features).all()
            or (features[:, 1] < -1e-12).any() or (features[:, 2] < 0).any()
            or (features[:, 2] > 1+1e-12).any() or (features[:, 3] < 0).any()
            or (features[:, 3] >= 1).any()):
        raise ValueError("valid finite-grid feature rows required")
    boundaries = torch.tensor([.25, .5, .75], dtype=torch.float64)
    intensity = torch.bucketize(features[:, 0].contiguous(), torch.tensor([.25, .5, 1., 2., 4.], dtype=torch.float64).log(), right=True)
    time = torch.bucketize(features[:, 3].contiguous(), boundaries, right=True)
    window = torch.bucketize(features[:, 2].contiguous(), boundaries, right=True)
    return (intensity*4+time)*4+window


def coverage_ledger(log_values: torch.Tensor, features: torch.Tensor) -> dict[str, Any]:
    bins = path_bins(features)
    if (log_values.shape != (len(features),) or log_values.dtype != torch.float64
            or log_values.device.type != "cpu" or log_values.requires_grad or not torch.isfinite(log_values).all()):
        raise ValueError("finite ordinary log contributions for every feature row required")
    counts = torch.bincount(bins, minlength=BIN_COUNT).tolist()
    sums: list[float | None] = [None]*BIN_COUNT
    squares: list[float | None] = [None]*BIN_COUNT
    for b in bins.unique().tolist():
        values = log_values[bins == b]
        sums[b] = float(torch.logsumexp(values, 0))
        squares[b] = float(torch.logsumexp(2*values, 0))
    total = float(torch.logsumexp(log_values, 0))
    top = torch.argsort(log_values, descending=True, stable=True)[:16]
    return {"count": len(log_values), "bin_counts": counts, "bin_log_sums": sums,
            "bin_log_sum_squares": squares, "log_sum": total,
            "log_sum_squares": float(torch.logsumexp(2*log_values, 0)),
            "observed_weighted_feature_means": ((log_values-total).exp()[:, None]*features).sum(0).tolist(),
            "top": [{"log_contribution": float(log_values[i]), "bin": int(bins[i]), "features": features[i].tolist()} for i in top],
            "feature_names": list(FEATURES), "scope": "all_sample_fixed_bins_observed_only_not_population_coverage"}


def _log_merge(values: list[float | None]) -> float | None:
    finite = [v for v in values if v is not None]
    return float(torch.logsumexp(torch.tensor(finite, dtype=torch.float64), 0)) if finite else None


def merge_coverage(parts: list[dict[str, Any]]) -> dict[str, Any]:
    if not parts or any(p["count"] < 1 or p["feature_names"] != list(FEATURES)
                        or len(p["bin_counts"]) != BIN_COUNT for p in parts):
        raise ValueError("complete identical feature ledgers required")
    counts = [sum(p["bin_counts"][b] for p in parts) for b in range(BIN_COUNT)]
    sums = [_log_merge([p["bin_log_sums"][b] for p in parts]) for b in range(BIN_COUNT)]
    squares = [_log_merge([p["bin_log_sum_squares"][b] for p in parts]) for b in range(BIN_COUNT)]
    total, total_squares = _log_merge([p["log_sum"] for p in parts]), _log_merge([p["log_sum_squares"] for p in parts])
    assert total is not None and total_squares is not None
    bin_total = _log_merge(sums)
    bin_squares = _log_merge(squares)
    if (sum(counts) != sum(p["count"] for p in parts) or bin_total is None or bin_squares is None
            or abs(bin_total-total) > 1e-10 or abs(bin_squares-total_squares) > 1e-10
            or any((count == 0 and (s is not None or ss is not None))
                   or (count > 0 and (s is None or ss is None))
                   for count, s, ss in zip(counts, sums, squares, strict=True))):
        raise ValueError("bin partition fails to reconstruct ordinary moments")
    masses = torch.tensor([math.exp(p["log_sum"]-total) for p in parts], dtype=torch.float64)
    means = torch.tensor([p["observed_weighted_feature_means"] for p in parts], dtype=torch.float64)
    return {**parts[0], "count": sum(counts), "bin_counts": counts, "bin_log_sums": sums,
            "bin_log_sum_squares": squares, "log_sum": total, "log_sum_squares": total_squares,
            "observed_weighted_feature_means": (masses[:, None]*means).sum(0).tolist(),
            "top": sorted((t for p in parts for t in p["top"]), key=lambda t: t["log_contribution"], reverse=True)[:16]}


def summarize_bins(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    ledger = merge_coverage([ledger])
    n = ledger["count"]
    if n < 2:
        raise ValueError("at least two total samples for bin uncertainty diagnostics")
    rows = []
    for b, (count, ls, lss) in enumerate(zip(ledger["bin_counts"], ledger["bin_log_sums"], ledger["bin_log_sum_squares"], strict=True)):
        i, remainder = divmod(b, 16)
        peak, window = divmod(remainder, 4)
        if ls is None or lss is None:
            rows.append({"bin": b, "intensity_bin": i, "time_bin": peak, "window_bin": window,
                         "sample_count": 0, "status": "unobserved_not_zero_probability",
                         "log_ordinary_mean": None, "relative_se": None})
            continue
        rows.append({"bin": b, "intensity_bin": i, "time_bin": peak, "window_bin": window,
            "sample_count": count, "status": "observed_finite_sample_only", "log_ordinary_mean": ls-math.log(n),
            "relative_se": math.sqrt(max(0., n*math.exp(lss-2*ls)-1)/(n-1)),
            "observed_contribution_fraction": math.exp(ls-ledger["log_sum"]),
            "observed_squared_contribution_fraction": math.exp(lss-ledger["log_sum_squares"])})
    return rows
