"""Frozen-q descriptive rules and known-stratum nonnegative ratio statistics.

Zero indicators are exact zeros; normal/delta intervals are development only.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any

import torch

FEATURES = ("log_integrated_variance", "log_peak_variance_ratio", "peak_time",
    "variance_hhi", "variance_time_center", "single_template_residual_fraction")


def encode(tensor: torch.Tensor) -> Any:
    if tensor.ndim:
        return [encode(v) for v in tensor]
    value = float(tensor)
    return None if value == -math.inf else value


def decode(value: Any) -> torch.Tensor:
    def replace(v: Any) -> Any:
        return [replace(x) for x in v] if isinstance(v, list) else -math.inf if v is None else v
    return torch.tensor(replace(value), dtype=torch.float64)


def moments(logs: torch.Tensor) -> dict[str, Any]:
    if (logs.ndim != 2 or len(logs) < 2 or logs.shape[1] < 1 or logs.dtype != torch.float64
            or logs.device.type != "cpu" or logs.requires_grad or torch.isnan(logs).any()
            or torch.isposinf(logs).any()):
        raise ValueError("detached CPU float64 logs with exact log-zero allowed required")
    size = logs.shape[1]
    cross = torch.empty((size, size), dtype=torch.float64)
    for a in range(size):
        for b in range(a, size):
            cross[a, b] = cross[b, a] = torch.logsumexp(logs[:, a]+logs[:, b], 0)
    return {"count": len(logs), "log_sum": encode(torch.logsumexp(logs, 0)),
        "log_cross_sum": encode(cross), "log_max": encode(logs.max(0).values)}


def merge(parts: list[dict[str, Any]]) -> dict[str, Any]:
    if not parts:
        raise ValueError("nonempty complete parts required")
    return {"count": sum(p["count"] for p in parts),
        "log_sum": encode(torch.logsumexp(torch.stack([decode(p["log_sum"]) for p in parts]), 0)),
        "log_cross_sum": encode(torch.logsumexp(torch.stack([decode(p["log_cross_sum"]) for p in parts]), 0)),
        "log_max": encode(torch.stack([decode(p["log_max"]) for p in parts]).max(0).values)}


def estimate(parts: list[dict[str, Any]], masses: torch.Tensor) -> dict[str, Any]:
    if (not parts or masses.dtype != torch.float64 or masses.device.type != "cpu"
            or masses.shape != (len(parts),) or not torch.isfinite(masses).all()
            or (masses <= 0).any() or abs(float(masses.sum())-1) > 1e-12):
        raise ValueError("positive normalized known stratum masses required")
    scale = torch.stack([decode(p["log_max"]) for p in parts]).max(0).values
    scale = torch.where(torch.isneginf(scale), 0., scale)
    means, covariances, largest = [], [], []
    for p, mass in zip(parts, masses, strict=True):
        n = p["count"]
        if type(n) is not int or n < 2:
            raise ValueError("at least two rows per stratum")
        sums, cross, maximum = [decode(p[k]) for k in ("log_sum", "log_cross_sum", "log_max")]
        if (sums.shape != scale.shape or maximum.shape != scale.shape or cross.shape != (len(scale), len(scale))
                or not torch.equal(cross, cross.T) or any(torch.isnan(v).any() or torch.isposinf(v).any() for v in (sums, cross, maximum))):
            raise ValueError("malformed zero-aware moments")
        mean = (sums-scale).exp()/n
        covariance = ((cross-scale[:, None]-scale[None, :]).exp()/n-torch.outer(mean, mean))/(n-1)
        if float(torch.linalg.eigvalsh(covariance).min()) < -1e-11:
            raise ValueError("non-PSD moment covariance")
        means.append(mean)
        covariances.append(covariance)
        largest.append(mass*(maximum-scale).exp()/n)
    mean = masses @ torch.stack(means)
    covariance = (masses.square()[:, None, None]*torch.stack(covariances)).sum(0)
    return {"count": sum(p["count"] for p in parts), "scale": scale.tolist(), "mean": mean.tolist(),
        "covariance": covariance.tolist(), "maximum_term": torch.stack(largest).max(0).values.tolist(),
        "strata": len(parts)}


def contrasts(summary: dict[str, Any], q: int) -> dict[str, Any]:
    """Columns Y,HU,HE,I,YI,HUI,HEI; scale back only ratio pairs."""
    if q not in (0, 1):
        raise ValueError("fixed two-q family required")
    scale = torch.tensor(summary["scale"], dtype=torch.float64)
    mean = torch.tensor(summary["mean"], dtype=torch.float64)
    cov = torch.tensor(summary["covariance"], dtype=torch.float64)
    if mean.shape != (7,) or cov.shape != (7, 7) or mean[0] <= 0 or mean[1+q] <= 0:
        raise ValueError("positive event and risk denominator required")

    def ratio(num: int, den: int) -> tuple[float, torch.Tensor]:
        if mean[num] == 0:
            # An empirical all-zero column has zero empirical covariance too.
            # Avoid exp(1000)*0; this is not a bound on unobserved tail mass.
            return 0., torch.zeros(7, dtype=torch.float64)
        factor = math.exp(float(scale[num]-scale[den]))
        value = float(mean[num]/mean[den])*factor
        grad = torch.zeros(7, dtype=torch.float64)
        grad[num], grad[den] = factor/mean[den], -value/mean[den]
        return value, grad
    tau, gt = ratio(5+q, 1+q)
    event, ge = ratio(4, 0)
    base = float(mean[3]*scale[3].exp())
    gb = torch.zeros(7, dtype=torch.float64)
    gb[3] = scale[3].exp()
    z = -NormalDist().inv_cdf(.05/(2*24))
    intervals: list[dict[str, Any]] = []
    for name, value, grad in (("risk_mass_minus_020", tau-.2, gt),
            ("risk_mass_minus_twice_G_mass", tau-2*base, gt-2*gb),
            ("risk_mass_minus_event_mass", tau-event, gt-ge)):
        se = math.sqrt(max(0., float(grad @ cov @ grad)))
        intervals.append({"name": name, "value": value, "se": se, "lower": value-z*se, "upper": value+z*se})
    return {"risk_mass": tau, "G_mass": base, "event_mass": event,
        "enrichment": tau/base if base else None, "intervals": intervals,
        "all_lower_positive": all(i["lower"] > 0 for i in intervals),
        "z": z, "coverage": "asymptotic_development_not_tail_certificate"}


def candidate_rules(features: dict[str, torch.Tensor], logs: dict[str, torch.Tensor]) -> dict[str, Any]:
    if len(features) != 2 or set(features) != set(logs):
        raise ValueError("complete two-cell discovery required")
    for c, values in features.items():
        if (values.ndim != 2 or values.shape[1] != 6 or len(values) != len(logs[c])
                or values.dtype != torch.float64 or not torch.isfinite(values).all()
                or logs[c].shape != (len(values), 3) or not torch.isfinite(logs[c]).all()):
            raise ValueError("six finite features and three event/risk columns required")
    rows: list[dict[str, Any]] = []
    for j, name in enumerate(FEATURES):
        for direction, quantile in (("low", .1), ("high", .9)):
            cells: dict[str, Any] = {}
            for cell, values in features.items():
                cut = float(torch.quantile(values[:, j], quantile))
                indicator = values[:, j] <= cut if direction == "low" else values[:, j] >= cut
                base = float(indicator.double().mean())
                masses = [float((torch.logsumexp(column[indicator], 0)-torch.logsumexp(column, 0)).exp())
                    for column in logs[cell].T]
                cells[cell] = {"cut": cut, "G_mass": base, "event_mass": masses[0],
                    "risk_mass": masses[1:], "enrichment": [v/base for v in masses[1:]]}
            eligible = all(.02 <= v["G_mass"] <= .30 and v["risk_mass"][0] >= .2 and v["enrichment"][0] >= 2 for v in cells.values())
            rows.append({"feature": name, "index": j, "direction": direction, "cells": cells,
                "eligible": eligible, "score": min(v["enrichment"][0] for v in cells.values())})
    eligible_rows = [r for r in rows if r["eligible"]]
    selected = max(eligible_rows, key=lambda r: r["score"]) if eligible_rows else None
    return {"candidates": rows, "selected": selected, "selection_is_discovery_only": True}


def rule_mask(values: torch.Tensor, rule: dict[str, Any], cell: str) -> torch.Tensor:
    j, cut = rule["index"], rule["cells"][cell]["cut"]
    if rule["feature"] != FEATURES[j] or rule["direction"] not in ("low", "high") or not math.isfinite(cut):
        raise ValueError("frozen rule identity required")
    return values[:, j] <= cut if rule["direction"] == "low" else values[:, j] >= cut


def diagnostic_logs(logs: torch.Tensor, indicator: torch.Tensor) -> torch.Tensor:
    if logs.ndim != 2 or logs.shape[1] != 3 or indicator.dtype != torch.bool or indicator.shape != (len(logs),):
        raise ValueError("Y/HU/HE and exact boolean rule required")
    mask = torch.where(indicator, 0., -math.inf).double()
    return torch.cat((logs, mask[:, None], logs+mask[:, None]), 1)
