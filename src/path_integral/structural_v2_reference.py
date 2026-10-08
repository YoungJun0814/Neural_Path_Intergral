"""Finite-grid reference design and corroboration; not tail-coverage certificates."""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any

import torch

from src.path_integral.r1_bottleneck_diagnostics import summarize_log_contributions


def exact_smc_work(spec: dict[str, Any]) -> int:
    """Current weighted SMC mutates all but the final bridge transition."""
    for key, lower in (("particles", 2), ("levels", 2), ("mutation_steps", 0)):
        value = spec[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < lower:
            raise ValueError("invalid fixed SMC work specification")
    return spec["particles"] * (1 + (spec["levels"] - 2) * spec["mutation_steps"])


def log_moments(values: torch.Tensor) -> dict[str, Any]:
    """Sufficient moments and concentration; all logs must be finite."""
    if values.device.type != "cpu" or not torch.isfinite(values).all():
        raise ValueError("reference units must have finite CPU log contributions")
    summary = summarize_log_contributions(values)
    if summary.log_mean is None or summary.relative_se is None:
        raise FloatingPointError("zero reference mass")
    n = values.numel()
    ordered = torch.sort(values, descending=True).values
    total = torch.logsumexp(values, dim=0)
    result = dict(summary.__dict__)
    result.update(log_sum=float(total), log_sum_squares=float(torch.logsumexp(2 * values, dim=0)),
                  log_maximum=float(ordered[0]),
                  top_one_percent_share=float(torch.exp(torch.logsumexp(
                      ordered[:max(1, math.ceil(.01 * n))], dim=0) - total)))
    return result


def merge_moments(parts: list[dict[str, Any]]) -> dict[str, Any]:
    if not parts or any(p["count"] < 2 for p in parts):
        raise ValueError("missing reference moments")
    n = sum(p["count"] for p in parts)
    sums = torch.tensor([p["log_sum"] for p in parts], dtype=torch.float64)
    squares = torch.tensor([p["log_sum_squares"] for p in parts], dtype=torch.float64)
    if not torch.isfinite(sums).all() or not torch.isfinite(squares).all():
        raise ValueError("nonfinite saved moments")
    log_sum, log_squares = float(torch.logsumexp(sums, 0)), float(torch.logsumexp(squares, 0))
    log_mean, log_second = log_sum - math.log(n), log_squares - math.log(n)
    ratio = log_second - 2 * log_mean
    if ratio < -1e-10:
        raise ValueError("negative saved variance beyond roundoff")
    rse = math.sqrt(max(0., math.expm1(ratio)) / (n - 1))
    return {"count": n, "log_sum": log_sum, "log_sum_squares": log_squares,
            "log_mean": log_mean, "log_second_moment": log_second, "relative_se": rse,
            "maximum_fraction": math.exp(max(p["log_maximum"] for p in parts) - log_sum)}


def sensitivity(log_unit_means: list[float], *, bootstrap_seed: int,
                bootstrap_replicates: int = 2000) -> dict[str, Any]:
    """Whole SMC runs or equal-size independent IID block means, not particles."""
    logs = torch.tensor(log_unit_means, dtype=torch.float64)
    if logs.numel() < 2 or not torch.isfinite(logs).all():
        raise ValueError("insufficient independent units for sensitivity")
    if isinstance(bootstrap_replicates, bool) or bootstrap_replicates < 2:
        raise ValueError("invalid bootstrap count")
    scaled = torch.exp(logs - torch.max(logs))
    n, total = len(logs), float(scaled.sum())
    mean = total / n
    loo = (total - scaled) / (n - 1)
    generator = torch.Generator().manual_seed(bootstrap_seed)
    indices = torch.randint(n, (bootstrap_replicates, n), generator=generator)
    boot = scaled[indices].mean(1) / mean
    q = torch.quantile(boot, torch.tensor((.025, .975), dtype=torch.float64))
    return {"unit_count": n, "maximum_unit_contribution_fraction": float(scaled.max()) / total,
            "maximum_leave_one_out_relative_shift": float(torch.max(torch.abs(loo / mean - 1))),
            "between_unit_relative_se": float(scaled.std(unbiased=True)) / math.sqrt(n) / mean,
            "bootstrap_relative_mean_interval": q.tolist(),
            "bootstrap_replicates": bootstrap_replicates,
            "coverage_claim": "finite_sample_bootstrap_sensitivity_not_tail_certificate"}


def relative_equivalence(a: dict[str, Any], b: dict[str, Any], *, comparisons: int,
                         alpha: float = .05, margin: float = .10) -> dict[str, Any]:
    """Symmetric relative delta-method CI for independent reference streams."""
    if isinstance(comparisons, bool) or comparisons < 1 or not 0 < alpha < 1 or not 0 < margin < 2:
        raise ValueError("invalid family-wise endpoint")
    la, lb, ra, rb = (a["log_mean"], b["log_mean"], a["relative_se"], b["relative_se"])
    if any(not math.isfinite(x) for x in (la, lb, ra, rb)) or min(ra, rb) < 0:
        raise ValueError("invalid positive reference summary")
    scale = max(la, lb)
    x, y = math.exp(la - scale), math.exp(lb - scale)
    if min(x, y) == 0:
        return {"pass": False, "status": "unresolved_numerical_range"}
    difference = 2 * (x - y) / (x + y)
    se = 4 * x * y * math.hypot(ra, rb) / (x + y)**2
    z = NormalDist().inv_cdf(1 - alpha / (2 * comparisons))
    upper = abs(difference) + z * se
    return {"symmetric_relative_difference": difference, "standard_error": se,
            "confidence_z": z, "upper_absolute_difference": upper, "margin": margin,
            "pass": upper <= margin, "covariance": 0.,
            "status": "development_delta_method_not_distribution_free"}


def precision_count(pilot: dict[str, Any], *, target_rse: float, safety_factor: float,
                    minimum: int, maximum: int, multiple: int = 1) -> dict[str, Any]:
    if (not 0 < target_rse < 1 or not math.isfinite(safety_factor) or safety_factor < 1
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 1
                   for v in (minimum, maximum, multiple)) or maximum < minimum):
        raise ValueError("invalid allocation limits")
    rse, count = pilot["relative_se"], pilot["count"]
    if not math.isfinite(rse) or rse < 0 or isinstance(count, bool) or count < 2:
        raise ValueError("invalid pilot")
    required = max(minimum, math.ceil(safety_factor * count * rse**2 / target_rse**2 / multiple) * multiple)
    return {"required_count": required, "maximum_count": maximum,
            "status": "allocated" if required <= maximum else "unresolved_sample_budget",
            "target_relative_se": target_rse, "safety_factor": safety_factor,
            "interpretation": "pilot_forecast_not_precision_guarantee"}
