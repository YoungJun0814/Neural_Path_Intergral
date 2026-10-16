"""Finite-grid risk potential and independent-pilot allocation (not a certificate)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch

from src.path_integral.r1_bottleneck_diagnostics import summarize_log_contributions


def log_risk_potential(
    log_g: torch.Tensor, log_q_over_p: torch.Tensor, *, defensive_mass: float,
) -> torch.Tensor:
    """log h = log(delta) + 2 log(g) - log(q/p); E_p[h] = delta M2(q).

    q must be frozen and contain delta*p. Strictly positive bounded payoffs only;
    no clipping or silent replacement of nonfinite values is permitted.
    """
    if (not math.isfinite(defensive_mass) or not 0 < defensive_mass <= 1
            or log_g.ndim != 1 or log_q_over_p.shape != log_g.shape
            or log_g.dtype != torch.float64 or log_q_over_p.dtype != torch.float64
            or log_g.device.type != "cpu" or log_q_over_p.device.type != "cpu"
            or not torch.isfinite(log_g).all() or not torch.isfinite(log_q_over_p).all()
            or bool((log_g > 0).any())):
        raise ValueError("risk requires finite CPU float64 log(g), 0 < g <= 1")
    if bool((log_q_over_p < math.log(defensive_mass) - 1e-12).any()):
        raise ValueError("density violates declared defensive lower bound")
    result = math.log(defensive_mass) + 2 * log_g - log_q_over_p
    if not torch.isfinite(result).all() or bool((result > 1e-12).any()):
        raise FloatingPointError("invalid bounded risk potential")
    return result


@dataclass(frozen=True)
class PrecisionAllocation:
    pilot_count: int
    target_relative_se: float
    safety_factor: float
    planned_count: int | None
    maximum_count: int
    status: str


def allocate_precision(
    pilot_log_values: torch.Tensor, *, target_relative_se: float,
    safety_factor: float, batch_size: int, maximum_count: int,
) -> PrecisionAllocation:
    """Freeze n from a disjoint pilot. Safety factor is not a confidence bound."""
    if (not math.isfinite(target_relative_se) or not 0 < target_relative_se < 1
            or not math.isfinite(safety_factor) or safety_factor < 1
            or batch_size < 2 or maximum_count < batch_size):
        raise ValueError("invalid allocation contract")
    summary = summarize_log_contributions(pilot_log_values)
    status = "allocated"
    planned: int | None = None
    if summary.relative_se is None or not math.isfinite(summary.relative_se):
        status = "unresolved_pilot"
    else:
        # RSE^2 = empirical CV^2/(n-1); convert to unbiased sample CV^2.
        cv2 = summary.relative_se**2 * summary.count
        raw = safety_factor * cv2 / target_relative_se**2
        if not math.isfinite(raw):
            status = "unresolved_pilot"
        else:
            planned = max(batch_size, math.ceil(raw / batch_size) * batch_size)
            if planned > maximum_count:
                status = "unresolved_sample_budget"
    return PrecisionAllocation(summary.count, target_relative_se, safety_factor,
                               planned, maximum_count, status)


def summarize_risk_replicates(
    log_normalizers: torch.Tensor, *, defensive_mass: float,
    expected_replicates: int, maximum_relative_se: float,
) -> dict[str, Any]:
    """SE unit is a WHOLE SMC run, never an individual terminal particle.

    Z_hat/delta estimates M2 for a frozen q. Empirical precision is a development
    gate only: a small RSE cannot certify absence of unobserved high-risk modes.
    """
    if (not 0 < defensive_mass <= 1 or not math.isfinite(defensive_mass)
            or expected_replicates < 2 or not 0 < maximum_relative_se < 1
            or log_normalizers.ndim != 1 or log_normalizers.dtype != torch.float64
            or log_normalizers.device.type != "cpu"
            or not torch.isfinite(log_normalizers).all()
            or bool((log_normalizers > 1e-12).any())):
        raise ValueError("invalid independent risk replication contract")
    if log_normalizers.numel() < 2:
        return {"completed_replicates": log_normalizers.numel(), "status": "unresolved_incomplete_risk",
                "mean": None, "standard_error": None, "relative_se": None}
    summary = summarize_log_contributions(log_normalizers - math.log(defensive_mass))
    if summary.log_mean is None or summary.relative_se is None:
        raise FloatingPointError("risk replicate summary has zero mass")
    try:
        mean = math.exp(summary.log_mean)
    except OverflowError:
        mean = math.inf
    if mean == 0 or not math.isfinite(mean):
        return {**asdict(summary), "completed_replicates": summary.count, "mean": None,
                "standard_error": None, "status": "unresolved_risk_numerical_range",
                "se_unit": "independent_whole_smc_normalizer"}
    status = "development_precision_pass_not_oracle"
    if summary.count != expected_replicates:
        status = "unresolved_incomplete_risk"
    elif summary.relative_se > maximum_relative_se:
        status = "unresolved_risk_precision"
    return {**asdict(summary), "completed_replicates": summary.count, "mean": mean,
            "standard_error": mean * summary.relative_se, "status": status,
            "se_unit": "independent_whole_smc_normalizer"}


def log_auxiliary_second_moment(
    log_g: torch.Tensor, log_q_over_p: torch.Tensor, log_auxiliary_over_p: torch.Tensor,
) -> torch.Tensor:
    """X~r: Y=g^2 p^2/(q r), E_r[Y]=M2(q); all proposals frozen/normalized.

    This is ordinary IS for risk, not self-normalized fitting and not a change
    to the model q. r=q gives the usual squared ordinary IS contribution.
    """
    values = (log_g, log_q_over_p, log_auxiliary_over_p)
    if (log_g.ndim != 1 or any(x.shape != log_g.shape or x.dtype != torch.float64
            or x.device.type != "cpu" or not torch.isfinite(x).all() for x in values)
            or bool((log_g > 0).any())):
        raise ValueError("auxiliary risk requires finite CPU float64 log densities and payoff")
    result = 2*log_g-log_q_over_p-log_auxiliary_over_p
    if not torch.isfinite(result).all():
        raise FloatingPointError("nonfinite auxiliary risk contribution")
    return result
