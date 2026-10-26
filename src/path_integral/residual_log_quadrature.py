"""Log-scaled conditional Gaussian moments; numerical checks, not interval proofs."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.integrate import quad
from scipy.special import log_ndtr, logsumexp, roots_legendre

from .residual_structural_oracle import LastVarianceResidualSlice


def slice_log_probability(profile: LastVarianceResidualSlice, u: np.ndarray) -> np.ndarray:
    """Independent scalar/vector expression, without constructing full paths."""
    if (u.ndim != 1 or not np.isfinite(u).all()
            or not all(math.isfinite(x) for x in (profile.log_initial_last_variance, profile.beta,
                profile.preceding_terminal, profile.preceding_integrated_variance,
                profile.last_dw, profile.dt, profile.rho, profile.log_level))
            or profile.dt <= 0 or profile.preceding_integrated_variance <= 0 or abs(profile.rho) >= 1):
        raise ValueError("finite nondegenerate residual profile and vector required")
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        v = np.exp(profile.log_initial_last_variance+profile.beta*u)
        x = ((profile.log_level-profile.preceding_terminal+.5*profile.dt*v
              -profile.rho*np.sqrt(v)*profile.last_dw)
             /np.sqrt((1-profile.rho**2)*(profile.preceding_integrated_variance+profile.dt*v)))
        result = np.asarray(log_ndtr(x), dtype=np.float64)
    if not np.isfinite(result).all() or (v <= 0).any():
        raise FloatingPointError("scalar slice numerical range unresolved")
    return result


def _log_difference(a: float, b: float) -> float:
    if a == b:
        return -math.inf
    hi, lo = max(a, b), min(a, b)
    return hi+math.log(-math.expm1(lo-hi))


def log_gaussian_moments(log_g: Callable[[np.ndarray], np.ndarray], *, cutoff: float = 16.,
                         relative_tolerance: float = 1e-6,
                         charge: Callable[[int], None] | None = None) -> dict[str, Any]:
    """E[g], E[g²] for a declared 0<=g<=1, with explicit Gaussian tail accounting.

    The callback bound is an API assumption, not certified by testing some nodes.
    QUADPACK error estimates and cross-rule differences are not rigorous bounds.
    """
    if (not math.isfinite(cutoff) or cutoff < 2 or cutoff > 32 or cutoff != int(cutoff)
            or not math.isfinite(relative_tolerance) or not 1e-12 <= relative_tolerance <= .01):
        raise ValueError("integer cutoff 2..32 and supported positive relative tolerance required")
    evaluations = 0

    def evaluate(u: np.ndarray) -> np.ndarray:
        nonlocal evaluations
        if charge:
            charge(len(u))
        evaluations += len(u)
        value = np.asarray(log_g(u), dtype=np.float64)
        if value.shape != u.shape or not np.isfinite(value).all() or (value > 1e-12).any():
            raise ValueError("finite log of a declared bounded probability required")
        return value

    edges = np.arange(-cutoff, cutoff+1, dtype=np.float64)
    rule_logs, dense_nodes, dense_logs = [], None, None
    for order in (16, 32, 64):
        nodes, weights = roots_legendre(order)
        u = .5*(edges[:-1, None]+edges[1:, None])+.5*nodes[None, :]
        logs = evaluate(u.ravel()).reshape(u.shape)
        base = -.5*u*u-.5*math.log(2*math.pi)+np.log(.5*weights)[None, :]
        rule_logs.append([float(logsumexp(base+power*logs)) for power in (1, 2)])
        dense_nodes, dense_logs = u, logs
    assert dense_nodes is not None and dense_logs is not None
    moments: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    log_tail = math.log(2)+float(log_ndtr(-cutoff))
    for k, power in enumerate((1, 2)):
        panel_logs, error_logs = [], []
        for j, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            shift = float(np.max(power*dense_logs[j]-.5*dense_nodes[j]**2-.5*math.log(2*math.pi)))

            def integrand(u: float, power: int = power, shift: float = shift) -> float:
                value = power*float(evaluate(np.array([u]))[0])-.5*u*u-.5*math.log(2*math.pi)-shift
                if value > 600:
                    raise FloatingPointError("local quadrature scale unresolved")
                return math.exp(value)
            result = quad(integrand, float(left), float(right), epsabs=0., epsrel=1e-10,
                          limit=100, full_output=1)
            value, error = float(result[0]), float(result[1])
            if not math.isfinite(value) or not math.isfinite(error) or min(value, error) < 0:
                raise FloatingPointError("adaptive quadrature nonfinite")
            if len(result) > 3:
                messages.append({"power": power, "panel": j, "message": str(result[3])})
            panel_logs.append(shift+math.log(value) if value > 0 else -math.inf)
            error_logs.append(shift+math.log(error) if error > 0 else -math.inf)
        estimate = float(logsumexp(panel_logs))
        if not math.isfinite(estimate):
            raise FloatingPointError("positive log moment unresolved")
        log_internal = float(logsumexp([float(logsumexp(error_logs)),
            _log_difference(estimate, rule_logs[-1][k]), _log_difference(rule_logs[-1][k], rule_logs[-2][k])]))
        log_error = float(logsumexp([log_internal, log_tail]))
        relative = math.exp(log_error-estimate) if log_error-estimate < 700 else None
        lower = estimate+math.log1p(-relative) if relative is not None and relative < 1 else None
        upper = float(np.logaddexp(estimate, log_error))
        resolved = relative is not None and relative <= relative_tolerance
        moments.append({"power": power, "log_estimate": estimate,
            "legendre_log_estimates": [r[k] for r in rule_logs],
            "log_internal_error_proxy": log_internal if math.isfinite(log_internal) else None,
            "log_tail_upper": log_tail, "log_total_error_proxy_plus_tail": log_error,
            "relative_error_proxy_plus_tail": relative, "log_lower_proxy": lower,
            "log_upper_proxy": upper, "relative_numerically_resolved": resolved,
            "rigorous_internal_error_bound": False})
    first, second = moments
    compatible = (first["log_lower_proxy"] is not None and second["log_lower_proxy"] is not None
        and second["log_lower_proxy"] <= first["log_upper_proxy"]+1e-12
        and second["log_upper_proxy"] >= 2*first["log_lower_proxy"]-1e-12)
    resolved = all(m["relative_numerically_resolved"] for m in moments) and not messages and compatible
    return {"status": "RELATIVE_NUMERICAL_RESOLUTION" if resolved else "UNRESOLVED",
        "relative_numerically_resolved": resolved, "cutoff": cutoff, "panel_width": 1.,
        "legendre_orders": [16, 32, 64], "relative_tolerance": relative_tolerance,
        "scalar_evaluations": evaluations, "moments": moments, "adaptive_messages": messages,
        "moment_inequalities_compatible": compatible,
        "log_conditional_relative_second_moment": second["log_estimate"]-2*first["log_estimate"],
        "log_conditional_ratio_lower_proxy": (second["log_lower_proxy"]-2*first["log_upper_proxy"] if resolved else None),
        "log_conditional_ratio_upper_proxy": (second["log_upper_proxy"]-2*first["log_lower_proxy"] if resolved else None),
        "not_rigorous_certificate_or_unbiased_production_estimator": True}
