"""Analytic critical-point splitting and vectorized conditional moment checks."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.optimize import brentq
from scipy.special import log_ndtr, logsumexp, roots_legendre

from .residual_structural_oracle import LastVarianceResidualSlice


def validate_profile(p: LastVarianceResidualSlice) -> None:
    if (not all(math.isfinite(x) for x in (p.log_initial_last_variance, p.beta,
            p.preceding_terminal, p.preceding_integrated_variance, p.last_dw, p.dt, p.rho, p.log_level))
            or p.beta < 0 or p.dt <= 0 or p.preceding_integrated_variance <= 0 or abs(p.rho) >= 1):
        raise ValueError("finite beta>=0, nondegenerate scalar profile required")


def critical_points(p: LastVarianceResidualSlice, cutoff: float = 16.) -> dict[str, Any]:
    validate_profile(p)
    if not math.isfinite(cutoff) or not 2 <= cutoff <= 32:
        raise ValueError("finite cutoff 2..32 required")
    if p.beta == 0:
        return {"points": [], "constant": True, "near_multiple_root": False, "positive_root_bound": 2}
    a, b, c, d, e = (p.log_level-p.preceding_terminal, .5*p.dt,
        -p.rho*p.last_dw, p.preceding_integrated_variance, p.dt)
    coefficients = np.array([b*e, 2*b*d-e*a, c*d], dtype=np.float64)
    if not np.isfinite(coefficients).all() or coefficients[0] <= 0:
        raise FloatingPointError("critical polynomial coefficients unresolved")
    coefficients /= np.max(np.abs(coefficients))
    A, B, C = coefficients.tolist()
    if A <= 0:
        raise FloatingPointError("scaled positive cubic coefficient underflow")
    lower, upper = [math.exp(.5*(p.log_initial_last_variance+p.beta*u)) for u in (-cutoff, cutoff)]
    if not 0 < lower < upper or not math.isfinite(upper):
        raise FloatingPointError("critical t-domain unresolved")

    def polynomial(t: float) -> float:
        value = (A*t*t+B)*t+C
        if not math.isfinite(value):
            raise FloatingPointError("critical polynomial overflow")
        return value
    brackets = [lower, upper]
    ambiguous = False
    if B < 0:
        turning = math.sqrt(-B/(3*A))
        if lower < turning < upper:
            brackets.insert(1, turning)
            tolerance = 256*np.finfo(np.float64).eps*(abs(A*turning**3)+abs(B*turning)+abs(C))
            ambiguous = abs(polynomial(turning)) <= tolerance
    roots = []
    for left, right in zip(brackets[:-1], brackets[1:], strict=True):
        fl, fr = polynomial(left), polynomial(right)
        if fl == 0:
            roots.append(left)
        if fr == 0:
            roots.append(right)
        if (fl < 0 < fr) or (fr < 0 < fl):
            roots.append(brentq(polynomial, left, right, xtol=5e-324, rtol=1e-13, maxiter=200))
    points = sorted(set((2*math.log(t)-p.log_initial_last_variance)/p.beta for t in roots))
    points = [u for u in points if -cutoff < u < cutoff]
    if len(points) > 2:
        raise FloatingPointError("positive cubic root count exceeds analytic bound")
    return {"points": points, "constant": False, "near_multiple_root": ambiguous,
        "positive_root_bound": 2, "scaled_cubic_coefficients": coefficients.tolist()}


def _batch_logs(profiles: list[LastVarianceResidualSlice], u: np.ndarray) -> np.ndarray:
    shape = (len(profiles),)+(1,)*(u.ndim-1)

    def field(name: str) -> np.ndarray:
        return np.array([getattr(p, name) for p in profiles], dtype=np.float64).reshape(shape)
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        variance = np.exp(field("log_initial_last_variance")+field("beta")*u)
        x = ((field("log_level")-field("preceding_terminal")+.5*field("dt")*variance
             -field("rho")*np.sqrt(variance)*field("last_dw"))
            /np.sqrt((1-field("rho")**2)*(field("preceding_integrated_variance")+field("dt")*variance)))
        logs = np.asarray(log_ndtr(x))
    if not np.isfinite(logs).all() or (variance <= 0).any():
        raise FloatingPointError("batched conditional slice numerical range unresolved")
    return logs


def _log_mass(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    # Positive-tail panels use survival probabilities, avoiding 1-Phi cancellation.
    positive = left >= 0
    hi = np.where(positive, log_ndtr(-left), log_ndtr(right))
    lo = np.where(positive, log_ndtr(-right), log_ndtr(left))
    with np.errstate(divide="ignore", invalid="raise"):
        return hi+np.log(-np.expm1(lo-hi))


def _log_abs_difference(a: float, b: float) -> float:
    return -math.inf if a == b else max(a, b)+math.log(-math.expm1(min(a, b)-max(a, b)))


def batched_moments(profiles: list[LastVarianceResidualSlice], *, cutoff: int = 16,
                    charge: Callable[[int], None] | None = None) -> dict[str, Any]:
    if not profiles or type(cutoff) is not int or not 2 <= cutoff <= 32:
        raise ValueError("nonempty profile list and integer cutoff 2..32 required")
    critical = [critical_points(p, float(cutoff)) for p in profiles]
    edges = [sorted(set(np.arange(-cutoff, cutoff+1, dtype=np.float64).tolist()+r["points"])) for r in critical]
    panels = max(len(e)-1 for e in edges)
    left = np.full((len(profiles), panels), float(cutoff), dtype=np.float64)
    right = left.copy()
    for j, e in enumerate(edges):
        left[j, :len(e)-1], right[j, :len(e)-1] = e[:-1], e[1:]
    half, middle = .5*(right-left), .5*(right+left)
    count = 0

    def evaluate(u: np.ndarray) -> np.ndarray:
        nonlocal count
        if charge:
            charge(u.size)
        count += u.size
        return _batch_logs(profiles, u)
    rules = []
    for order in (16, 32, 64):
        nodes, weights = roots_legendre(order)
        u = middle[:, :, None]+half[:, :, None]*nodes[None, None, :]
        logs = evaluate(u)
        with np.errstate(divide="ignore"):
            base = -.5*u*u-.5*math.log(2*math.pi)+np.log(half)[:, :, None]+np.log(weights)[None, None, :]
        rules.append(np.stack([logsumexp(base+power*logs, axis=(1, 2)) for power in (1, 2)], axis=1))
    endpoints = np.stack([left, right], axis=2)
    endlogs = evaluate(endpoints)
    mass = _log_mass(left, right)
    lowg, highg = endlogs.min(2), endlogs.max(2)
    logtail = math.log(2)+float(log_ndtr(-cutoff))
    rows = []
    for j in range(len(profiles)):
        moments: list[dict[str, Any]] = []
        for k, power in enumerate((1, 2)):
            value = float(rules[-1][j, k])
            lower = float(logsumexp(mass[j]+power*lowg[j]))
            upper = float(np.logaddexp(logsumexp(mass[j]+power*highg[j]), logtail))
            log_error = float(logsumexp([value+math.log(1e-12), logtail,
                _log_abs_difference(float(rules[-1][j, k]), float(rules[-2][j, k])),
                _log_abs_difference(float(rules[-2][j, k]), float(rules[-3][j, k]))]))
            relative = math.exp(log_error-value) if log_error-value < 700 else None
            enclosed = lower-1e-10 <= value <= upper+1e-10
            moments.append({"power": power, "log_estimate": value,
                "order_log_estimates": [float(r[j, k]) for r in rules],
                "log_monotone_envelope_lower": lower, "log_monotone_envelope_upper": upper,
                "inside_numerical_envelope": enclosed, "relative_error_proxy_plus_tail": relative,
                "log_tail_upper": logtail,
                "relative_numerically_resolved": relative is not None and relative <= 1e-6 and enclosed})
        valid = (all(m["relative_numerically_resolved"] for m in moments)
            and not critical[j]["near_multiple_root"] and moments[1]["log_estimate"] >= 2*moments[0]["log_estimate"]-1e-8
            and moments[1]["log_estimate"] <= moments[0]["log_estimate"]+1e-8)
        rows.append({"critical": critical[j], "panel_count": len(edges[j])-1, "moments": moments,
            "relative_numerically_resolved": valid, "not_outward_rounded_or_rigorous_interval": True})
    return {"rows": rows, "scalar_evaluations_including_padding_and_endpoints": count,
        "all_relative_numerically_resolved": all(r["relative_numerically_resolved"] for r in rows),
        "not_model_speedup_or_population_certificate": True}


@dataclass(frozen=True)
class JointBlindResidualSlice:
    coordinates: tuple[int, int]
    variance_indices: tuple[int, int]
    log_variances: tuple[float, float]
    beta: float
    base_terminal: float
    base_integrated: float
    dws: tuple[float, float]
    dt: float
    rho: float
    log_level: float

    def __post_init__(self) -> None:
        if (self.coordinates != (57, 61) or self.variance_indices != (29, 31)
                or len(self.log_variances) != 2 or len(self.dws) != 2
                or not all(math.isfinite(x) for x in (*self.log_variances, *self.dws, self.beta,
                    self.base_terminal, self.base_integrated, self.dt, self.rho, self.log_level))
                or self.beta < 0 or self.base_integrated <= 0 or self.dt <= 0 or abs(self.rho) >= 1):
            raise ValueError("finite nondegenerate fixed N32 joint residual profile required")

    @classmethod
    def create(cls, cache: Any, z: torch.Tensor) -> JointBlindResidualSlice:
        p = cache.problem
        if (p.steps != 32 or z.shape != (1, 64) or z.dtype != torch.float64 or z.device.type != "cpu"
                or z.requires_grad or not torch.isfinite(z).all() or cache.kernel.local_cholesky[0, 1] != 0):
            raise ValueError("one detached finite CPU float64 N32 context required")
        base = z.clone()
        base[:, [57, 61, 63]] = 0
        _, _, variance = cache.quantities(base)
        v = variance[0, :-1]
        dw = (base.reshape(1, 32, 2) @ cache.kernel.local_cholesky.T)[0, :, 0]
        keep = torch.ones(32, dtype=torch.bool)
        keep[[29, 31]] = False
        terminal = math.log(p.spot)+float((-.5*p.step_dt*v[keep]+p.rho*v[keep].sqrt()*dw[keep]).sum())
        return cls((57, 61), (29, 31), (float(v[29].log()), float(v[31].log())),
            p.eta*math.sqrt(2*p.hurst)*float(cache.kernel.local_cholesky[1, 1]),
            terminal, float(p.step_dt*v[keep].sum()), (float(dw[29]), float(dw[31])),
            p.step_dt, p.rho, math.log(p.task.level))

    def log_probability(self, coordinates: np.ndarray) -> np.ndarray:
        if coordinates.ndim != 2 or coordinates.shape[1] != 2 or not np.isfinite(coordinates).all():
            raise ValueError("finite n-by-two residual coordinates required")
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            v = np.exp(np.array(self.log_variances)[None, :]+self.beta*coordinates)
            x = ((self.log_level-self.base_terminal+.5*self.dt*v.sum(1)
                  -self.rho*(np.sqrt(v)*np.array(self.dws)[None, :]).sum(1))
                 /np.sqrt((1-self.rho**2)*(self.base_integrated+self.dt*v.sum(1))))
            logs = np.asarray(log_ndtr(x))
        if not np.isfinite(logs).all() or (v <= 0).any():
            raise FloatingPointError("joint residual range unresolved")
        return logs
