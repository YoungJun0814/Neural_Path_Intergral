"""Exact Gaussian checks and the last-used Volterra variance residual slice.

Diagnostics only; no production proposal or estimator is replaced here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


def bump_shift_moments(center: tuple[float, ...], width: float, mean: tuple[float, ...]) -> dict[str, Any]:
    if (not center or len(center) != len(mean) or not math.isfinite(width) or width <= 0
            or any(not math.isfinite(v) for v in (*center, *mean))):
        raise ValueError("finite equal-dimensional Gaussian bump/shift required")
    d, s2 = len(center), width**2
    optimal = tuple(c/(1+s2) for c in center)
    log_mu = .5*d*math.log(s2/(1+s2))-sum(c*c for c in center)/(2*(1+s2))
    floor = d*math.log((s2+1)/math.sqrt(s2*(s2+2)))
    mean_gap = (s2+1)/(s2+2)*sum((m-o)**2 for m, o in zip(mean, optimal, strict=True))
    return {"log_mu": log_mu, "log_m2": 2*log_mu+floor+mean_gap,
        "log_relative_second_moment": floor+mean_gap, "log_covariance_family_floor": floor,
        "log_mean_mismatch_penalty": mean_gap, "optimal_identity_shift": list(optimal),
        "risk_target_mean": [(2*c-s2*m)/(s2+2) for c, m in zip(center, mean, strict=True)],
        "risk_target_variance": s2/(s2+2)}


def gaussian_log_rho(target: tuple[float, ...], target_variance: float,
                     proposal: tuple[float, ...], proposal_variance: float) -> float:
    """log integral r²/G, including the exact finite-moment condition."""
    if (not target or len(target) != len(proposal)
            or any(not math.isfinite(v) for v in (*target, *proposal, target_variance, proposal_variance))
            or min(target_variance, proposal_variance) <= 0):
        raise ValueError("nondegenerate finite isotropic normals required")
    denominator = 2*proposal_variance-target_variance
    if denominator <= 0:
        return math.inf
    return (len(target)*math.log(proposal_variance/math.sqrt(target_variance*denominator))
        +sum((a-b)**2 for a, b in zip(target, proposal, strict=True))/denominator)


def untouched_bump_floor(center: tuple[float, ...], width: float) -> float:
    """log E[g²]/E[g]² when no proposal change acts on these coordinates."""
    result = bump_shift_moments(center, width, tuple(0. for _ in center))
    return float(result["log_relative_second_moment"])


@dataclass(frozen=True)
class LastVarianceResidualSlice:
    steps: int
    log_initial_last_variance: float
    beta: float
    preceding_terminal: float
    preceding_integrated_variance: float
    last_dw: float
    dt: float
    rho: float
    log_level: float

    @classmethod
    def create(cls, cache: Any, coordinates: torch.Tensor) -> LastVarianceResidualSlice:
        p = cache.problem
        if (coordinates.shape != (1, 2*p.steps) or coordinates.dtype != torch.float64
                or coordinates.device.type != "cpu" or coordinates.requires_grad
                or not torch.isfinite(coordinates).all() or cache.kernel.local_cholesky[0, 1] != 0):
            raise ValueError("one detached CPU float64 context and triangular local law required")
        z = coordinates.clone()
        z[0, 2*p.steps-3] = 0
        z[0, 2*p.steps-1] = 0
        _, _, variance = cache.quantities(z)
        dw = (z.reshape(1, p.steps, 2) @ cache.kernel.local_cholesky.T)[0, :, 0]
        preceding = variance[0, :p.steps-1]
        terminal = math.log(p.spot)+float((-.5*preceding*p.step_dt+p.rho*preceding.sqrt()*dw[:-1]).sum())
        integrated = float(p.step_dt*preceding.sum())
        return cls(p.steps, float(variance[0, p.steps-1].log()),
            p.eta*math.sqrt(2*p.hurst)*float(cache.kernel.local_cholesky[1, 1]),
            terminal, integrated, float(dw[-1]), p.step_dt, p.rho, math.log(p.task.level))

    def log_probability(self, residual: torch.Tensor) -> torch.Tensor:
        if (residual.ndim != 1 or residual.dtype != torch.float64 or residual.device.type != "cpu"
                or residual.requires_grad or not torch.isfinite(residual).all()):
            raise ValueError("finite detached CPU float64 residual vector required")
        variance = (self.log_initial_last_variance+self.beta*residual).exp()
        value = ((self.log_level-self.preceding_terminal+.5*self.dt*variance
            -self.rho*variance.sqrt()*self.last_dw)
            /math.sqrt(1-self.rho**2)/(self.preceding_integrated_variance+self.dt*variance).sqrt())
        if not torch.isfinite(variance).all() or (variance <= 0).any() or not torch.isfinite(value).all():
            raise FloatingPointError("residual slice numerical range unresolved")
        return torch.special.log_ndtr(value)
