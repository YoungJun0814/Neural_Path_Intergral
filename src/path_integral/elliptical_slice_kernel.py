"""Batched Gaussian-prior elliptical slice transition with explicit cap failure."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EllipticalSliceResult:
    particles: torch.Tensor
    log_potential: torch.Tensor
    potential_evaluations: int
    maximum_attempts: int


def elliptical_slice_transition(particles: torch.Tensor, log_values: torch.Tensor, *,
                                beta: float, log_potential: Callable[[torch.Tensor], torch.Tensor],
                                generator: torch.Generator,
                                maximum_attempts: int = 1024) -> EllipticalSliceResult:
    if (particles.ndim != 2 or particles.shape[1] < 1 or len(particles) < 1
            or log_values.shape != (len(particles),)
            or any(v.dtype != torch.float64 or v.device.type != "cpu" or not torch.isfinite(v).all()
                   for v in (particles, log_values))
            or not math.isfinite(beta) or not 0 <= beta <= 1
            or isinstance(maximum_attempts, bool) or not isinstance(maximum_attempts, int)
            or maximum_attempts < 1):
        raise ValueError("invalid Gaussian ellipse transition input")
    count = len(particles)
    noise = torch.randn(particles.shape, dtype=torch.float64, generator=generator)
    threshold = beta * log_values + torch.rand(count, dtype=torch.float64, generator=generator).log()
    theta = 2 * math.pi * torch.rand(count, dtype=torch.float64, generator=generator)
    lower, upper = theta - 2 * math.pi, theta.clone()
    result, values = particles.clone(), log_values.clone()
    active = torch.arange(count)
    calls = 0
    for attempt in range(1, maximum_attempts + 1):
        angles = theta[active]
        candidate = particles[active] * angles.cos()[:, None] + noise[active] * angles.sin()[:, None]
        candidate_values = log_potential(candidate)
        calls += len(active)
        if (candidate_values.shape != (len(active),) or candidate_values.dtype != torch.float64
                or candidate_values.device.type != "cpu" or not torch.isfinite(candidate_values).all()):
            raise FloatingPointError("ellipse potential must remain finite CPU float64")
        accept = beta * candidate_values > threshold[active]
        accepted = active[accept]
        result[accepted], values[accepted] = candidate[accept], candidate_values[accept]
        active = active[~accept]
        if len(active) == 0:
            return EllipticalSliceResult(result, values, calls, attempt)
        negative = theta[active] < 0
        lower[active[negative]] = theta[active[negative]]
        upper[active[~negative]] = theta[active[~negative]]
        theta[active] = lower[active] + (upper[active] - lower[active]) * torch.rand(
            len(active), dtype=torch.float64, generator=generator)
    raise TimeoutError(f"ellipse_cap_failure: calls={calls}, unfinished={len(active)}; no truncated kernel returned")
