"""Causal finite-grid path functionals for downside-excursion rare events."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def _validate_finite_spot(spot: torch.Tensor, step_dt: float) -> None:
    if spot.ndim != 2 or spot.shape[1] < 2:
        raise ValueError("spot must have shape (paths, steps + 1)")
    if not spot.is_floating_point() or not torch.isfinite(spot).all():
        raise ValueError("spot paths must be finite floating-point tensors")
    if bool((spot <= 0.0).any()):
        raise ValueError("spot paths must be strictly positive")
    if not math.isfinite(step_dt) or step_dt <= 0.0:
        raise ValueError("step_dt must be finite and positive")


@dataclass(frozen=True)
class TerminalThresholdTask:
    """A finite-grid terminal downside event ``S_T <= level``."""

    level: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.level) or self.level <= 0.0:
            raise ValueError("terminal level must be finite and positive")

    def hard_event(self, spot: torch.Tensor, step_dt: float) -> torch.Tensor:
        _validate_finite_spot(spot, step_dt)
        return spot[:, -1] <= self.level


@dataclass(frozen=True)
class DiscreteBarrierHitTask:
    """A right-endpoint finite-grid downside barrier event."""

    barrier: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.barrier) or self.barrier <= 0.0:
            raise ValueError("barrier must be finite and positive")

    def hard_event(self, spot: torch.Tensor, step_dt: float) -> torch.Tensor:
        _validate_finite_spot(spot, step_dt)
        return torch.amin(spot, dim=1) <= self.barrier


@dataclass(frozen=True)
class DownsideExcursionTask:
    """Barrier-hit plus stress-occupation event on a right-endpoint grid."""

    hit_barrier: float
    stress_level: float
    minimum_occupation: float
    hit_scale: float
    occupation_scale: float

    def __post_init__(self) -> None:
        values = (
            self.hit_barrier,
            self.stress_level,
            self.minimum_occupation,
            self.hit_scale,
            self.occupation_scale,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("task thresholds and scales must be finite and positive")
        if self.hit_barrier >= self.stress_level:
            raise ValueError("hit_barrier must be strictly below stress_level")

    @staticmethod
    def _validate_spot(spot: torch.Tensor, step_dt: float) -> None:
        _validate_finite_spot(spot, step_dt)

    def prefix_state(
        self, spot: torch.Tensor, step_dt: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return running minimum, right-endpoint occupation, and hit state."""
        self._validate_spot(spot, step_dt)
        running_minimum = torch.cummin(spot, dim=1).values
        increments = (spot[:, 1:] <= self.stress_level).to(spot.dtype) * step_dt
        occupation = torch.cat(
            (
                torch.zeros(spot.shape[0], 1, device=spot.device, dtype=spot.dtype),
                torch.cumsum(increments, dim=1),
            ),
            dim=1,
        )
        hit = running_minimum <= self.hit_barrier
        return running_minimum, occupation, hit

    def hard_event(self, spot: torch.Tensor, step_dt: float) -> torch.Tensor:
        running_minimum, occupation, _hit = self.prefix_state(spot, step_dt)
        return (running_minimum[:, -1] <= self.hit_barrier) & (
            occupation[:, -1] + 1e-15 >= self.minimum_occupation
        )

    def soft_payoff(self, spot: torch.Tensor, step_dt: float) -> torch.Tensor:
        """Return a bounded training payoff; final estimators use ``hard_event``."""
        running_minimum, occupation, _hit = self.prefix_state(spot, step_dt)
        hit_margin = (self.hit_barrier - running_minimum[:, -1]) / self.hit_scale
        occupation_margin = (
            occupation[:, -1] - self.minimum_occupation
        ) / self.occupation_scale
        return torch.sigmoid(hit_margin) * torch.sigmoid(occupation_margin)

    def score(self, spot: torch.Tensor, step_dt: float) -> torch.Tensor:
        """A monotone CEM level score whose nonnegative set is the hard event."""
        running_minimum, occupation, _hit = self.prefix_state(spot, step_dt)
        hit_margin = (self.hit_barrier - running_minimum[:, -1]) / self.hit_scale
        occupation_margin = (
            occupation[:, -1] - self.minimum_occupation
        ) / self.occupation_scale
        return torch.minimum(hit_margin, occupation_margin)
