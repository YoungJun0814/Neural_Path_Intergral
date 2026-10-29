"""Finite-dimensional Gaussian-split HMC; no continuum or mixing guarantee."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from .gradient_umbrella_smc import _coordinates, evaluate_family_gradient

Evaluator = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


@dataclass(frozen=True)
class SplitTrajectory:
    position: torch.Tensor
    flipped_momentum: torch.Tensor
    components: torch.Tensor
    potential: torch.Tensor
    drift: torch.Tensor
    log_acceptance: torch.Tensor
    callback_rows: int


def gaussian_split_trajectory(z: torch.Tensor, momentum: torch.Tensor,
                              components: torch.Tensor, potential: torch.Tensor,
                              drift: torch.Tensor, *, evaluate: Evaluator,
                              beta: float, step_size: float, steps: int) -> SplitTrajectory:
    """Kick/rotation/kick, momentum flip, and stable full Hamiltonian correction.

    A deterministic capped force remains a reversible volume-preserving shear.
    It is not asserted to be the gradient of a modified potential. Target values
    are never capped; acceptance uses the original target Hamiltonian.
    """
    for x in (z, momentum, drift):
        _coordinates(x)
        if x.shape != z.shape:
            raise ValueError("phase-space shape mismatch")
    if (potential.shape != (len(z),) or potential.dtype != torch.float64
            or potential.device.type != "cpu" or not torch.isfinite(potential).all()
            or components.ndim != 2 or len(components) != len(z)
            or components.dtype != torch.float64 or components.device.type != "cpu"
            or not torch.isfinite(components).all() or type(steps) is not int or steps < 1
            or isinstance(beta, bool) or not math.isfinite(beta) or not 0 <= beta <= 1
            or isinstance(step_size, bool) or not math.isfinite(step_size) or not 0 < step_size < math.pi):
        raise ValueError("finite fixed Hamiltonian trajectory required")
    x, p, h, force, comp = z.clone(), momentum.clone(), potential, drift, components
    gaussian_energy_change = torch.zeros(len(z), dtype=torch.float64)
    cosine, sine = math.cos(step_size), math.sin(step_size)
    for _ in range(steps):
        kick = .5*step_size*beta*force
        gaussian_energy_change += (p*kick).sum(1)+.5*kick.square().sum(1)
        p = p+kick
        x, p = cosine*x+sine*p, cosine*p-sine*x
        comp, h, force = evaluate(x)
        kick = .5*step_size*beta*force
        gaussian_energy_change += (p*kick).sum(1)+.5*kick.square().sum(1)
        p = p+kick
    ratio = beta*(h-potential)-gaussian_energy_change
    if not torch.isfinite(ratio).all():
        raise FloatingPointError("nonfinite full Hamiltonian correction")
    return SplitTrajectory(x, -p, comp, h, force, ratio, len(z)*steps)


def probability_evaluator(callback: Callable[[torch.Tensor], torch.Tensor], *,
                          maximum_drift_norm: float = 20.) -> Evaluator:
    def evaluate(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return evaluate_family_gradient(callback, z, torch.zeros(1, dtype=torch.float64),
                                        maximum_drift_norm=maximum_drift_norm)
    return evaluate
