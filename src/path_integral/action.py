"""Discrete Brownian likelihood and path-action conventions.

The target Brownian motion and proposal Brownian motion are related by

``dB^M = dB^Q + u dt``.

Consequently, this module always returns ``log(dM/dQ)``.  Keeping that
orientation explicit prevents the most common sign error in controlled
importance-sampling implementations.
"""

from __future__ import annotations

import math

import torch


def _validate_brownian_arrays(
    controls: torch.Tensor,
    proposal_increments: torch.Tensor,
    dt: float,
) -> None:
    if controls.shape != proposal_increments.shape:
        raise ValueError("controls and proposal_increments must have identical shapes")
    if controls.ndim < 2:
        raise ValueError("expected shape (..., time_steps, brownian_drivers)")
    if controls.shape[-2] < 1 or controls.shape[-1] < 1:
        raise ValueError("time_steps and brownian_drivers must both be nonempty")
    if not controls.is_floating_point() or not proposal_increments.is_floating_point():
        raise TypeError("controls and proposal_increments must be floating-point tensors")
    if controls.device != proposal_increments.device:
        raise ValueError("controls and proposal_increments must be on the same device")
    if controls.dtype != proposal_increments.dtype:
        raise ValueError("controls and proposal_increments must have the same dtype")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if not torch.isfinite(controls).all() or not torch.isfinite(proposal_increments).all():
        raise ValueError("controls and proposal_increments must be finite")


def brownian_log_likelihood(
    controls: torch.Tensor,
    proposal_increments: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Return the discrete ``log(dM/dQ)`` for a Brownian mean shift.

    Both inputs have shape ``(..., time_steps, brownian_drivers)``.  The
    returned tensor has the leading batch shape ``...``.  Controls must be
    left-adapted to the increments supplied by the caller; this function
    cannot verify causality from an already materialized tensor.
    """
    _validate_brownian_arrays(controls, proposal_increments, dt)
    stochastic_integral = torch.sum(controls * proposal_increments, dim=(-2, -1))
    energy = dt * torch.sum(controls.square(), dim=(-2, -1))
    return -stochastic_integral - 0.5 * energy


def path_action(
    potential: torch.Tensor | float,
    controls: torch.Tensor,
    proposal_increments: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Return ``Phi + sum(u dB_Q) + 0.5 sum(||u||^2 dt)``.

    ``potential`` may be scalar or broadcastable to the leading batch shape.
    It represents ``Phi=-log(g)`` for a strictly positive soft functional.
    """
    _validate_brownian_arrays(controls, proposal_increments, dt)
    potential_tensor = torch.as_tensor(potential, dtype=controls.dtype, device=controls.device)
    if not torch.isfinite(potential_tensor).all():
        raise ValueError("potential must be finite; use a soft target for path-action training")

    stochastic_integral = torch.sum(controls * proposal_increments, dim=(-2, -1))
    energy = dt * torch.sum(controls.square(), dim=(-2, -1))
    try:
        return potential_tensor + stochastic_integral + 0.5 * energy
    except RuntimeError as exc:
        raise ValueError("potential is not broadcastable to the path batch shape") from exc


def log_tilted_weight(
    potential: torch.Tensor | float,
    controls: torch.Tensor,
    proposal_increments: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Return ``log(g dM/dQ) = -path_action`` in the log domain."""
    return -path_action(potential, controls, proposal_increments, dt)
