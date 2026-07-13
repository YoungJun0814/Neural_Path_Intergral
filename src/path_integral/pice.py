"""Path-integral cross-entropy projections in target Brownian coordinates."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ConstantPICEFit:
    """Self-normalized reverse-KL projection onto constant Brownian drifts."""

    control: torch.Tensor
    effective_sample_size: torch.Tensor
    effective_sample_fraction: torch.Tensor


def fit_constant_pice(
    target_brownian_terminal: torch.Tensor,
    log_target_over_behavior: torch.Tensor,
    horizon: float,
) -> ConstantPICEFit:
    r"""Fit a constant drift by a PICE weighted score equation.

    ``target_brownian_terminal`` is ``B_T^M`` reconstructed on trajectories
    drawn from a behavior proposal.  It may have shape ``(paths,)`` for one
    driver or ``(paths, drivers)`` for an independent Brownian basis.
    ``log_target_over_behavior`` is proportional to
    ``log(g dM/dQ_behavior)``; its unknown normalizing constant cancels.

    The solution is ``u = E_weighted[B_T^M] / T``.  The normalized weights are
    for training only and must not replace the ordinary likelihood-weighted
    estimator used for final probability or price reporting.
    """
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be finite and positive")
    if target_brownian_terminal.ndim not in (1, 2):
        raise ValueError("target_brownian_terminal must have shape (paths,) or (paths, drivers)")
    if log_target_over_behavior.ndim != 1:
        raise ValueError("log_target_over_behavior must have shape (paths,)")
    paths = target_brownian_terminal.shape[0]
    if paths < 2 or log_target_over_behavior.shape[0] != paths:
        raise ValueError("both inputs must contain the same number of at least two paths")
    if not target_brownian_terminal.is_floating_point():
        raise TypeError("target_brownian_terminal must be floating point")
    if not log_target_over_behavior.is_floating_point():
        raise TypeError("log_target_over_behavior must be floating point")
    if target_brownian_terminal.device != log_target_over_behavior.device:
        raise ValueError("both inputs must be on the same device")
    if not torch.isfinite(target_brownian_terminal).all():
        raise ValueError("target_brownian_terminal must be finite")
    if not torch.isfinite(log_target_over_behavior).all():
        raise ValueError("log_target_over_behavior must be finite")

    normalized_weights = torch.softmax(log_target_over_behavior, dim=0)
    if target_brownian_terminal.ndim == 1:
        control = torch.sum(normalized_weights * target_brownian_terminal) / horizon
    else:
        control = torch.sum(normalized_weights[:, None] * target_brownian_terminal, dim=0)
        control = control / horizon
    effective_sample_size = torch.reciprocal(torch.sum(normalized_weights.square()))
    return ConstantPICEFit(
        control=control,
        effective_sample_size=effective_sample_size,
        effective_sample_fraction=effective_sample_size / paths,
    )


def reconstruct_candidate_increments(
    target_increments: torch.Tensor,
    candidate_controls: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    r"""Return candidate ``dB^Q = dB^M - u_candidate dt`` off policy.

    Candidate controls must be evaluated causally on the candidate state
    reconstruction.  The function only performs the coordinate conversion;
    causality remains the caller's responsibility.
    """
    if target_increments.shape != candidate_controls.shape:
        raise ValueError("target_increments and candidate_controls must have identical shapes")
    if target_increments.ndim < 2:
        raise ValueError("expected shape (..., time_steps, brownian_drivers)")
    if not target_increments.is_floating_point() or not candidate_controls.is_floating_point():
        raise TypeError("target_increments and candidate_controls must be floating point")
    if target_increments.device != candidate_controls.device:
        raise ValueError("both inputs must be on the same device")
    if target_increments.dtype != candidate_controls.dtype:
        raise ValueError("both inputs must have the same dtype")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if not torch.isfinite(target_increments).all() or not torch.isfinite(
        candidate_controls
    ).all():
        raise ValueError("target_increments and candidate_controls must be finite")
    return target_increments - candidate_controls * dt
