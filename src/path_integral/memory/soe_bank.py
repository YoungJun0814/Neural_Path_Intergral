"""Positive sum-of-exponentials features for a rough Volterra kernel.

The bank is a controller feature approximation only.  It never replaces the
BLP target simulator and therefore introduces no pricing-law approximation in
the estimator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import nnls


@dataclass(frozen=True)
class SOEKernelFit:
    rates: torch.Tensor
    weights: torch.Tensor
    relative_l2_error: float
    maximum_relative_error: float


def fit_positive_soe_kernel(
    *,
    H: float,
    minimum_lag: float,
    maximum_lag: float,
    terms: int,
    fit_points: int = 256,
    dtype: torch.dtype = torch.float64,
) -> SOEKernelFit:
    r"""Fit ``sqrt(2H) r^(H-1/2)`` by positive exponentials on a lag interval."""
    if not 0.0 < H < 0.5:
        raise ValueError("H must lie in (0, 0.5)")
    if not (
        math.isfinite(minimum_lag)
        and math.isfinite(maximum_lag)
        and 0.0 < minimum_lag < maximum_lag
    ):
        raise ValueError("lags must satisfy 0 < minimum_lag < maximum_lag")
    if terms < 2 or fit_points < max(16, terms):
        raise ValueError("terms and fit_points are too small")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("dtype must be floating point")

    lags = np.geomspace(minimum_lag, maximum_lag, fit_points)
    # Rates cover long memory through the grid-scale singular region.  NNLS
    # fixes nonnegative weights and avoids cancellation-prone lifted states.
    rates = np.geomspace(0.25 / maximum_lag, 4.0 / minimum_lag, terms)
    design = np.exp(-np.outer(lags, rates))
    target = math.sqrt(2.0 * H) * lags ** (H - 0.5)
    # Relative weighting prevents the singular first few points from entirely
    # determining an ordinary least-squares fit.
    scaled_design = design / target[:, None]
    scaled_target = np.ones_like(target)
    weights, _residual = nnls(scaled_design, scaled_target)
    approximation = design @ weights
    relative = (approximation - target) / target
    return SOEKernelFit(
        rates=torch.tensor(rates, dtype=dtype),
        weights=torch.tensor(weights, dtype=dtype),
        relative_l2_error=float(np.sqrt(np.mean(relative**2))),
        maximum_relative_error=float(np.max(np.abs(relative))),
    )


class SOEKernelBank(torch.nn.Module):
    r"""Fixed causal states ``Z_k(t)=int exp(-lambda_k(t-s)) dW_s``."""

    rates: torch.Tensor
    weights: torch.Tensor

    def __init__(
        self,
        *,
        H: float,
        minimum_lag: float,
        maximum_lag: float,
        terms: int = 8,
        fit_points: int = 256,
    ) -> None:
        super().__init__()
        fit = fit_positive_soe_kernel(
            H=H,
            minimum_lag=minimum_lag,
            maximum_lag=maximum_lag,
            terms=terms,
            fit_points=fit_points,
        )
        self.H = float(H)
        self.minimum_lag = float(minimum_lag)
        self.maximum_lag = float(maximum_lag)
        self.terms = int(terms)
        self.relative_l2_error = fit.relative_l2_error
        self.maximum_relative_error = fit.maximum_relative_error
        self.register_buffer("rates", fit.rates)
        self.register_buffer("weights", fit.weights)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return torch.zeros(batch_size, self.terms, device=device, dtype=dtype)

    def update(
        self,
        state: torch.Tensor,
        target_brownian_increment: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Advance states using a cell-average causal Brownian increment."""
        if state.ndim != 2 or state.shape[1] != self.terms:
            raise ValueError("state must have shape (batch, terms)")
        if target_brownian_increment.shape != (state.shape[0],):
            raise ValueError("target increment must have shape (batch,)")
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        rates = self.rates.to(device=state.device, dtype=state.dtype)
        decay = torch.exp(-rates * dt)
        gain = -torch.expm1(-rates * dt) / (rates * dt)
        return decay * state + gain * target_brownian_increment[:, None]

    def weighted_features(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2 or state.shape[1] != self.terms:
            raise ValueError("state must have shape (batch, terms)")
        weights = self.weights.to(device=state.device, dtype=state.dtype)
        scale = torch.sqrt(torch.sum(weights.square())).clamp_min(1e-12)
        return state * weights / scale
