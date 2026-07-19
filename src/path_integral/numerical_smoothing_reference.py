"""Independent CPU reference for the Gaussian-root smoothing specialization.

This is an audit baseline, not the production implementation.  In the monotone
affine case the one-dimensional root-conditioned integral is exactly a normal CDF
or a signed difference of two normal CDFs.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.special import log_ndtr


def scipy_scaled_normal_cdf(
    log_scale: torch.Tensor, threshold: torch.Tensor
) -> torch.Tensor:
    """Evaluate ``exp(log_scale) Phi(threshold)`` through independent SciPy code."""

    _validate(log_scale, threshold)
    log_scale_numpy = log_scale.detach().cpu().numpy()
    threshold_numpy = threshold.detach().cpu().numpy()
    values = np.exp(log_scale_numpy + log_ndtr(threshold_numpy))
    return torch.as_tensor(values, dtype=log_scale.dtype, device=log_scale.device)


def scipy_scaled_normal_cdf_difference(
    log_scale: torch.Tensor,
    fine_threshold: torch.Tensor,
    coarse_threshold: torch.Tensor,
) -> torch.Tensor:
    """Independent signed fine-minus-coarse Gaussian preintegration reference."""

    _validate(log_scale, fine_threshold)
    _validate(log_scale, coarse_threshold)
    fine = fine_threshold.detach().cpu().numpy()
    coarse = coarse_threshold.detach().cpu().numpy()
    high = np.maximum(fine, coarse)
    low = np.minimum(fine, coarse)
    sign = np.sign(fine - coarse)
    log_absolute = np.full_like(high, -np.inf)
    unequal = high != low
    negative = (high <= 0.0) & unequal
    log_high = log_ndtr(high[negative])
    log_low = log_ndtr(low[negative])
    log_absolute[negative] = log_high + np.log(-np.expm1(log_low - log_high))
    positive = (low >= 0.0) & unequal
    log_survival_low = log_ndtr(-low[positive])
    log_survival_high = log_ndtr(-high[positive])
    log_absolute[positive] = log_survival_low + np.log(
        -np.expm1(log_survival_high - log_survival_low)
    )
    central = (low < 0.0) & (high > 0.0) & unequal
    central_difference = np.exp(log_ndtr(high[central])) - np.exp(
        log_ndtr(low[central])
    )
    log_absolute[central] = np.log(central_difference)
    values = sign * np.exp(
        log_scale.detach().cpu().numpy() + log_absolute
    )
    return torch.as_tensor(values, dtype=log_scale.dtype, device=log_scale.device)


def _validate(left: torch.Tensor, right: torch.Tensor) -> None:
    if (
        left.shape != right.shape
        or left.device != right.device
        or left.dtype != right.dtype
        or not left.is_floating_point()
        or not right.is_floating_point()
        or not torch.isfinite(left).all()
        or bool(torch.isnan(right).any())
    ):
        raise ValueError("reference arguments must be matching floating tensors")
