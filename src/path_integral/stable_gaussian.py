"""Numerically stable standard-Gaussian CDF primitives.

The signed-difference routines deliberately support extended-real thresholds.  They
never form ``Phi(left) - Phi(right)`` directly when both arguments are in the same
tail, which avoids catastrophic cancellation in multilevel corrections.
"""

from __future__ import annotations

import math

import torch


def _validate_matching_floats(left: torch.Tensor, right: torch.Tensor) -> None:
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        raise TypeError("arguments must be torch tensors")
    if left.shape != right.shape:
        raise ValueError("arguments must have identical shapes")
    if left.device != right.device or left.dtype != right.dtype:
        raise ValueError("arguments must share device and dtype")
    if not left.is_floating_point() or not right.is_floating_point():
        raise TypeError("arguments must be floating point")
    if bool(torch.isnan(left).any()) or bool(torch.isnan(right).any()):
        raise ValueError("arguments must not contain NaNs")


def signed_log_normal_cdf_difference(
    left: torch.Tensor,
    right: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sign and log magnitude of ``Phi(left) - Phi(right)``.

    Equal finite or infinite arguments return sign zero and log magnitude negative
    infinity.  All other extended-real combinations are evaluated by normal-CDF or
    survival-function identities without an ``inf - inf`` operation.
    """

    _validate_matching_floats(left, right)
    high = torch.maximum(left, right)
    low = torch.minimum(left, right)
    equal = left == right
    sign = torch.where(equal, torch.zeros_like(left), torch.sign(left - right))
    log_abs = torch.full_like(left, -math.inf)

    negative = (high <= 0.0) & ~equal
    if bool(negative.any()):
        log_high = torch.special.log_ndtr(high[negative])
        log_low = torch.special.log_ndtr(low[negative])
        log_abs[negative] = log_high + torch.log(-torch.expm1(log_low - log_high))

    positive = (low >= 0.0) & ~equal
    if bool(positive.any()):
        log_survival_low = torch.special.log_ndtr(-low[positive])
        log_survival_high = torch.special.log_ndtr(-high[positive])
        log_abs[positive] = log_survival_low + torch.log(
            -torch.expm1(log_survival_high - log_survival_low)
        )

    central = (low < 0.0) & (high > 0.0) & ~equal
    if bool(central.any()):
        difference = torch.special.ndtr(high[central]) - torch.special.ndtr(low[central])
        log_abs[central] = torch.log(difference)

    if bool(torch.isnan(log_abs).any()):
        raise FloatingPointError("normal CDF difference produced NaN")
    return sign, log_abs


def stable_normal_cdf_difference(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Return the stable signed value ``Phi(left)-Phi(right)``."""

    sign, log_abs = signed_log_normal_cdf_difference(left, right)
    result = sign * torch.exp(log_abs)
    if not torch.isfinite(result).all():
        raise FloatingPointError("normal CDF difference became nonfinite")
    return result


def scaled_normal_cdf_difference(
    log_scale: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Return ``exp(log_scale) * (Phi(left)-Phi(right))`` stably."""

    _validate_matching_floats(left, right)
    if log_scale.shape != left.shape:
        raise ValueError("log scale and CDF arguments must have identical shapes")
    if log_scale.device != left.device or log_scale.dtype != left.dtype:
        raise ValueError("log scale and CDF arguments must share device and dtype")
    if not log_scale.is_floating_point():
        raise TypeError("log scale must be floating point")
    if not torch.isfinite(log_scale).all():
        raise ValueError("log scale must be finite")
    sign, log_abs = signed_log_normal_cdf_difference(left, right)
    result = sign * torch.exp(log_scale + log_abs)
    if not torch.isfinite(result).all():
        raise FloatingPointError("scaled normal CDF difference became nonfinite")
    return result


def scaled_normal_cdf(
    log_scale: torch.Tensor,
    argument: torch.Tensor,
) -> torch.Tensor:
    """Return ``exp(log_scale) * Phi(argument)`` in the log domain."""

    if not isinstance(log_scale, torch.Tensor) or not isinstance(argument, torch.Tensor):
        raise TypeError("arguments must be torch tensors")
    if log_scale.shape != argument.shape:
        raise ValueError("log scale and argument must have identical shapes")
    if log_scale.device != argument.device or log_scale.dtype != argument.dtype:
        raise ValueError("log scale and argument must share device and dtype")
    if not log_scale.is_floating_point() or not argument.is_floating_point():
        raise TypeError("arguments must be floating point")
    if not torch.isfinite(log_scale).all() or bool(torch.isnan(argument).any()):
        raise ValueError("log scale must be finite and argument must not contain NaNs")
    result = torch.exp(log_scale + torch.special.log_ndtr(argument))
    if not torch.isfinite(result).all():
        raise FloatingPointError("scaled normal CDF became nonfinite")
    return result
