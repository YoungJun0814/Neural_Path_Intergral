"""Extreme-tail tests for the V11 stable Gaussian arithmetic module."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.special import log_ndtr, ndtr

from src.path_integral.stable_gaussian import (
    scaled_normal_cdf,
    scaled_normal_cdf_difference,
    signed_log_normal_cdf_difference,
    stable_normal_cdf_difference,
)


def _reference(left: float, right: float) -> float:
    if left == right:
        return 0.0
    sign = 1.0 if left > right else -1.0
    high, low = max(left, right), min(left, right)
    if high <= 0.0:
        magnitude = math.exp(float(log_ndtr(high))) * (
            -math.expm1(float(log_ndtr(low) - log_ndtr(high)))
        )
    elif low >= 0.0:
        magnitude = math.exp(float(log_ndtr(-low))) * (
            -math.expm1(float(log_ndtr(-high) - log_ndtr(-low)))
        )
    else:
        magnitude = float(ndtr(high) - ndtr(low))
    return sign * magnitude


def test_signed_cdf_difference_matches_scipy_across_extreme_cases() -> None:
    pairs = [
        (-40.0, -41.0),
        (-12.0, -12.0001),
        (-2.0, -7.0),
        (2.0, -3.0),
        (12.0001, 12.0),
        (41.0, 40.0),
        (math.inf, -math.inf),
        (-math.inf, math.inf),
        (math.inf, math.inf),
        (-math.inf, -math.inf),
    ]
    left = torch.tensor([pair[0] for pair in pairs], dtype=torch.float64)
    right = torch.tensor([pair[1] for pair in pairs], dtype=torch.float64)
    actual = stable_normal_cdf_difference(left, right).numpy()
    expected = np.asarray([_reference(*pair) for pair in pairs])
    assert np.allclose(actual, expected, rtol=3e-13, atol=1e-300)
    sign, log_abs = signed_log_normal_cdf_difference(left, right)
    reconstructed = (sign * torch.exp(log_abs)).numpy()
    assert np.allclose(reconstructed, expected, rtol=3e-13, atol=1e-300)
    assert np.isfinite(actual).all()


def test_scaled_cdf_and_difference_remain_finite_in_tails() -> None:
    log_scale = torch.tensor([10.0, 20.0, -30.0, 0.0], dtype=torch.float64)
    argument = torch.tensor([-12.0, -15.0, 8.0, math.inf], dtype=torch.float64)
    actual = scaled_normal_cdf(log_scale, argument)
    expected = torch.exp(log_scale + torch.special.log_ndtr(argument))
    assert torch.allclose(actual, expected, atol=0.0, rtol=2e-15)

    left = torch.tensor([-12.0, 15.0, 8.0, math.inf], dtype=torch.float64)
    right = torch.tensor([-13.0, 14.0, 8.1, math.inf], dtype=torch.float64)
    difference = scaled_normal_cdf_difference(log_scale, left, right)
    expected_difference = torch.exp(log_scale) * stable_normal_cdf_difference(left, right)
    assert torch.allclose(difference, expected_difference, atol=1e-300, rtol=3e-13)
    assert torch.isfinite(difference).all()


def test_invalid_gaussian_arguments_are_rejected() -> None:
    with pytest.raises(ValueError, match="identical"):
        stable_normal_cdf_difference(
            torch.zeros(2, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)
        )
    with pytest.raises(ValueError, match="NaNs"):
        stable_normal_cdf_difference(
            torch.tensor([math.nan], dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="finite"):
        scaled_normal_cdf_difference(
            torch.tensor([math.inf], dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64),
            torch.ones(1, dtype=torch.float64),
        )
