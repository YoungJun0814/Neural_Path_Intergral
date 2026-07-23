"""No-hidden-volatility-floor tests for the rBergomi target law."""

from __future__ import annotations

import math

import pytest
import torch

from src.physics_engine import strict_lognormal_variance


def test_strict_lognormal_variance_matches_the_unfloored_model() -> None:
    factors = torch.tensor([-20.0, -2.0, 0.0, 3.0], dtype=torch.float64)
    observed = strict_lognormal_variance(factors, xi=0.04)
    expected = 0.04 * torch.exp(factors)
    assert torch.allclose(observed, expected, rtol=2e-15, atol=0.0)
    assert observed[0] < 1e-10


@pytest.mark.parametrize(
    "target",
    [torch.finfo(torch.float64).tiny, torch.finfo(torch.float64).max],
)
def test_strict_lognormal_variance_accepts_normal_range_boundaries(
    target: float,
) -> None:
    factor = math.log(target) - math.log(0.04)
    value = strict_lognormal_variance(
        torch.tensor([factor], dtype=torch.float64), xi=0.04
    )
    assert float(value) == pytest.approx(target, rel=1e-12)


@pytest.mark.parametrize(
    "factor",
    [
        math.log(torch.finfo(torch.float64).tiny) - math.log(0.04) - 1.0,
        math.log(torch.finfo(torch.float64).max) - math.log(0.04) + 1.0,
    ],
)
def test_strict_lognormal_variance_fails_instead_of_clamping(factor: float) -> None:
    with pytest.raises(FloatingPointError, match="floating-point range"):
        strict_lognormal_variance(
            torch.tensor([factor], dtype=torch.float64),
            xi=0.04,
        )
