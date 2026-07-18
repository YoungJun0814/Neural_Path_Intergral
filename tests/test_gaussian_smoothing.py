"""Algebraic and numerical oracles for monotone Gaussian smoothing."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.integrate import quad
from scipy.special import ndtr

from src.path_integral.gaussian_smoothing import (
    decompose_gaussian_shift,
    downside_excursion_thresholds,
    orthogonal_gaussian_residual,
    positive_flat_direction,
    stable_normal_cdf_difference,
)
from src.path_integral.path_functionals import DownsideExcursionTask


def _task(minimum_occupation: float = 0.2) -> DownsideExcursionTask:
    return DownsideExcursionTask(
        hit_barrier=90.0,
        stress_level=95.0,
        minimum_occupation=minimum_occupation,
        hit_scale=2.0,
        occupation_scale=0.05,
    )


def test_gaussian_direction_decomposition_is_orthogonal_and_reconstructs() -> None:
    torch.manual_seed(9101)
    normals = torch.randn(128, 17, dtype=torch.float64)
    direction = positive_flat_direction(17, device="cpu")
    coordinate, residual = orthogonal_gaussian_residual(normals, direction)
    reconstructed = coordinate.unsqueeze(1) * direction + residual
    assert torch.allclose(reconstructed, normals, atol=2e-15, rtol=0.0)
    assert float(torch.max(torch.abs(residual @ direction))) < 2e-15

    shift = torch.linspace(-0.5, 0.7, 17, dtype=torch.float64)
    split = decompose_gaussian_shift(shift, direction)
    assert torch.allclose(
        split.parallel_coefficient * direction + split.orthogonal_shift,
        shift,
        atol=2e-15,
        rtol=0.0,
    )
    assert abs(float(torch.dot(split.orthogonal_shift, direction))) < 2e-15


@pytest.mark.parametrize(
    "direction",
    (
        torch.tensor([1.0, 0.0], dtype=torch.float64),
        torch.tensor([1.0, -1.0], dtype=torch.float64) / math.sqrt(2.0),
        torch.tensor([1.0, 1.0], dtype=torch.float64),
        torch.tensor([math.nan, 1.0], dtype=torch.float64),
    ),
)
def test_invalid_smoothing_directions_are_rejected(direction: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        orthogonal_gaussian_residual(torch.zeros(4, 2, dtype=torch.float64), direction)


def test_downside_threshold_is_pathwise_equivalent_to_the_declared_event() -> None:
    torch.manual_seed(9102)
    paths = 5_000
    steps = 8
    step_dt = 0.1
    coordinate = torch.randn(paths, dtype=torch.float64)
    innovations = 0.03 * torch.randn(paths, steps, dtype=torch.float64)
    intercept = torch.cat(
        (
            torch.full((paths, 1), math.log(100.0), dtype=torch.float64),
            math.log(100.0) + torch.cumsum(innovations - 0.012, dim=1),
        ),
        dim=1,
    )
    increments = 0.012 + 0.02 * torch.rand(paths, steps, dtype=torch.float64)
    slope = torch.cat(
        (
            torch.zeros(paths, 1, dtype=torch.float64),
            torch.cumsum(increments, dim=1),
        ),
        dim=1,
    )
    task = _task()
    threshold = downside_excursion_thresholds(intercept, slope, step_dt=step_dt, task=task)
    spot = torch.exp(intercept + slope * coordinate.unsqueeze(1))
    direct = task.hard_event(spot, step_dt)
    assert torch.equal(direct, coordinate <= threshold.combined)
    assert threshold.required_occupation_count == 2


def test_threshold_handles_initial_hit_and_impossible_occupation() -> None:
    intercept = torch.tensor(
        [[math.log(80.0), math.log(100.0), math.log(100.0)]], dtype=torch.float64
    )
    slope = torch.tensor([[0.0, 0.2, 0.4]], dtype=torch.float64)
    possible = downside_excursion_thresholds(intercept, slope, step_dt=0.1, task=_task(0.1))
    impossible = downside_excursion_thresholds(intercept, slope, step_dt=0.1, task=_task(0.3))
    assert torch.isposinf(possible.hit).all()
    assert torch.isneginf(impossible.occupation).all()
    assert torch.isneginf(impossible.combined).all()


def _reference_cdf_difference(left: float, right: float) -> float:
    sign = 1.0 if left >= right else -1.0
    high, low = max(left, right), min(left, right)
    if high <= 0.0:
        magnitude = float(ndtr(high) - ndtr(low))
    elif low >= 0.0:
        magnitude = float(ndtr(-low) - ndtr(-high))
    else:
        magnitude = float(ndtr(high) - ndtr(low))
    return sign * magnitude


def test_stable_normal_cdf_difference_covers_both_tails_and_infinities() -> None:
    left = torch.tensor(
        [-11.0, 12.0, 2.0, 1.5, math.inf, -math.inf, math.inf],
        dtype=torch.float64,
    )
    right = torch.tensor(
        [-12.0, 11.0, -1.0, 1.5, math.inf, -math.inf, -math.inf],
        dtype=torch.float64,
    )
    actual = stable_normal_cdf_difference(left, right).numpy()
    expected = np.asarray(
        [
            _reference_cdf_difference(-11.0, -12.0),
            _reference_cdf_difference(12.0, 11.0),
            _reference_cdf_difference(2.0, -1.0),
            0.0,
            0.0,
            0.0,
            1.0,
        ]
    )
    assert np.allclose(actual, expected, rtol=2e-13, atol=1e-300)
    assert np.isfinite(actual).all()


def test_shifted_truncated_gaussian_integral_matches_closed_form() -> None:
    standard_normal_density = 1.0 / math.sqrt(2.0 * math.pi)
    for threshold, shift in ((-2.3, -1.1), (-0.4, 0.8), (1.7, 2.2)):
        numeric, error = quad(
            lambda z, resolved_shift=shift: (
                standard_normal_density
                * math.exp(
                    -0.5 * z * z - resolved_shift * z - 0.5 * resolved_shift * resolved_shift
                )
            ),
            -math.inf,
            threshold,
            epsabs=1e-13,
            epsrel=1e-13,
        )
        assert error < 1e-11
        assert numeric == pytest.approx(float(ndtr(threshold + shift)), abs=3e-14)
