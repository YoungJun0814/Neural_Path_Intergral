"""Predeclared rate-window and clustered-bootstrap tests."""

from __future__ import annotations

import math

import pytest
import torch

from src.path_integral.rate_analysis import (
    CorrectionRateObservation,
    correction_rate_observation,
    identify_rate_window,
)


def _synthetic_observations() -> list[CorrectionRateObservation]:
    observations: list[CorrectionRateObservation] = []
    for replicate in range(20):
        cluster_factor = math.exp(0.03 * math.sin(replicate))
        for level in range(1, 7):
            h = 2.0**-level
            observations.append(
                CorrectionRateObservation(
                    level=level,
                    replicate=replicate,
                    paths=1000,
                    threshold_l1=cluster_factor * h**0.4,
                    threshold_l2=cluster_factor * h**0.8,
                    raw_second_moment=cluster_factor * h**0.45,
                    dcs_second_moment=cluster_factor * h**0.8,
                    raw_variance=cluster_factor * h**0.45,
                    dcs_variance=cluster_factor * h**0.8,
                    raw_kurtosis=4.0,
                    dcs_kurtosis=3.0,
                    raw_zero_fraction=0.5,
                    dcs_zero_fraction=0.0,
                    raw_positive_fraction=0.25,
                    raw_negative_fraction=0.25,
                    dcs_positive_fraction=0.5,
                    dcs_negative_fraction=0.5,
                    raw_work_units=1.0 / h,
                    dcs_work_units=1.2 / h,
                )
            )
    return observations


def test_rate_window_recovers_synthetic_exponents() -> None:
    analysis = identify_rate_window(
        _synthetic_observations(),
        bootstrap_repetitions=300,
        bootstrap_seed=12345,
    )
    assert analysis.identified
    assert analysis.levels == (1, 2, 3, 4, 5, 6)
    assert analysis.exponents["threshold_l1"] == pytest.approx(0.4, abs=1e-12)
    assert analysis.exponents["threshold_l2"] == pytest.approx(0.8, abs=1e-12)
    assert analysis.exponents["raw_variance"] == pytest.approx(0.45, abs=1e-12)
    assert analysis.exponents["dcs_variance"] == pytest.approx(0.8, abs=1e-12)
    assert analysis.exponents["dcs_second_minus_threshold_l2"] == pytest.approx(
        0.0, abs=1e-12
    )
    difference_interval = analysis.confidence_intervals_95[
        "dcs_second_minus_threshold_l2"
    ]
    assert difference_interval[0] <= 0.0 <= difference_interval[1]


def test_rate_is_unidentified_when_levels_are_missing() -> None:
    observations = [item for item in _synthetic_observations() if item.level != 3]
    analysis = identify_rate_window(
        observations, bootstrap_repetitions=200, bootstrap_seed=9
    )
    assert not analysis.identified
    assert "insufficient consecutive" in analysis.reason


def test_path_diagnostics_are_computed_without_path_level_regression() -> None:
    threshold = torch.tensor([0.2, -0.1, 0.0, 0.3], dtype=torch.float64)
    raw = torch.tensor([1.0, -1.0, 0.0, 0.0], dtype=torch.float64)
    dcs = torch.tensor([0.2, -0.1, 0.01, 0.0], dtype=torch.float64)
    observation = correction_rate_observation(
        level=2,
        replicate=7,
        threshold_difference=threshold,
        raw_correction=raw,
        dcs_correction=dcs,
        raw_work_units=10.0,
        dcs_work_units=12.0,
    )
    assert observation.paths == 4
    assert observation.threshold_l1 == pytest.approx(0.15)
    assert observation.raw_zero_fraction == 0.5
    assert observation.raw_positive_fraction == 0.25
    assert observation.raw_negative_fraction == 0.25


def test_invalid_nonfinite_diagnostics_are_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        correction_rate_observation(
            level=1,
            replicate=0,
            threshold_difference=torch.tensor([0.0, math.nan]),
            raw_correction=torch.zeros(2),
            dcs_correction=torch.zeros(2),
            raw_work_units=1.0,
            dcs_work_units=1.0,
        )
