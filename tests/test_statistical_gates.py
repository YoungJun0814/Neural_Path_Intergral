"""Exact edge-case and deterministic decision tests for V5 statistical gates."""

from __future__ import annotations

import math

import pytest
import torch

from src.path_integral import (
    conservative_bernoulli_variance_upper,
    exact_binomial_probability_interval,
    heavy_tail_diagnostics,
    holm_rejections,
    paired_log_work_summary,
    paired_power_forecast,
    reference_agreement,
)


def test_zero_hit_binomial_interval_does_not_claim_zero_variance() -> None:
    interval = exact_binomial_probability_interval(0, 100, confidence_level=0.95)
    expected_upper = 1.0 - 0.025 ** (1.0 / 100.0)
    assert interval.lower == 0.0
    assert interval.upper == pytest.approx(expected_upper)
    assert conservative_bernoulli_variance_upper(interval) > 0.0


def test_all_hit_interval_and_interior_interval_are_valid() -> None:
    all_hit = exact_binomial_probability_interval(50, 50)
    interior = exact_binomial_probability_interval(7, 50)
    assert all_hit.upper == 1.0 and all_hit.lower < 1.0
    assert interior.lower < 7 / 50 < interior.upper
    assert conservative_bernoulli_variance_upper(interior) >= (7 / 50) * (43 / 50)


def test_paired_log_work_summary_uses_cluster_pairing() -> None:
    method = [5.0, 10.0, 20.0, 40.0]
    baseline = [10.0, 20.0, 40.0, 80.0]
    summary = paired_log_work_summary(method, baseline)
    assert summary.geometric_mean_baseline_over_method == pytest.approx(2.0)
    assert summary.method_better_fraction == 1.0
    assert summary.standard_error == pytest.approx(0.0, abs=1e-15)
    assert summary.confidence_interval[0] == pytest.approx(math.log(2.0))


def test_holm_stops_after_first_nonrejection() -> None:
    decisions = holm_rejections({"a": 0.001, "b": 0.02, "c": 0.04}, familywise_alpha=0.05)
    assert decisions == {"a": True, "b": True, "c": True}
    stopped = holm_rejections({"a": 0.001, "b": 0.03, "c": 0.031}, familywise_alpha=0.05)
    assert stopped == {"a": True, "b": False, "c": False}


def test_signed_contributions_do_not_receive_a_misleading_positive_weight_ess() -> None:
    signed = heavy_tail_diagnostics(torch.tensor([-1.0, 0.5, 1.5]))
    positive = heavy_tail_diagnostics(torch.tensor([1.0, 2.0, 3.0]))
    assert signed.positive_weight_ess is None
    assert positive.positive_weight_ess == pytest.approx(36.0 / 14.0)
    assert 0.0 < signed.maximum_absolute_to_sum_absolute < 1.0


def test_reference_agreement_uses_combined_uncertainty() -> None:
    accepted = reference_agreement(0.10, 0.01, 0.12, 0.01, maximum_z_score=4.0)
    rejected = reference_agreement(0.10, 0.001, 0.12, 0.001, maximum_z_score=4.0)
    assert accepted.agrees
    assert not rejected.agrees
    assert rejected.combined_z_score > accepted.combined_z_score


def test_power_forecast_increases_with_noise_and_decreases_with_effect() -> None:
    base = paired_power_forecast(mean_log_effect=0.2, standard_deviation=0.4)
    noisy = paired_power_forecast(mean_log_effect=0.2, standard_deviation=0.8)
    strong = paired_power_forecast(mean_log_effect=0.4, standard_deviation=0.4)
    assert (
        noisy.required_clusters_normal_approximation > base.required_clusters_normal_approximation
    )
    assert (
        strong.required_clusters_normal_approximation < base.required_clusters_normal_approximation
    )
