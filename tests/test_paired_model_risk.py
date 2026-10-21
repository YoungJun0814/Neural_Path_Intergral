import math
from statistics import NormalDist

import pytest
import torch

from src.path_integral.paired_model_risk import paired_risk_reduction


def _values() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([2., 4., 7., 8., 10., 14.], dtype=torch.float64),
        torch.tensor([1., 3., 6., 6., 11., 12.], dtype=torch.float64),
    )


def test_paired_risk_reduction_matches_manual_covariance_not_independent_se():
    baseline, candidate = _values()
    result = paired_risk_reduction(baseline.log(), candidate.log(), comparisons=3)
    n = baseline.numel()
    mean_baseline, mean_candidate = float(baseline.mean()), float(candidate.mean())
    ratio = mean_candidate / mean_baseline
    covariance = float(((baseline - mean_baseline) * (candidate - mean_candidate)).sum()) / (n - 1)
    independent_variance = (
        float(candidate.var(unbiased=True))
        + ratio**2 * float(baseline.var(unbiased=True))
    ) / (n * mean_baseline**2)
    paired_variance = independent_variance - 2 * ratio * covariance / (n * mean_baseline**2)
    expected_se = math.sqrt(paired_variance)
    z = NormalDist().inv_cdf(1 - .05 / 6)

    assert result["sample_count"] == n
    assert result["relative_m2_reduction"] == pytest.approx(1 - ratio)
    assert result["paired_relative_se"] == pytest.approx(expected_se)
    assert result["paired_relative_se"] < math.sqrt(independent_variance) / 3
    assert result["confidence_z"] == pytest.approx(z)
    assert result["lower_relative_reduction"] == pytest.approx(1 - ratio - z * expected_se)
    assert result["upper_relative_reduction"] == pytest.approx(1 - ratio + z * expected_se)
    assert result["candidate_lower_m2_signal"] == (result["lower_relative_reduction"] > 0)
    assert not result["same_observed_values"]
    assert result["se_unit"] == "independent_fresh_G_draw"
    assert result["status"] == "development_delta_method_not_tail_or_cost_certificate"
    scale = math.exp(result["log_scale"])
    absolute = baseline-candidate
    assert result["scaled_mean_difference"]*scale == pytest.approx(float(absolute.mean()))
    assert result["scaled_difference_se"]*scale == pytest.approx(float(absolute.std(unbiased=True))/math.sqrt(n))
    assert result["scaled_difference_lower"] == pytest.approx(result["scaled_mean_difference"]-z*result["scaled_difference_se"])


def test_paired_risk_exact_noop_has_zero_difference_and_no_improvement_signal():
    logs = torch.tensor([-3000., -2998., -2996., -2994.], dtype=torch.float64)
    result = paired_risk_reduction(logs, logs.clone())
    assert result["sample_count"] == 4
    assert result["relative_m2_reduction"] == 0
    assert result["paired_relative_se"] == 0
    assert result["lower_relative_reduction"] == 0
    assert result["upper_relative_reduction"] == 0
    assert result["same_observed_values"]
    assert not result["candidate_lower_m2_signal"]
    assert not result["candidate_lower_absolute_signal"]
    assert result["scaled_mean_difference"] == result["scaled_difference_se"] == 0.


def test_paired_risk_swap_respects_baseline_denominator_and_delta_method():
    baseline, candidate = _values()
    forward = paired_risk_reduction(baseline.log(), candidate.log())
    reverse = paired_risk_reduction(candidate.log(), baseline.log())
    reduction = forward["relative_m2_reduction"]
    assert reverse["relative_m2_reduction"] == pytest.approx(-reduction / (1 - reduction))
    assert reverse["paired_relative_se"] == pytest.approx(
        forward["paired_relative_se"] / (1 - reduction)**2
    )


@pytest.mark.parametrize("offset", [-10_000., 10_000.])
def test_paired_risk_common_log_scale_does_not_change_diagnostic(offset):
    baseline, candidate = _values()
    original = paired_risk_reduction(baseline.log(), candidate.log())
    shifted = paired_risk_reduction(baseline.log() + offset, candidate.log() + offset)
    for key in (
        "relative_m2_reduction", "paired_relative_se", "lower_relative_reduction",
        "upper_relative_reduction", "confidence_z",
    ):
        assert shifted[key] == pytest.approx(original[key], abs=2e-12, rel=2e-11)
    assert shifted["candidate_lower_m2_signal"] == original["candidate_lower_m2_signal"]
    assert shifted["same_observed_values"] == original["same_observed_values"]


def test_proportional_candidate_retains_pairing_and_stricter_family_critical_value():
    baseline, _ = _values()
    candidate = .8 * baseline
    one = paired_risk_reduction(baseline.log(), candidate.log(), comparisons=1)
    many = paired_risk_reduction(baseline.log(), candidate.log())
    assert many["relative_m2_reduction"] == pytest.approx(.2)
    assert many["paired_relative_se"] < 1e-15
    assert many["candidate_lower_m2_signal"]
    assert many["confidence_z"] > one["confidence_z"]


@pytest.mark.parametrize("invalid", [True, 0, -1, 1.5, "2"])
def test_paired_risk_rejects_invalid_comparisons(invalid):
    baseline, candidate = _values()
    with pytest.raises(ValueError):
        paired_risk_reduction(baseline.log(), candidate.log(), comparisons=invalid)


@pytest.mark.parametrize("invalid", [True, 0., 1., -0.1, float("nan"), float("inf"), "0.05"])
def test_paired_risk_rejects_invalid_alpha(invalid):
    baseline, candidate = _values()
    with pytest.raises(ValueError):
        paired_risk_reduction(baseline.log(), candidate.log(), alpha=invalid)


@pytest.mark.parametrize("which", [0, 1])
@pytest.mark.parametrize("bad_kind", ["short", "empty", "matrix", "float32", "nan", "inf", "grad", "meta"])
def test_paired_risk_rejects_invalid_vectors_in_either_position(which, bad_kind):
    baseline, candidate = _values()
    logs = [baseline.log(), candidate.log()]
    bad = logs[which]
    if bad_kind == "short":
        bad = bad[:1]
    elif bad_kind == "empty":
        bad = bad[:0]
    elif bad_kind == "matrix":
        bad = bad.reshape(2, 3)
    elif bad_kind == "float32":
        bad = bad.float()
    elif bad_kind == "nan":
        bad = bad.clone()
        bad[0] = float("nan")
    elif bad_kind == "inf":
        bad = bad.clone()
        bad[0] = -float("inf")
    elif bad_kind == "grad":
        bad = bad.clone().requires_grad_(True)
    elif bad_kind == "meta":
        bad = bad.to("meta")
    logs[which] = bad
    with pytest.raises(ValueError):
        paired_risk_reduction(*logs)


def test_paired_risk_rejects_mismatched_valid_counts():
    baseline, candidate = _values()
    with pytest.raises(ValueError, match="same sample count"):
        paired_risk_reduction(baseline.log(), candidate[:3].log())


def test_paired_risk_rejects_non_tensor_input():
    with pytest.raises(ValueError):
        paired_risk_reduction([1., 2.], torch.zeros(2, dtype=torch.float64))  # type: ignore[arg-type]


def test_paired_risk_does_not_clip_unrepresentable_baseline_mean():
    baseline = torch.full((2,), -1000., dtype=torch.float64)
    candidate = torch.zeros(2, dtype=torch.float64)
    with pytest.raises(FloatingPointError, match="baseline mean"):
        paired_risk_reduction(baseline, candidate)


def test_paired_risk_lower_tail_quantile_handles_tiny_representable_alpha():
    logs = torch.zeros(2, dtype=torch.float64)
    result = paired_risk_reduction(logs, logs, alpha=1e-100)
    assert math.isfinite(result["confidence_z"])
    assert result["confidence_z"] > 20


def test_paired_risk_rejects_unrepresentable_family_tail_without_clipping():
    logs = torch.zeros(2, dtype=torch.float64)
    with pytest.raises(FloatingPointError, match="quantile"):
        paired_risk_reduction(logs, logs, comparisons=10**1000)
