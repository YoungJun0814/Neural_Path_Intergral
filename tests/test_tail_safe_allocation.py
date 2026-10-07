from __future__ import annotations

import math

import torch

from src.path_integral.tail_safe_allocation import (
    BoundedRange,
    StreamingMoments,
    TailSafeAllocationPolicy,
    bounded_variance_certificate,
    certify_empirical_bernstein_tail_safe_final,
    certify_tail_safe_final,
    collect_streaming_moments,
    empirical_bernstein_variance_certificate,
    plan_empirical_bernstein_tail_safe_allocation,
    plan_tail_safe_allocation,
)


def test_streaming_moments_match_monolithic_and_merge() -> None:
    values = torch.linspace(-0.4, 0.9, 1001, dtype=torch.float64)
    streamed = collect_streaming_moments(
        total_units=values.numel(),
        chunk_units=73,
        evaluator=lambda offset, count: values[offset : offset + count],
    )
    assert streamed.count == values.numel()
    assert math.isclose(streamed.mean, float(torch.mean(values)), abs_tol=1e-15)
    assert math.isclose(
        streamed.sample_variance,
        float(torch.var(values, unbiased=True)),
        rel_tol=2e-14,
    )
    assert math.isclose(streamed.mean_square, float(torch.mean(values.square())), rel_tol=2e-14)
    assert math.isclose(
        streamed.mean_square_sample_variance,
        float(torch.var(values.square(), unbiased=True)),
        rel_tol=2e-14,
    )


def test_zero_variance_pilot_cannot_produce_zero_uncertainty() -> None:
    pilot = StreamingMoments()
    pilot.update(torch.zeros(4096, dtype=torch.float64))
    certificate = bounded_variance_certificate(
        pilot, bounds=BoundedRange(0.0, 1.0), confidence_level=0.95
    )
    assert certificate.empirical_sample_variance == 0.0
    assert certificate.variance_upper > 0.0
    plan = plan_tail_safe_allocation(
        pilot,
        bounds=BoundedRange(0.0, 1.0),
        target_estimator_variance=4e-12,
        unit_kind="iid_path",
        policy=TailSafeAllocationPolicy(maximum_units=100_000),
    )
    assert plan.plugin_required_units == 32
    assert plan.certified_required_units > plan.plugin_required_units
    assert plan.resource_censored


def test_rqmc_minimum_counts_randomizations_not_points() -> None:
    pilot = StreamingMoments()
    pilot.update(torch.full((16,), 0.01, dtype=torch.float64))
    plan = plan_tail_safe_allocation(
        pilot,
        bounds=BoundedRange(0.0, 1.0),
        target_estimator_variance=0.1,
        unit_kind="rqmc_randomization",
        policy=TailSafeAllocationPolicy(minimum_rqmc_randomizations=16),
    )
    assert plan.planned_units >= 16


def test_independent_final_zero_variance_does_not_auto_attain() -> None:
    pilot = StreamingMoments()
    pilot.update(torch.zeros(64, dtype=torch.float64))
    policy = TailSafeAllocationPolicy(
        confidence_level=0.95,
        minimum_iid_units=32,
        maximum_units=128,
    )
    plan = plan_tail_safe_allocation(
        pilot,
        bounds=BoundedRange(0.0, 1.0),
        target_estimator_variance=1e-9,
        unit_kind="iid_path",
        policy=policy,
    )
    final = StreamingMoments()
    final.update(torch.zeros(plan.planned_units, dtype=torch.float64))
    result = certify_tail_safe_final(plan, final, bounds=BoundedRange(0.0, 1.0))
    assert result.final_certificate.empirical_sample_variance == 0.0
    assert result.estimator_variance_upper > 0.0
    assert not result.target_attained


def test_declared_bound_violation_is_rejected() -> None:
    moments = StreamingMoments()
    moments.update(torch.tensor([0.0, 1.1], dtype=torch.float64))
    try:
        bounded_variance_certificate(moments, bounds=BoundedRange(0.0, 1.0))
    except ValueError as error:
        assert "upper bound" in str(error)
    else:
        raise AssertionError("bound violation was accepted")


def test_empirical_bernstein_intersection_has_correct_error_split() -> None:
    values = torch.cat(
        (torch.zeros(900, dtype=torch.float64), torch.ones(100, dtype=torch.float64))
    )
    moments = StreamingMoments()
    moments.update(values)
    certificate = empirical_bernstein_variance_certificate(
        moments, bounds=BoundedRange(0.0, 1.0), confidence_level=0.95
    )
    assert math.isclose(certificate.per_bound_failure_probability, 0.025)
    assert math.isclose(certificate.simultaneous_confidence_level, 0.95)
    assert certificate.mean_square_upper == min(
        certificate.hoeffding_mean_square_upper,
        certificate.empirical_bernstein_mean_square_upper,
    )
    assert certificate.variance_upper == min(
        certificate.mean_square_upper, certificate.popoviciu_variance_upper
    )
    assert certificate.variance_upper >= 0.1 * 0.9


def test_empirical_bernstein_zero_pilot_remains_tail_safe_end_to_end() -> None:
    pilot = StreamingMoments()
    pilot.update(torch.zeros(256, dtype=torch.float64))
    policy = TailSafeAllocationPolicy(
        confidence_level=0.95,
        minimum_iid_units=32,
        maximum_units=512,
    )
    plan = plan_empirical_bernstein_tail_safe_allocation(
        pilot,
        bounds=BoundedRange(0.0, 5.0),
        target_estimator_variance=1e-8,
        unit_kind="iid_path",
        policy=policy,
    )
    assert plan.pilot_certificate.variance_upper > 0.0
    assert plan.resource_censored
    final = StreamingMoments()
    final.update(torch.zeros(plan.planned_units, dtype=torch.float64))
    result = certify_empirical_bernstein_tail_safe_final(plan, final, bounds=BoundedRange(0.0, 5.0))
    assert result.estimator_variance_upper > 0.0
    assert not result.target_attained
