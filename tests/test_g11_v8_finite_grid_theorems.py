"""Executable oracles for the G11 V8 finite-grid theorem stack."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.integrate import quad
from scipy.special import logsumexp, ndtr

from src.path_integral import (
    DiscreteBarrierHitTask,
    GaussianMixtureShiftSpec,
    TerminalThresholdTask,
    build_orthonormal_control_span,
    evaluate_marginal_likelihood,
    scalar_task_threshold,
    scalar_threshold_strictness_certificate,
)


def test_target_proposal_certificate_is_exact_bernoulli_variance() -> None:
    thresholds = torch.tensor(
        [-40.0, -3.0, 0.0, 2.0, 40.0],
        dtype=torch.float64,
    )
    certificate = scalar_threshold_strictness_certificate(
        thresholds,
        torch.zeros(1, dtype=torch.float64),
        torch.zeros((thresholds.numel(), 1), dtype=torch.float64),
        torch.ones(1, dtype=torch.float64),
    )

    expected_log_gap = torch.special.log_ndtr(thresholds) + torch.special.log_ndtr(
        -thresholds
    )
    assert torch.allclose(
        certificate.conditional_variance_gap_log_lower_bound,
        expected_log_gap,
        atol=2e-13,
        rtol=0.0,
    )
    assert certificate.strict_under_theorem.all()
    assert certificate.finite_log_certificate.all()
    assert certificate.maximum_probability_partition_error <= 2e-16


def test_mixture_certificate_is_below_exact_conditional_variance_gap() -> None:
    thresholds = np.asarray([-1.4, -0.1, 1.2], dtype=np.float64)
    projected_means = np.asarray([-1.1, 0.4, 1.3], dtype=np.float64)
    residual_component_logs = np.asarray(
        [
            [0.1, -0.3, 0.5],
            [-0.7, 0.2, 0.1],
            [0.4, -0.2, -0.6],
        ],
        dtype=np.float64,
    )
    weights = np.asarray([0.2, 0.5, 0.3], dtype=np.float64)
    certificate = scalar_threshold_strictness_certificate(
        torch.from_numpy(thresholds),
        torch.from_numpy(projected_means),
        torch.from_numpy(residual_component_logs),
        torch.from_numpy(weights),
    )

    for row, threshold in enumerate(thresholds):
        log_terms = np.log(weights) + residual_component_logs[row]
        residual_log_q_over_p = float(logsumexp(log_terms))
        alpha = np.exp(log_terms - residual_log_q_over_p)
        proposal_event = float(np.sum(alpha * ndtr(threshold - projected_means)))
        proposal_complement = float(
            np.sum(alpha * ndtr(projected_means - threshold))
        )

        def integrand(
            z: float,
            alpha_for_row: np.ndarray = alpha,
        ) -> float:
            component_log_ratio = (
                projected_means * z - 0.5 * np.square(projected_means)
            )
            log_m = float(
                logsumexp(np.log(alpha_for_row) + component_log_ratio)
            )
            log_phi = -0.5 * z * z - 0.5 * math.log(2.0 * math.pi)
            return math.exp(log_phi - log_m)

        inverse_mixture_integral = quad(
            integrand,
            -math.inf,
            float(threshold),
            epsabs=2e-13,
            epsrel=2e-13,
            limit=300,
        )[0]
        target_event = float(ndtr(threshold))
        residual_likelihood = math.exp(-residual_log_q_over_p)
        exact_gap = residual_likelihood**2 * (
            inverse_mixture_integral - target_event**2
        )
        certified_lower = math.exp(
            float(
                certificate.conditional_variance_gap_log_lower_bound[row]
            )
        )

        assert float(certificate.proposal_event_log_probability[row]) == pytest.approx(
            math.log(proposal_event), abs=3e-15
        )
        assert float(
            certificate.proposal_complement_log_probability[row]
        ) == pytest.approx(math.log(proposal_complement), abs=3e-15)
        assert exact_gap > 0.0
        assert certified_lower > 0.0
        assert certified_lower <= exact_gap + 2e-14


def test_certificate_reuses_the_exact_residual_mixture_density() -> None:
    means = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [-0.8, 0.3, 0.2],
            [0.4, -0.2, 0.7],
        ],
        dtype=torch.float64,
    )
    weights = torch.tensor([0.2, 0.5, 0.3], dtype=torch.float64)
    spec = GaussianMixtureShiftSpec(means, weights)
    basis = torch.tensor([[1.0], [0.0], [0.0]], dtype=torch.float64)
    span = build_orthonormal_control_span(spec, basis)
    samples = torch.tensor(
        [[-0.2, 0.4, -0.1], [1.1, -0.7, 0.3], [0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    density = evaluate_marginal_likelihood(samples, spec, span)
    thresholds = torch.tensor([-1.2, 0.3, 1.8], dtype=torch.float64)

    certificate = scalar_threshold_strictness_certificate(
        thresholds,
        span.projected_means[:, 0],
        density.residual_component_log_q_over_p,
        weights,
    )

    assert torch.allclose(
        certificate.residual_log_likelihood,
        density.residual_log_likelihood,
        atol=0.0,
        rtol=0.0,
    )
    assert torch.allclose(
        torch.logsumexp(certificate.component_log_posterior_weight, dim=1),
        torch.zeros(3, dtype=torch.float64),
        atol=3e-16,
        rtol=0.0,
    )
    assert certificate.strict_under_theorem.all()
    assert certificate.maximum_probability_partition_error <= 3e-16


def test_extended_thresholds_are_not_misreported_as_strict() -> None:
    thresholds = torch.tensor(
        [-math.inf, math.inf, 0.0],
        dtype=torch.float64,
    )
    certificate = scalar_threshold_strictness_certificate(
        thresholds,
        torch.tensor([-0.5, 0.7], dtype=torch.float64),
        torch.zeros((3, 2), dtype=torch.float64),
        torch.tensor([0.4, 0.6], dtype=torch.float64),
    )

    assert torch.equal(
        certificate.strict_under_theorem,
        torch.tensor([False, False, True]),
    )
    assert torch.isneginf(
        certificate.conditional_variance_gap_log_lower_bound[:2]
    ).all()
    assert torch.isfinite(
        certificate.conditional_variance_gap_log_lower_bound[2:]
    ).all()


@pytest.mark.parametrize(
    ("threshold", "means", "residual", "weights", "match"),
    [
        (
            torch.tensor([math.nan], dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64),
            torch.zeros((1, 1), dtype=torch.float64),
            torch.ones(1, dtype=torch.float64),
            "NaN",
        ),
        (
            torch.zeros(2, dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64),
            torch.zeros((1, 1), dtype=torch.float64),
            torch.ones(1, dtype=torch.float64),
            "shape",
        ),
        (
            torch.zeros(1, dtype=torch.float64),
            torch.tensor([math.inf], dtype=torch.float64),
            torch.zeros((1, 1), dtype=torch.float64),
            torch.ones(1, dtype=torch.float64),
            "finite",
        ),
        (
            torch.zeros(1, dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64),
            torch.zeros((1, 1), dtype=torch.float64),
            torch.tensor([0.0], dtype=torch.float64),
            "positive",
        ),
    ],
)
def test_invalid_strictness_inputs_fail_closed(
    threshold: torch.Tensor,
    means: torch.Tensor,
    residual: torch.Tensor,
    weights: torch.Tensor,
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        scalar_threshold_strictness_certificate(
            threshold,
            means,
            residual,
            weights,
        )


def test_terminal_threshold_uses_closed_tie_and_is_finite() -> None:
    intercept = torch.tensor(
        [[math.log(100.0), math.log(95.0)]],
        dtype=torch.float64,
    )
    slope = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    task = TerminalThresholdTask(level=95.0)
    threshold = scalar_task_threshold(
        intercept,
        slope,
        step_dt=0.25,
        task=task,
    )
    coordinate = torch.zeros(1, dtype=torch.float64)
    spot = torch.exp(intercept + slope * coordinate.unsqueeze(1))

    assert torch.equal(threshold, torch.zeros_like(threshold))
    assert task.hard_event(spot, 0.25).item() is True
    assert (coordinate <= threshold).item() is True
    assert torch.isfinite(threshold).all()


def test_discrete_barrier_threshold_uses_closed_tie_and_initial_hit_infinity() -> None:
    slope = torch.tensor([[0.0, 0.5, 1.0]], dtype=torch.float64)
    task = DiscreteBarrierHitTask(barrier=90.0)
    not_initially_hit = torch.tensor(
        [[math.log(100.0), math.log(100.0), math.log(90.0)]],
        dtype=torch.float64,
    )
    threshold = scalar_task_threshold(
        not_initially_hit,
        slope,
        step_dt=0.25,
        task=task,
    )
    coordinate = torch.zeros(1, dtype=torch.float64)
    spot = torch.exp(not_initially_hit + slope * coordinate.unsqueeze(1))

    assert threshold.item() == pytest.approx(0.0, abs=0.0)
    assert task.hard_event(spot, 0.25).item() is True
    assert (coordinate <= threshold).item() is True

    initially_hit = not_initially_hit.clone()
    initially_hit[:, 0] = math.log(89.0)
    initial_threshold = scalar_task_threshold(
        initially_hit,
        slope,
        step_dt=0.25,
        task=task,
    )
    assert torch.isposinf(initial_threshold).all()


def test_zero_post_initial_slope_is_rejected() -> None:
    intercept = torch.zeros((1, 2), dtype=torch.float64)
    slope = torch.zeros((1, 2), dtype=torch.float64)

    with pytest.raises(ValueError, match="strictly positive"):
        scalar_task_threshold(
            intercept,
            slope,
            step_dt=0.25,
            task=TerminalThresholdTask(level=1.0),
        )
