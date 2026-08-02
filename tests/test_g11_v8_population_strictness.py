from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.integrate import quad
from scipy.special import logsumexp, ndtr

from src.path_integral.gaussian_mixture_strictness import (
    moment_localized_population_certificate,
)


def test_moment_localized_certificate_matches_closed_formula_in_one_dimension() -> None:
    means = torch.tensor([[0.0], [2.0]], dtype=torch.float64)
    weights = torch.tensor([0.2, 0.8], dtype=torch.float64)
    direction = torch.tensor([1.0], dtype=torch.float64)

    certificate = moment_localized_population_certificate(
        component_means=means,
        weights=weights,
        direction=direction,
        threshold_abs_moment_upper_bound=1.0,
        threshold_moment_order=2.0,
        threshold_radius=2.0,
        residual_radius=0.0,
    )

    expected_eta = 0.75
    expected_log_gap = math.log(expected_eta) + 2.0 * math.log(ndtr(-2.0)) + math.log(ndtr(-4.0))
    assert certificate.residual_dimension == 0
    assert certificate.defensive_mass == pytest.approx(0.2)
    assert certificate.target_residual_ball_probability == 1.0
    assert certificate.localized_target_probability_lower_bound == pytest.approx(expected_eta)
    assert certificate.residual_density_ratio_log_upper_bound == pytest.approx(0.0)
    assert certificate.maximum_projected_component_mean == pytest.approx(2.0)
    assert certificate.maximum_residual_component_norm == pytest.approx(0.0)
    assert certificate.variance_gap_log_lower_bound == pytest.approx(expected_log_gap)
    assert certificate.strict_under_theorem is True


def test_population_bound_is_below_integrated_pointwise_bound() -> None:
    means = torch.tensor([[0.0, 0.0], [1.1, 0.7]], dtype=torch.float64)
    weights = torch.tensor([0.25, 0.75], dtype=torch.float64)
    direction = torch.tensor([1.0, 0.0], dtype=torch.float64)
    threshold_radius = 2.5
    residual_radius = 3.0

    # a(r)=0.4r-1 has E[a(R)^2]=1.16 under the target residual law.
    certificate = moment_localized_population_certificate(
        component_means=means,
        weights=weights,
        direction=direction,
        threshold_abs_moment_upper_bound=1.16,
        threshold_moment_order=2.0,
        threshold_radius=threshold_radius,
        residual_radius=residual_radius,
    )

    def target_density(r: float) -> float:
        return math.exp(-0.5 * r * r) / math.sqrt(2.0 * math.pi)

    def integrated_pointwise_lower_bound(r: float) -> float:
        threshold = 0.4 * r - 1.0
        residual_log_ratios = np.asarray([0.0, 0.7 * r - 0.5 * 0.7**2], dtype=np.float64)
        log_d = float(logsumexp(np.log([0.25, 0.75]) + residual_log_ratios))
        posterior_log_weights = np.log([0.25, 0.75]) + residual_log_ratios - log_d
        log_s = float(
            logsumexp(posterior_log_weights + np.log(ndtr(threshold - np.asarray([0.0, 1.1]))))
        )
        log_one_minus_s = float(
            logsumexp(posterior_log_weights + np.log(ndtr(-threshold + np.asarray([0.0, 1.1]))))
        )
        return target_density(r) * math.exp(
            -log_d + 2.0 * math.log(ndtr(threshold)) + log_one_minus_s - log_s
        )

    integrated_bound, error = quad(
        integrated_pointwise_lower_bound,
        -10.0,
        10.0,
        epsabs=1e-13,
        epsrel=1e-11,
        limit=300,
    )
    certified_bound = math.exp(certificate.variance_gap_log_lower_bound)

    assert error < 1e-10
    assert certificate.strict_under_theorem is True
    assert certified_bound > 0.0
    assert certified_bound <= integrated_bound


def test_nonpositive_localized_mass_returns_a_non_strict_certificate() -> None:
    certificate = moment_localized_population_certificate(
        component_means=torch.tensor([[0.0, 0.0], [0.5, 0.5]], dtype=torch.float64),
        weights=torch.tensor([0.1, 0.9], dtype=torch.float64),
        direction=torch.tensor([1.0, 0.0], dtype=torch.float64),
        threshold_abs_moment_upper_bound=100.0,
        threshold_moment_order=2.0,
        threshold_radius=1.0,
        residual_radius=1.0,
    )

    assert certificate.localized_target_probability_lower_bound == 0.0
    assert certificate.variance_gap_log_lower_bound == -math.inf
    assert certificate.strict_under_theorem is False


def test_population_certificate_rejects_a_nondefensive_mixture() -> None:
    with pytest.raises(ValueError, match="zero-mean defensive"):
        moment_localized_population_certificate(
            component_means=torch.tensor([[1.0, 0.0]], dtype=torch.float64),
            weights=torch.tensor([1.0], dtype=torch.float64),
            direction=torch.tensor([1.0, 0.0], dtype=torch.float64),
            threshold_abs_moment_upper_bound=1.0,
            threshold_moment_order=2.0,
            threshold_radius=2.0,
            residual_radius=2.0,
        )


def test_population_certificate_does_not_round_a_shift_to_the_natural_component() -> None:
    with pytest.raises(ValueError, match="zero-mean defensive"):
        moment_localized_population_certificate(
            component_means=torch.tensor([[1e-15, 0.0]], dtype=torch.float64),
            weights=torch.tensor([1.0], dtype=torch.float64),
            direction=torch.tensor([1.0, 0.0], dtype=torch.float64),
            threshold_abs_moment_upper_bound=1.0,
            threshold_moment_order=2.0,
            threshold_radius=2.0,
            residual_radius=2.0,
        )


def test_population_certificate_requires_double_precision_cpu_inputs() -> None:
    with pytest.raises(TypeError, match="CPU float64"):
        moment_localized_population_certificate(
            component_means=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
            weights=torch.tensor([1.0], dtype=torch.float32),
            direction=torch.tensor([1.0, 0.0], dtype=torch.float32),
            threshold_abs_moment_upper_bound=1.0,
            threshold_moment_order=2.0,
            threshold_radius=2.0,
            residual_radius=2.0,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("threshold_abs_moment_upper_bound", -1.0, "nonnegative"),
        ("threshold_moment_order", 0.0, "moment order"),
        ("threshold_radius", 0.0, "threshold radius"),
        ("residual_radius", -1.0, "residual radius"),
    ],
)
def test_population_certificate_rejects_invalid_localizers(
    field: str, value: float, message: str
) -> None:
    arguments = {
        "component_means": torch.tensor([[0.0, 0.0]], dtype=torch.float64),
        "weights": torch.tensor([1.0], dtype=torch.float64),
        "direction": torch.tensor([1.0, 0.0], dtype=torch.float64),
        "threshold_abs_moment_upper_bound": 1.0,
        "threshold_moment_order": 2.0,
        "threshold_radius": 2.0,
        "residual_radius": 2.0,
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=message):
        moment_localized_population_certificate(**arguments)
