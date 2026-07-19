"""Oracle and property tests for generic Gaussian control-span marginalization."""

from __future__ import annotations

import math

import pytest
import torch

from src.path_integral.gaussian_span_marginalization import (
    GaussianMixtureShiftSpec,
    build_orthonormal_control_span,
    component_log_q_over_p,
    control_span_from_vectors,
    evaluate_marginal_likelihood,
    evaluate_marginalized_function,
    linear_threshold_conditional_probability,
    sample_gaussian_mixture,
)


def _orthonormal_basis(dimension: int, rank: int, seed: int) -> torch.Tensor:
    if rank == 0:
        return torch.empty((dimension, 0), dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn((dimension, rank), dtype=torch.float64, generator=generator)
    basis, _upper = torch.linalg.qr(matrix, mode="reduced")
    return basis


def test_density_factorization_and_defensive_bounds_over_500_random_cases() -> None:
    generator = torch.Generator().manual_seed(11_000_101)
    for case in range(500):
        dimension = (2, 3, 7, 32)[case % 4]
        components = (1, 2, 5)[case % 3]
        rank = case % min(3, dimension + 1)
        means = 0.8 * torch.randn(
            (components, dimension), dtype=torch.float64, generator=generator
        )
        means[0] = 0.0
        weights = torch.rand(components, dtype=torch.float64, generator=generator) + 0.05
        weights = weights / torch.sum(weights)
        spec = GaussianMixtureShiftSpec(means=means, weights=weights)
        span = build_orthonormal_control_span(
            spec,
            _orthonormal_basis(dimension, rank, 11_100_000 + case),
        )
        sample = sample_gaussian_mixture(
            spec,
            24,
            gaussian_generator=torch.Generator().manual_seed(11_200_000 + case),
            label_generator=torch.Generator().manual_seed(11_300_000 + case),
        )
        evaluation = evaluate_marginal_likelihood(
            sample.target_coordinates, spec, span
        )
        assert evaluation.maximum_component_reconstruction_error <= 2e-13
        assert evaluation.maximum_mixture_reconstruction_error <= 2e-13
        assert evaluation.maximum_sample_residual_projection <= 2e-13
        assert evaluation.maximum_full_bound_violation <= 2e-13
        assert evaluation.maximum_residual_bound_violation <= 2e-13
        assert evaluation.defensive_weight == pytest.approx(float(weights[0]), abs=2e-15)
        assert torch.allclose(
            evaluation.residual + evaluation.coordinate @ span.basis.T,
            sample.target_coordinates,
            atol=4e-15,
            rtol=0.0,
        )


def test_marginal_likelihood_is_invariant_to_basis_rotation_and_sign() -> None:
    generator = torch.Generator().manual_seed(11_400_101)
    means = torch.randn((4, 8), dtype=torch.float64, generator=generator)
    means[0] = 0.0
    weights = torch.tensor([0.2, 0.3, 0.1, 0.4], dtype=torch.float64)
    spec = GaussianMixtureShiftSpec(means, weights)
    basis = _orthonormal_basis(8, 3, 11_400_102)
    rotation = _orthonormal_basis(3, 3, 11_400_103)
    span = build_orthonormal_control_span(spec, basis)
    rotated = build_orthonormal_control_span(spec, basis @ rotation)
    signed = build_orthonormal_control_span(
        spec, basis * torch.tensor([-1.0, 1.0, -1.0], dtype=torch.float64)
    )
    samples = sample_gaussian_mixture(
        spec,
        2048,
        gaussian_generator=torch.Generator().manual_seed(11_400_104),
        label_generator=torch.Generator().manual_seed(11_400_105),
    ).target_coordinates
    reference = evaluate_marginal_likelihood(samples, spec, span)
    for candidate in (rotated, signed):
        actual = evaluate_marginal_likelihood(samples, spec, candidate)
        assert torch.allclose(
            actual.residual_log_q_over_p,
            reference.residual_log_q_over_p,
            atol=3e-14,
            rtol=0.0,
        )
        assert torch.allclose(actual.residual, reference.residual, atol=3e-14, rtol=0.0)


def test_linear_threshold_oracle_is_unbiased_and_rao_blackwellized() -> None:
    dimension = 7
    means = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.8, 0.3, -0.2, 0.0, 0.4, -0.1, 0.2],
            [0.2, -0.5, 0.1, -0.3, 0.0, 0.2, -0.4],
        ],
        dtype=torch.float64,
    )
    spec = GaussianMixtureShiftSpec(
        means, torch.tensor([0.2, 0.5, 0.3], dtype=torch.float64)
    )
    span = build_orthonormal_control_span(spec, _orthonormal_basis(dimension, 2, 11_500_101))
    sample = sample_gaussian_mixture(
        spec,
        200_000,
        gaussian_generator=torch.Generator().manual_seed(11_500_102),
        label_generator=torch.Generator().manual_seed(11_500_103),
    )
    density = evaluate_marginal_likelihood(sample.target_coordinates, spec, span)
    event_normal = torch.tensor([0.4, -0.2, 0.7, 0.1, -0.3, 0.5, -0.1], dtype=torch.float64)
    threshold = -0.65
    raw_value = (
        sample.target_coordinates @ event_normal <= threshold
    ).to(torch.float64)
    conditional = linear_threshold_conditional_probability(
        density.residual,
        span,
        event_normal,
        threshold,
    )
    estimate = evaluate_marginalized_function(
        density,
        raw_value=raw_value,
        conditional_target_value=conditional,
    )
    truth = float(
        torch.special.ndtr(
            torch.tensor(threshold / float(torch.linalg.vector_norm(event_normal)))
        )
    )
    for contribution in (
        estimate.raw_contribution,
        estimate.marginalized_contribution,
    ):
        standard_error = float(contribution.std(unbiased=True)) / math.sqrt(
            contribution.numel()
        )
        assert abs(float(contribution.mean()) - truth) <= max(2e-4, 4.0 * standard_error)
    difference = estimate.marginalized_contribution - estimate.raw_contribution
    paired_standard_error = float(difference.std(unbiased=True)) / math.sqrt(
        difference.numel()
    )
    assert abs(float(difference.mean())) <= 4.0 * paired_standard_error
    assert float(estimate.marginalized_contribution.var(unbiased=True)) <= float(
        estimate.raw_contribution.var(unbiased=True)
    )


def test_zero_rank_and_full_rank_spans_have_expected_limits() -> None:
    means = torch.tensor([[0.0, 0.0], [-0.4, 0.7]], dtype=torch.float64)
    spec = GaussianMixtureShiftSpec(means, torch.tensor([0.25, 0.75], dtype=torch.float64))
    points = torch.tensor([[0.2, -0.3], [1.1, 0.7]], dtype=torch.float64)

    zero = build_orthonormal_control_span(spec, torch.empty((2, 0), dtype=torch.float64))
    zero_eval = evaluate_marginal_likelihood(points, spec, zero)
    assert torch.equal(zero_eval.residual, points)
    assert torch.allclose(
        zero_eval.full_log_q_over_p,
        zero_eval.residual_log_q_over_p,
        atol=0.0,
        rtol=0.0,
    )

    full = build_orthonormal_control_span(spec, torch.eye(2, dtype=torch.float64))
    full_eval = evaluate_marginal_likelihood(points, spec, full)
    assert torch.allclose(full_eval.residual, torch.zeros_like(points), atol=0.0, rtol=0.0)
    assert torch.allclose(
        full_eval.residual_log_q_over_p,
        torch.zeros(points.shape[0], dtype=torch.float64),
        atol=2e-16,
        rtol=0.0,
    )
    assert torch.allclose(full_eval.residual_likelihood, torch.ones(2, dtype=torch.float64))


def test_control_span_from_vectors_recovers_rank_and_orientation() -> None:
    means = torch.zeros((1, 4), dtype=torch.float64)
    spec = GaussianMixtureShiftSpec(means, torch.ones(1, dtype=torch.float64))
    vectors = torch.tensor(
        [[-1.0, -2.0, 0.0, 0.0], [-2.0, -4.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    span = control_span_from_vectors(spec, vectors)
    assert span.rank == 1
    pivot = int(torch.argmax(torch.abs(span.basis[:, 0])))
    assert float(span.basis[pivot, 0]) > 0.0
    projection = vectors - (vectors @ span.basis) @ span.basis.T
    assert float(torch.max(torch.abs(projection))) <= 2e-15


def test_component_density_matches_torch_distribution_ratio() -> None:
    samples = torch.tensor([[0.2, -0.1], [-1.0, 0.7]], dtype=torch.float64)
    means = torch.tensor([[0.0, 0.0], [0.4, -0.8]], dtype=torch.float64)
    actual = component_log_q_over_p(samples, means)
    standard = torch.distributions.MultivariateNormal(
        torch.zeros(2, dtype=torch.float64), torch.eye(2, dtype=torch.float64)
    )
    shifted = torch.distributions.MultivariateNormal(means[1], torch.eye(2, dtype=torch.float64))
    expected_shifted = shifted.log_prob(samples) - standard.log_prob(samples)
    assert torch.allclose(actual[:, 0], torch.zeros(2, dtype=torch.float64), atol=0.0, rtol=0.0)
    assert torch.allclose(actual[:, 1], expected_shifted, atol=3e-16, rtol=0.0)


@pytest.mark.parametrize(
    ("means", "weights", "match"),
    [
        (torch.zeros(2, dtype=torch.float64), torch.ones(1, dtype=torch.float64), "shape"),
        (
            torch.zeros((2, 3), dtype=torch.float64),
            torch.tensor([0.5, -0.5], dtype=torch.float64),
            "positive",
        ),
        (
            torch.zeros((2, 3), dtype=torch.float64),
            torch.tensor([0.4, 0.4], dtype=torch.float64),
            "sum",
        ),
    ],
)
def test_invalid_mixture_specs_are_rejected(
    means: torch.Tensor,
    weights: torch.Tensor,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        GaussianMixtureShiftSpec(means, weights)


def test_nonorthonormal_basis_is_rejected() -> None:
    spec = GaussianMixtureShiftSpec(
        torch.zeros((1, 3), dtype=torch.float64), torch.ones(1, dtype=torch.float64)
    )
    with pytest.raises(ValueError, match="orthonormal"):
        build_orthonormal_control_span(
            spec,
            torch.tensor([[1.0], [1.0], [0.0]], dtype=torch.float64),
        )
