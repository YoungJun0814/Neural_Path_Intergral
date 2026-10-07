from __future__ import annotations

import math

import torch

from src.path_integral.residual_transport import (
    ResidualGaussianMixtureSpec,
    ResidualTransportTrainingConfig,
    evaluate_residual_contribution,
    evaluate_residual_likelihood,
    fit_weighted_residual_transport,
    project_orthogonal,
    sample_residual_mixture,
)


def _direction() -> torch.Tensor:
    value = torch.tensor([1.0, 2.0, 1.0, 0.5, 0.25], dtype=torch.float64)
    return value / torch.linalg.vector_norm(value)


def test_residual_mixture_sampling_likelihood_and_defensive_bound() -> None:
    direction = _direction()
    shifted = project_orthogonal(
        torch.tensor([0.3, -0.2, 0.4, -0.1, 0.2], dtype=torch.float64), direction
    )
    spec = ResidualGaussianMixtureSpec(
        direction=direction,
        means=torch.stack((torch.zeros_like(direction), shifted)),
        weights=torch.tensor([0.2, 0.8], dtype=torch.float64),
    )
    sample = sample_residual_mixture(
        spec,
        20_000,
        gaussian_generator=torch.Generator().manual_seed(1),
        label_generator=torch.Generator().manual_seed(2),
    )
    density = evaluate_residual_likelihood(sample.residual, spec)
    assert sample.maximum_projection_error < 2e-14
    assert density.maximum_projection_error < 2e-14
    assert density.maximum_bound_violation < 1e-12
    normalization_se = float(torch.std(density.likelihood, unbiased=True)) / math.sqrt(
        density.likelihood.numel()
    )
    assert abs(float(torch.mean(density.likelihood)) - 1.0) <= 4.0 * normalization_se


def test_weighted_fit_and_halfspace_oracle_are_exact() -> None:
    dimension = 5
    direction = _direction()
    event_normal = torch.tensor([0.5, -0.3, 0.8, 0.4, -0.6], dtype=torch.float64)
    threshold = -1.8
    loading = float(torch.dot(event_normal, direction))
    assert loading > 0.0
    train_generator = torch.Generator().manual_seed(10)
    target = torch.randn((6000, dimension), dtype=torch.float64, generator=train_generator)
    residual = project_orthogonal(target, direction)
    conditional_argument = (threshold - residual @ event_normal) / loading
    log_g = torch.special.log_ndtr(conditional_argument)
    result = fit_weighted_residual_transport(
        task_id="linear-halfspace",
        target_residuals=residual,
        log_target_weights=log_g,
        direction=direction,
        training_seed=11,
        config=ResidualTransportTrainingConfig(epochs=120, learning_rate=0.04),
    )
    assert result.loss_history[-1] < result.loss_history[0]
    assert result.proposal.exact_likelihood
    assert not result.proposal.self_normalized
    spec = result.proposal.spec()
    sample = sample_residual_mixture(
        spec,
        40_000,
        gaussian_generator=torch.Generator().manual_seed(12),
        label_generator=torch.Generator().manual_seed(13),
    )
    argument = (threshold - sample.residual @ event_normal) / loading
    evaluated = evaluate_residual_contribution(
        sample.residual,
        spec,
        log_conditional_value=torch.special.log_ndtr(argument),
    )
    estimate = float(torch.mean(evaluated.contribution))
    variance = float(torch.var(evaluated.contribution, unbiased=True))
    standard_error = math.sqrt(variance / evaluated.contribution.numel())
    reference = float(
        torch.special.ndtr(
            torch.tensor(threshold / float(torch.linalg.vector_norm(event_normal)))
        )
    )
    assert abs(estimate - reference) <= 4.0 * standard_error

    # Paired raw sampling uses the target conditional coordinate, hence the DCS
    # contribution must have the same mean and no larger variance.
    coordinate = torch.randn(
        sample.residual.shape[0], dtype=torch.float64, generator=torch.Generator().manual_seed(14)
    )
    raw = (coordinate <= argument).to(torch.float64) * evaluated.likelihood.likelihood
    difference = raw - evaluated.contribution
    difference_se = float(torch.std(difference, unbiased=True)) / math.sqrt(difference.numel())
    assert abs(float(torch.mean(difference))) <= 4.0 * difference_se
    assert float(torch.var(evaluated.contribution, unbiased=True)) <= float(
        torch.var(raw, unbiased=True)
    )


def test_nonorthogonal_frozen_mean_is_rejected() -> None:
    direction = _direction()
    try:
        ResidualGaussianMixtureSpec(
            direction=direction,
            means=torch.stack((torch.zeros_like(direction), direction)),
            weights=torch.tensor([0.1, 0.9], dtype=torch.float64),
        )
    except ValueError as error:
        assert "orthogonal" in str(error)
    else:
        raise AssertionError("nonorthogonal mean was accepted")
