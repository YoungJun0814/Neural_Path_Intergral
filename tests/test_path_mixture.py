"""Correctness tests for exact path-integral mixture likelihoods."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.path_integral import (
    all_expert_log_q_over_p,
    gaussian_single_drift_second_moment,
    gaussian_symmetric_mixture_log_q_over_p,
    gaussian_symmetric_mixture_second_moment,
    gaussian_two_tail_probability,
    log_mixture_q_over_p,
    sample_mixture_labels,
    selected_component_log_p_over_q,
)


def test_all_expert_density_uses_target_coordinate() -> None:
    target = torch.tensor(
        [[[0.2, -0.1], [0.05, 0.3]], [[-0.4, 0.2], [0.1, -0.2]]],
        dtype=torch.float64,
    )
    controls = torch.tensor(
        [
            [[[0.5, -0.2], [0.1, 0.3]], [[-0.5, 0.2], [-0.1, -0.3]]],
            [[[0.4, 0.1], [0.2, -0.2]], [[-0.4, -0.1], [-0.2, 0.2]]],
        ],
        dtype=torch.float64,
    )
    dt = 0.25
    expected = torch.sum(controls * target[:, None], dim=(-2, -1))
    expected = expected - 0.5 * dt * torch.sum(controls.square(), dim=(-2, -1))
    assert torch.equal(all_expert_log_q_over_p(controls, target, dt), expected)


def test_log_mixture_matches_symmetric_gaussian_oracle() -> None:
    terminal = torch.linspace(-7.0, 7.0, 1001, dtype=torch.float64)
    drift = 2.3
    horizon = 0.75
    components = torch.stack(
        (
            drift * terminal - 0.5 * drift * drift * horizon,
            -drift * terminal - 0.5 * drift * drift * horizon,
        ),
        dim=-1,
    )
    actual = log_mixture_q_over_p(
        components, torch.tensor([0.5, 0.5], dtype=torch.float64)
    )
    expected = gaussian_symmetric_mixture_log_q_over_p(
        terminal.numpy(), drift=drift, horizon=horizon
    )
    assert np.max(np.abs(actual.numpy() - expected)) < 1e-14


def test_selected_component_weight_is_label_preserving() -> None:
    component = torch.tensor([[1.0, 2.0], [-0.5, 0.75]], dtype=torch.float64)
    labels = torch.tensor([1, 0], dtype=torch.long)
    assert torch.equal(
        selected_component_log_p_over_q(component, labels),
        torch.tensor([-2.0, 0.5], dtype=torch.float64),
    )


def test_mixture_validation_rejects_invalid_weights_and_labels() -> None:
    component = torch.zeros(3, 2, dtype=torch.float64)
    with pytest.raises(ValueError, match="sum to one"):
        log_mixture_q_over_p(component, torch.tensor([0.4, 0.4]))
    with pytest.raises(ValueError, match="strictly positive"):
        log_mixture_q_over_p(component, torch.tensor([1.0, 0.0]))
    with pytest.raises(ValueError, match="invalid expert"):
        selected_component_log_p_over_q(component, torch.tensor([0, 1, 2]))


def test_sampled_labels_follow_declared_reproducible_weights() -> None:
    generator_one = torch.Generator().manual_seed(91)
    generator_two = torch.Generator().manual_seed(91)
    weights = torch.tensor([0.2, 0.8], dtype=torch.float64)
    first = sample_mixture_labels(weights, 20_000, generator=generator_one)
    second = sample_mixture_labels(weights, 20_000, generator=generator_two)
    assert torch.equal(first, second)
    assert float((first == 0).double().mean()) == pytest.approx(0.2, abs=0.01)


def test_gaussian_oracle_exposes_constant_drift_mode_failure() -> None:
    probability = gaussian_two_tail_probability(1.0, 3.0)
    assert gaussian_single_drift_second_moment(
        0.0, horizon=1.0, threshold=3.0
    ) == pytest.approx(probability, rel=1e-13)
    assert gaussian_single_drift_second_moment(
        1.0, horizon=1.0, threshold=3.0
    ) > 20.0 * probability
    mixture = gaussian_symmetric_mixture_second_moment(
        3.15, horizon=1.0, threshold=3.0
    )
    assert probability / mixture > 80.0


def test_balance_and_component_estimators_are_both_unbiased_at_audit_drift() -> None:
    paths = 300_000
    drift = 1.0
    threshold = 2.5
    generator = torch.Generator().manual_seed(551)
    labels = torch.randint(0, 2, (paths,), generator=generator)
    signs = torch.where(labels == 0, 1.0, -1.0).double()
    proposal = torch.randn(paths, generator=generator, dtype=torch.float64)
    target = proposal + signs * drift
    components = torch.stack(
        (drift * target - 0.5 * drift**2, -drift * target - 0.5 * drift**2), dim=-1
    )
    log_mixture = log_mixture_q_over_p(
        components, torch.tensor([0.5, 0.5], dtype=torch.float64)
    )
    event = (torch.abs(target) >= threshold).double()
    balance = event * torch.exp(-log_mixture)
    component = event * torch.exp(selected_component_log_p_over_q(components, labels))
    reference = gaussian_two_tail_probability(1.0, threshold)
    for values in (balance, component):
        standard_error = float(values.std(unbiased=True) / math.sqrt(paths))
        assert abs(float(values.mean()) - reference) <= 3.0 * standard_error
    assert float(balance.var(unbiased=True)) < float(component.var(unbiased=True))
