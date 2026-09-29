"""Exact-density and held-out geometry checks for weighted SMC-bank mixtures."""

from __future__ import annotations

import math

import torch

from src.path_integral.baselines.weighted_conditional_ce import proposal_parameters
from src.path_integral.weighted_bank_mixture import (
    assign_saved_cluster_geometry,
    assign_weighted_bank_clusters,
    fit_weighted_bank_mixture,
    proposal_from_parameters,
)


def test_bimodal_weighted_bank_recovers_two_regions_and_exact_mixture() -> None:
    generator = torch.Generator().manual_seed(123)
    left = -3.0 + 0.4 * torch.randn((100, 2), dtype=torch.float64, generator=generator)
    right = 3.0 + 0.4 * torch.randn((100, 2), dtype=torch.float64, generator=generator)
    samples = torch.cat((left, right))
    weights = torch.ones(200, dtype=torch.float64)
    fit = fit_weighted_bank_mixture(
        samples, weights, clusters=2, covariance_rank=1,
        defensive_mass=0.1,
    )
    assert len(fit.proposal.components) == 3
    assert math.isclose(fit.proposal.defensive_mass, 0.1)
    labels = assign_weighted_bank_clusters(samples, fit)
    assert torch.sum(labels[:100] == labels[0]) >= 95
    assert torch.sum(labels[100:] == labels[100]) >= 95
    assert labels[0] != labels[100]
    saved = {
        "feature_mean": fit.feature_mean.tolist(),
        "feature_directions": fit.feature_directions.tolist(),
        "feature_scales": fit.feature_scales.tolist(),
        "centers": fit.centers.tolist(),
    }
    torch.testing.assert_close(assign_saved_cluster_geometry(samples, saved), labels)
    drawn = fit.proposal.sample(100, path_seed=44, label_seed=45)
    component_log_density = torch.stack([
        c.log_q_over_p(drawn.samples) for c in fit.proposal.components
    ], dim=1)
    direct = torch.logsumexp(component_log_density + torch.log(fit.proposal.weights), dim=1)
    torch.testing.assert_close(drawn.log_q_over_p, direct, rtol=1e-12, atol=1e-12)
    reconstructed = proposal_from_parameters(proposal_parameters(fit.proposal))
    torch.testing.assert_close(reconstructed.log_q_over_p(drawn.samples), direct,
                               rtol=1e-12, atol=1e-12)


def test_weighted_bank_ess_uses_normalized_weights() -> None:
    x = torch.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=torch.float64)
    w = torch.tensor([1.0, 1.0, 1.0, 3.0], dtype=torch.float64)
    fit = fit_weighted_bank_mixture(x, w, clusters=1, covariance_rank=0)
    assert math.isclose(fit.weighted_bank_ess, 36.0 / 12.0)
    assert math.isclose(float(fit.proposal.weights.sum()), 1.0)


def test_invalid_weighted_bank_rejected() -> None:
    x = torch.randn(3, 2, dtype=torch.float64)
    w = torch.zeros(3, dtype=torch.float64)
    try:
        fit_weighted_bank_mixture(x, w, clusters=1, covariance_rank=0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero-mass bank accepted")
