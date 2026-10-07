"""Oracle checks for fixed-calendar weighted SMC and resampling genealogy."""

from __future__ import annotations

import math

import torch

from src.path_integral.weighted_tempered_smc import (
    WeightedSMCConfig,
    estimate_weighted_tempered_normalizer,
)


def _constant(x: torch.Tensor) -> torch.Tensor:
    return torch.full((x.shape[0],), math.log(0.25), dtype=torch.float64)


def test_constant_potential_exact_for_all_resampling_calendars() -> None:
    for scheme in ("multinomial", "stratified"):
        for every in (1, 3, 100):
            result = estimate_weighted_tempered_normalizer(
                _constant, dimension=3,
                config=WeightedSMCConfig(
                    particles=64, temperatures=(0.0, 0.1, 0.4, 0.7, 1.0),
                    mutation_steps=1, pcn_scale=0.5, replicates=2, seed=7,
                    resample_every=every, resampling_scheme=scheme,
                    retain_final_particles=True,
                ),
            )
            assert math.isclose(result.mean, 0.25, rel_tol=1e-14)
            assert result.final_particles is not None
            assert result.final_weights is not None
            for weights in result.final_weights.split(64):
                assert math.isclose(float(weights.sum()), 1.0, rel_tol=1e-14)
            assert result.potential_evaluations == 2 * 64 * 4


def test_no_resampling_no_mutation_is_exact_plain_monte_carlo() -> None:
    seed = 55
    generator = torch.Generator().manual_seed(seed)
    reference_samples = torch.randn((128, 2), dtype=torch.float64, generator=generator)
    def potential(x: torch.Tensor) -> torch.Tensor:
        return torch.special.log_ndtr(x[:, 0] - 0.5)
    reference = float(torch.mean(torch.exp(potential(reference_samples))))
    result = estimate_weighted_tempered_normalizer(
        potential, dimension=2,
        config=WeightedSMCConfig(
            particles=128, temperatures=(0.0, 0.05, 0.3, 1.0),
            mutation_steps=0, pcn_scale=0.5, replicates=1, seed=seed,
            resample_every=100, resampling_scheme="stratified",
        ),
    )
    assert math.isclose(result.mean, reference, rel_tol=1e-14)
    assert result.potential_evaluations == 128
    assert result.replicate_diagnostics[0]["resampling_stages"] == 0


def test_stratified_resampling_preserves_all_ancestors_at_uniform_weights() -> None:
    result = estimate_weighted_tempered_normalizer(
        _constant, dimension=1,
        config=WeightedSMCConfig(
            particles=40, temperatures=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            mutation_steps=0, pcn_scale=0.5, replicates=1, seed=77,
            resample_every=1, resampling_scheme="stratified",
        ),
    )
    assert result.replicate_diagnostics[0]["final_unique_initial_ancestors"] == 40


def test_gaussian_cdf_normalizer_matches_analytic_value() -> None:
    threshold = 1.0
    result = estimate_weighted_tempered_normalizer(
        lambda x: torch.special.log_ndtr(x[:, 0] - threshold), dimension=2,
        config=WeightedSMCConfig(
            particles=256,
            temperatures=tuple((index / 16) ** 2 for index in range(17)),
            mutation_steps=1, pcn_scale=0.55, replicates=40, seed=1234,
            resample_every=4, resampling_scheme="stratified",
        ),
    )
    analytic = float(torch.special.ndtr(torch.tensor(-threshold / math.sqrt(2))))
    assert abs(result.mean - analytic) < 4.0 * result.standard_error


def test_invalid_resampling_configuration_rejected() -> None:
    try:
        WeightedSMCConfig(16, (0.0, 1.0), 1, 0.5, 1, 7, resample_every=0)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid calendar was accepted")
