import math

import torch

from src.path_integral.tempered_conditional_smc import (
    TemperedSMCConfig,
    estimate_tempered_normalizer,
)


def _config(*, replicates: int = 12) -> TemperedSMCConfig:
    return TemperedSMCConfig(
        particles=512,
        temperatures=tuple((index / 12) ** 2 for index in range(13)),
        mutation_steps=2,
        pcn_scale=0.45,
        replicates=replicates,
        seed=1617,
    )


def test_constant_potential_normalizer_is_exact_for_every_replicate() -> None:
    target = 2.5e-7

    def log_potential(points: torch.Tensor) -> torch.Tensor:
        return torch.full((points.shape[0],), math.log(target), dtype=torch.float64)

    result = estimate_tempered_normalizer(log_potential, dimension=4, config=_config())
    torch.testing.assert_close(
        result.replicate_estimates,
        torch.full_like(result.replicate_estimates, target),
        rtol=2e-14,
        atol=0.0,
    )


def test_gaussian_potential_matches_analytic_normalizer() -> None:
    coefficient = 0.7
    dimension = 3

    def log_potential(points: torch.Tensor) -> torch.Tensor:
        return -0.5 * coefficient * torch.sum(points.square(), dim=1)

    result = estimate_tempered_normalizer(
        log_potential,
        dimension=dimension,
        config=_config(replicates=24),
    )
    oracle = (1.0 + coefficient) ** (-0.5 * dimension)
    assert abs(result.mean - oracle) <= 4.0 * result.standard_error + 0.008
    assert 0.0 < result.mutation_acceptance_rate < 1.0
    assert 0.0 < result.minimum_incremental_ess_fraction <= 1.0
    assert result.potential_evaluations > 0


def test_retained_final_particles_are_from_the_final_tempered_population() -> None:
    config = TemperedSMCConfig(
        particles=256,
        temperatures=tuple((index / 8) ** 2 for index in range(9)),
        mutation_steps=2,
        pcn_scale=0.4,
        replicates=3,
        seed=2718,
        retain_final_particles=True,
    )

    def log_potential(points: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.sum(points.square(), dim=1)

    result = estimate_tempered_normalizer(log_potential, dimension=2, config=config)
    assert result.final_particles is not None
    assert result.final_particles.shape == (3 * 256, 2)
    assert torch.isfinite(result.final_particles).all()
    # Target density is proportional to exp(-||x||^2/2) times N(0,I), hence
    # each coordinate has variance 1/2.  This loose oracle detects accidentally
    # returning the beta=0 population without making the SMC test brittle.
    variance = torch.var(result.final_particles, dim=0, unbiased=True)
    assert torch.max(torch.abs(variance - 0.5)) < 0.18
