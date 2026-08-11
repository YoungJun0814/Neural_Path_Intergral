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
