from __future__ import annotations

import torch

from src.path_integral.residual_coupling_transport import ResidualHouseholderCoordinates
from src.path_integral.residual_smc import (
    AdaptiveResidualSMCConfig,
    run_adaptive_residual_smc,
)


def _direction(dimension: int) -> torch.Tensor:
    result = torch.arange(1, dimension + 1, dtype=torch.float64)
    return result / torch.linalg.vector_norm(result)


def test_adaptive_smc_schedule_ess_and_seed_determinism() -> None:
    direction = _direction(7)
    mapping = ResidualHouseholderCoordinates.build(direction)

    def log_potential(residual: torch.Tensor) -> torch.Tensor:
        coordinates = mapping.to_coordinates(residual)
        return -2.0 * (coordinates[:, 0] - 2.0).square()

    config = AdaptiveResidualSMCConfig(
        particles=512,
        target_ess_fraction=0.75,
        pcn_scale=0.3,
        pcn_sweeps_per_stage=2,
    )
    first = run_adaptive_residual_smc(
        direction=direction,
        log_potential_fn=log_potential,
        root_seed=1,
        config=config,
    )
    second = run_adaptive_residual_smc(
        direction=direction,
        log_potential_fn=log_potential,
        root_seed=1,
        config=config,
    )
    betas = [stage.beta_next for stage in first.stages]
    assert all(left < right for left, right in zip([0.0, *betas[:-1]], betas, strict=True))
    assert betas[-1] == 1.0
    assert all(stage.ess_target_met for stage in first.stages)
    assert all(0.0 <= stage.pcn_acceptance_rate <= 1.0 for stage in first.stages)
    assert not first.particles_are_final_inferential_units
    assert first.used_seeds == second.used_seeds
    assert torch.equal(first.residual_particles, second.residual_particles)


def test_pcn_smc_targets_a_known_gaussian_tilt() -> None:
    direction = _direction(6)
    mapping = ResidualHouseholderCoordinates.build(direction)
    coefficient = 0.5

    def log_potential(residual: torch.Tensor) -> torch.Tensor:
        coordinates = mapping.to_coordinates(residual)
        return -0.5 * coefficient * torch.sum(coordinates.square(), dim=1)

    result = run_adaptive_residual_smc(
        direction=direction,
        log_potential_fn=log_potential,
        root_seed=20,
        config=AdaptiveResidualSMCConfig(
            particles=4096,
            target_ess_fraction=0.8,
            pcn_scale=0.4,
            pcn_sweeps_per_stage=5,
        ),
    )
    coordinates = mapping.to_coordinates(result.residual_particles)
    expected_variance = 1.0 / (1.0 + coefficient)
    assert abs(float(torch.mean(coordinates))) < 0.04
    assert abs(float(torch.var(coordinates, unbiased=True)) - expected_variance) < 0.06


def test_smc_rejects_positive_conditional_log_potential() -> None:
    direction = _direction(4)

    def invalid(residual: torch.Tensor) -> torch.Tensor:
        return torch.ones(residual.shape[0], dtype=torch.float64)

    try:
        run_adaptive_residual_smc(
            direction=direction,
            log_potential_fn=invalid,
            root_seed=30,
            config=AdaptiveResidualSMCConfig(particles=32),
        )
    except ValueError as error:
        assert "must not exceed zero" in str(error)
    else:
        raise AssertionError("positive conditional log potential was accepted")
