from __future__ import annotations

import torch

from src.path_integral.residual_coupling_transport import (
    ResidualCouplingTrainingConfig,
    ResidualHouseholderCoordinates,
    fit_residual_coupling_transport,
    residual_coupling_log_q_over_p,
    sample_residual_coupling_transport,
)


def test_householder_residual_coordinates_round_trip() -> None:
    direction = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    direction /= torch.linalg.vector_norm(direction)
    mapping = ResidualHouseholderCoordinates.build(direction)
    coordinates = torch.randn((100, 3), dtype=torch.float64)
    residual = mapping.from_coordinates(coordinates)
    assert torch.max(torch.abs(residual @ direction)) < 1e-13
    assert torch.max(torch.abs(mapping.to_coordinates(residual) - coordinates)) < 1e-13


def test_defensive_residual_flow_has_exact_normalization_and_bound() -> None:
    direction = torch.tensor([1.0, -1.0, 2.0, 0.5], dtype=torch.float64)
    direction /= torch.linalg.vector_norm(direction)
    mapping = ResidualHouseholderCoordinates.build(direction)
    target_coordinates = torch.randn(
        (512, 3), dtype=torch.float64, generator=torch.Generator().manual_seed(1)
    )
    target_residual = mapping.from_coordinates(target_coordinates)
    log_target = -2.0 * (target_coordinates[:, 0] - 1.5).square()
    proposal = fit_residual_coupling_transport(
        task_id="toy-flow",
        residual=target_residual,
        direction=direction,
        log_unnormalized_target_over_sampling=log_target,
        training_seed=2,
        config=ResidualCouplingTrainingConfig(layers=2, epochs=15),
    )
    sample = sample_residual_coupling_transport(
        proposal, 30_000, gaussian_seed=3, label_seed=4
    )
    assert abs(float(torch.mean(sample.likelihood)) - 1.0) < 0.04
    assert float(torch.max(sample.likelihood)) <= 10.0 * (1.0 + 1e-10)
    direct = torch.exp(-residual_coupling_log_q_over_p(sample.residual, proposal))
    assert torch.max(torch.abs(direct - sample.likelihood)) < 1e-12
