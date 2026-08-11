from __future__ import annotations

import torch

from src.path_integral.low_rank_residual_flow import (
    LowRankResidualFlowTrainingConfig,
    fit_low_rank_residual_flow,
    low_rank_flow_forward,
    low_rank_flow_inverse,
    low_rank_flow_log_q_over_p,
    sample_low_rank_residual_flow,
)
from src.path_integral.residual_coupling_transport import ResidualHouseholderCoordinates


def _training_problem(dimension: int, samples: int = 512):
    direction = torch.arange(1, dimension + 1, dtype=torch.float64)
    direction /= torch.linalg.vector_norm(direction)
    mapping = ResidualHouseholderCoordinates.build(direction)
    coordinates = torch.randn(
        (samples, dimension - 1),
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(dimension),
    )
    coordinates[:, 0] += 1.5
    return direction, mapping.from_coordinates(coordinates)


def test_low_rank_flow_inverse_jacobian_likelihood_and_defensive_bound() -> None:
    direction, residual = _training_problem(10)
    proposal = fit_low_rank_residual_flow(
        task_id="low-rank-toy",
        residual=residual,
        direction=direction,
        training_seed=1,
        config=LowRankResidualFlowTrainingConfig(
            layers=3, rank=3, epochs=20, partition_style="ordered_blocks"
        ),
    )
    base = torch.randn((200, 9), dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    output, forward_log_det = low_rank_flow_forward(base, proposal)
    rebuilt, inverse_forward_log_det = low_rank_flow_inverse(output, proposal)
    assert torch.max(torch.abs(rebuilt - base)) < 2e-12
    assert torch.max(torch.abs(forward_log_det - inverse_forward_log_det)) < 2e-12
    sample = sample_low_rank_residual_flow(proposal, 40_000, gaussian_seed=3, label_seed=4)
    direct = torch.exp(-low_rank_flow_log_q_over_p(sample.residual, proposal))
    assert torch.max(torch.abs(direct - sample.likelihood)) < 1e-12
    standard_error = float(torch.std(sample.likelihood, unbiased=True)) / (
        sample.likelihood.numel() ** 0.5
    )
    assert abs(float(torch.mean(sample.likelihood)) - 1.0) <= 4.0 * standard_error
    assert float(torch.max(sample.likelihood)) <= 10.0 * (1.0 + 1e-10)
    assert sample.maximum_projection_error < 2e-13


def test_reported_conditioner_work_is_linear_in_dimension_at_fixed_rank() -> None:
    costs = []
    for dimension in (16, 32):
        direction, residual = _training_problem(dimension, samples=64)
        proposal = fit_low_rank_residual_flow(
            task_id=f"scaling-{dimension}",
            residual=residual,
            direction=direction,
            training_seed=dimension,
            config=LowRankResidualFlowTrainingConfig(
                layers=2, rank=2, epochs=2, partition_style="alternating"
            ),
        )
        costs.append(proposal.training_cost.algorithmic_work_units)
    ratio = costs[1] / costs[0]
    assert 1.8 < ratio < 2.3


def test_low_rank_training_is_seed_deterministic() -> None:
    direction, residual = _training_problem(8, samples=128)
    config = LowRankResidualFlowTrainingConfig(layers=2, rank=2, epochs=5)
    first = fit_low_rank_residual_flow(
        task_id="deterministic",
        residual=residual,
        direction=direction,
        training_seed=99,
        config=config,
    )
    second = fit_low_rank_residual_flow(
        task_id="deterministic",
        residual=residual,
        direction=direction,
        training_seed=99,
        config=config,
    )
    assert first.sha256 != second.sha256  # measured runtime belongs to the full hash
    assert first.layers == second.layers
