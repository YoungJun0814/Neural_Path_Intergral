from __future__ import annotations

import torch

from src.path_integral.baseline_framework import BaselineCostLedger
from src.path_integral.low_rank_residual_flow import (
    LowRankResidualFlowTrainingConfig,
    fit_low_rank_residual_flow,
)
from src.path_integral.residual_coupling_transport import ResidualHouseholderCoordinates
from src.path_integral.tempered_residual_flow_mixture import (
    TemperedFlowComponent,
    freeze_tempered_residual_flow_mixture,
    sample_tempered_residual_flow_mixture,
    tempered_mixture_log_q_over_p,
)


def _proposal():
    direction = torch.arange(1, 9, dtype=torch.float64)
    direction /= torch.linalg.vector_norm(direction)
    mapping = ResidualHouseholderCoordinates.build(direction)
    base = torch.randn((256, 7), dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    flows = []
    for index, shift in enumerate((0.5, 1.5)):
        coordinates = base.clone()
        coordinates[:, 0] += shift
        flows.append(
            fit_low_rank_residual_flow(
                task_id="tempered-toy",
                residual=mapping.from_coordinates(coordinates),
                direction=direction,
                training_seed=10 + index,
                config=LowRankResidualFlowTrainingConfig(
                    layers=2, rank=2, epochs=5, defensive_weight=0.2
                ),
            )
        )
    return freeze_tempered_residual_flow_mixture(
        task_id="tempered-toy",
        components=(
            TemperedFlowComponent(0.5, 0.4, flows[0]),
            TemperedFlowComponent(1.0, 0.6, flows[1]),
        ),
        training_seed=99,
        training_cost=BaselineCostLedger(),
    )


def test_tempered_balance_mixture_has_exact_normalization_and_bound() -> None:
    proposal = _proposal()
    sample = sample_tempered_residual_flow_mixture(proposal, 50_000, root_seed=100)
    direct = torch.exp(-tempered_mixture_log_q_over_p(sample.residual, proposal))
    assert torch.max(torch.abs(direct - sample.likelihood)) < 1e-12
    se = float(torch.std(sample.likelihood, unbiased=True)) / sample.likelihood.numel() ** 0.5
    assert abs(float(torch.mean(sample.likelihood)) - 1.0) <= 5.0 * se
    assert float(torch.max(sample.likelihood)) <= proposal.likelihood_bound * (1 + 1e-10)
    assert sample.maximum_projection_error < 1e-12
    assert set(sample.component_labels.tolist()) == {0, 1}


def test_tempered_mixture_sampling_is_seed_deterministic() -> None:
    proposal = _proposal()
    first = sample_tempered_residual_flow_mixture(proposal, 100, root_seed=123)
    second = sample_tempered_residual_flow_mixture(proposal, 100, root_seed=123)
    assert torch.equal(first.residual, second.residual)
    assert torch.equal(first.component_labels, second.component_labels)
    assert torch.equal(first.inner_flow_labels, second.inner_flow_labels)
    assert torch.equal(first.likelihood, second.likelihood)
    assert len(first.used_seeds) == len(set(first.used_seeds))
