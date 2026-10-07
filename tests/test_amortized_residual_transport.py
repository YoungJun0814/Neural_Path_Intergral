from __future__ import annotations

import torch

from src.path_integral.amortized_residual_transport import (
    AmortizedResidualTrainingConfig,
    emit_residual_transport,
    fit_amortized_residual_generator,
)
from src.path_integral.baseline_framework import BaselineCostLedger
from src.path_integral.residual_transport import (
    ResidualGaussianMixtureSpec,
    evaluate_residual_likelihood,
    freeze_residual_transport,
    sample_residual_mixture,
)


def _teacher(index: int):
    direction = torch.zeros(6, dtype=torch.float64)
    price = torch.tensor([1.0, 1.0 + 0.1 * index, 1.2], dtype=torch.float64)
    direction[3:] = price / torch.linalg.vector_norm(price)
    shifted = torch.tensor(
        [[0.1 * index, -0.2, 0.3, 0.2, -0.1, 0.1]], dtype=torch.float64
    )
    shifted = shifted - (shifted @ direction).unsqueeze(1) * direction
    spec = ResidualGaussianMixtureSpec(
        direction=direction,
        means=torch.cat((torch.zeros((1, 6), dtype=torch.float64), shifted)),
        weights=torch.tensor([0.1, 0.9], dtype=torch.float64),
    )
    return freeze_residual_transport(
        task_id=f"teacher-{index}",
        spec=spec,
        training_seed=index,
        training_objective="toy-teacher",
        training_cost=BaselineCostLedger(),
    )


def test_amortized_generator_emits_exact_defensive_proposal() -> None:
    features = torch.tensor(
        [[0.05, -3.0], [0.10, -4.0], [0.15, -5.0], [0.20, -6.0]],
        dtype=torch.float64,
    )
    generator = fit_amortized_residual_generator(
        task_features=features,
        teacher_proposals=[_teacher(i) for i in range(4)],
        local_dimension=3,
        training_seed=10,
        config=AmortizedResidualTrainingConfig(hidden_features=8, epochs=100),
    )
    proposal = emit_residual_transport(
        generator,
        task_id="held-out",
        task_features=torch.tensor([0.12, -4.5], dtype=torch.float64),
    )
    spec = proposal.spec()
    assert torch.max(torch.abs(spec.direction[:3])) == 0.0
    assert torch.all(spec.direction[3:] > 0.0)
    assert proposal.exact_likelihood and not proposal.self_normalized
    sample = sample_residual_mixture(
        spec,
        30_000,
        gaussian_generator=torch.Generator().manual_seed(20),
        label_generator=torch.Generator().manual_seed(21),
    )
    likelihood = evaluate_residual_likelihood(sample.residual, spec).likelihood
    assert abs(float(torch.mean(likelihood)) - 1.0) < 0.04
