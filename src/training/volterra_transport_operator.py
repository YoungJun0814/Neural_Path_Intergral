"""Supervised teacher training for the structure-preserving V15 operator."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from src.models.volterra_transport_operator import VolterraTransportOperator


@dataclass(frozen=True)
class VolterraOperatorTeachers:
    features: torch.Tensor
    coefficients: torch.Tensor
    mode_weights: torch.Tensor
    precision_matrices: torch.Tensor
    mode_mask: torch.Tensor

    def validate_for(self, model: VolterraTransportOperator) -> None:
        count = self.features.shape[0]
        config = model.config
        expected_coefficients = (count, config.modes, config.rank)
        expected_matrices = (count, config.modes, config.rank, config.rank)
        if self.features.shape != (count, config.feature_dimension):
            raise ValueError("teacher features have the wrong shape")
        if self.coefficients.shape != expected_coefficients:
            raise ValueError("teacher coefficients have the wrong shape")
        if self.mode_weights.shape != (count, config.modes):
            raise ValueError("teacher weights have the wrong shape")
        if self.precision_matrices.shape != expected_matrices:
            raise ValueError("teacher precisions have the wrong shape")
        if self.mode_mask.shape != (count, config.modes) or self.mode_mask.dtype != torch.bool:
            raise ValueError("teacher mode mask has the wrong contract")
        tensors = (
            self.features,
            self.coefficients,
            self.mode_weights,
            self.precision_matrices,
        )
        if any(item.device.type != "cpu" or item.dtype != torch.float64 for item in tensors):
            raise ValueError("operator teachers must be CPU float64")
        if any(not torch.isfinite(item).all() for item in tensors):
            raise ValueError("operator teachers must be finite")
        if torch.any(self.mode_weights < 0.0):
            raise ValueError("teacher mode weights must be nonnegative")
        active_mass = torch.sum(self.mode_weights * self.mode_mask, dim=1)
        if torch.any(torch.abs(active_mass - 1.0) > 1e-10):
            raise ValueError("active teacher weights must sum to one")


@dataclass(frozen=True)
class VolterraOperatorTrainingConfig:
    epochs: int = 500
    learning_rate: float = 3e-3
    coefficient_weight: float = 1.0
    mixture_weight: float = 0.2
    precision_weight: float = 0.1

    def __post_init__(self) -> None:
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int) or self.epochs < 1:
            raise ValueError("operator epochs must be a positive integer")
        scales = (
            self.learning_rate,
            self.coefficient_weight,
            self.mixture_weight,
            self.precision_weight,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in scales):
            raise ValueError("operator training scales must be finite and positive")


@dataclass(frozen=True)
class VolterraOperatorTrainingResult:
    initial_loss: float
    final_loss: float
    loss_history: tuple[float, ...]


def _teacher_loss(
    model: VolterraTransportOperator,
    teachers: VolterraOperatorTeachers,
    config: VolterraOperatorTrainingConfig,
) -> torch.Tensor:
    prediction = model(teachers.features)
    mask = teachers.mode_mask.to(torch.float64)
    coefficient_error = torch.sum(
        (prediction.coefficients - teachers.coefficients).square(),
        dim=2,
    )
    precision_error = torch.mean(
        (prediction.precision_matrices - teachers.precision_matrices).square(),
        dim=(2, 3),
    )
    denominator = torch.clamp(torch.sum(mask), min=1.0)
    coefficient_loss = torch.sum(mask * coefficient_error) / denominator
    precision_loss = torch.sum(mask * precision_error) / denominator
    mixture_loss = torch.mean((prediction.mode_weights - teachers.mode_weights).square())
    return (
        config.coefficient_weight * coefficient_loss
        + config.mixture_weight * mixture_loss
        + config.precision_weight * precision_loss
    )


def train_volterra_transport_operator(
    model: VolterraTransportOperator,
    teachers: VolterraOperatorTeachers,
    *,
    seed: int,
    config: VolterraOperatorTrainingConfig | None = None,
) -> VolterraOperatorTrainingResult:
    config = config or VolterraOperatorTrainingConfig()
    teachers.validate_for(model)
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = []
    for _ in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = _teacher_loss(model, teachers, config)
        if not torch.isfinite(loss):
            raise FloatingPointError("operator teacher loss became nonfinite")
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return VolterraOperatorTrainingResult(
        initial_loss=history[0],
        final_loss=history[-1],
        loss_history=tuple(history),
    )
