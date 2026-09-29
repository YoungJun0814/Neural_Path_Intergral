"""Task-conditioned generators that emit frozen exact residual proposals."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import torch

from src.path_integral.baseline_framework import BaselineCostLedger
from src.path_integral.provenance import process_peak_resident_memory_bytes
from src.path_integral.residual_transport import (
    FrozenResidualTransport,
    ResidualGaussianMixtureSpec,
    freeze_residual_transport,
)


@dataclass(frozen=True)
class AmortizedResidualTrainingConfig:
    hidden_features: int = 32
    epochs: int = 500
    learning_rate: float = 0.01
    direction_loss_weight: float = 1.0
    mean_loss_weight: float = 1.0
    mixture_loss_weight: float = 0.1
    maximum_mean_norm: float = 20.0
    positive_floor: float = 1e-6

    def __post_init__(self) -> None:
        if self.hidden_features < 2 or self.epochs < 1:
            raise ValueError("amortized architecture counts are invalid")
        positive = (
            self.learning_rate,
            self.direction_loss_weight,
            self.mean_loss_weight,
            self.mixture_loss_weight,
            self.maximum_mean_norm,
            self.positive_floor,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("amortized scales must be finite and positive")


@dataclass(frozen=True)
class FrozenAmortizedResidualGenerator:
    schema: str
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    hidden_weight: tuple[tuple[float, ...], ...]
    hidden_bias: tuple[float, ...]
    output_weight: tuple[tuple[float, ...], ...]
    output_bias: tuple[float, ...]
    dimension: int
    components: int
    defensive_weight: float
    local_dimension: int
    maximum_mean_norm: float
    positive_floor: float
    training_seed: int
    training_cost: BaselineCostLedger
    frozen: bool
    emits_exact_likelihood: bool
    sha256: str


def _canonical_shifted(proposal: FrozenResidualTransport) -> tuple[torch.Tensor, torch.Tensor]:
    means = torch.tensor(proposal.component_means[1:], dtype=torch.float64)
    weights = torch.tensor(proposal.component_weights[1:], dtype=torch.float64)
    # Mixture labels are not identifiable.  A deterministic norm/lexicographic
    # order prevents label switching from contaminating the supervised target.
    keys = [
        (float(torch.linalg.vector_norm(row)), tuple(float(x) for x in row), index)
        for index, row in enumerate(means)
    ]
    order = [item[2] for item in sorted(keys)]
    return means[order], weights[order]


def _generator_payload(**kwargs: object) -> dict[str, object]:
    return {
        "schema": "npi.g11.v12-amortized-residual-generator.v1",
        **kwargs,
        "frozen": True,
        "emits_exact_likelihood": True,
    }


def _freeze_generator(
    *,
    feature_mean: torch.Tensor,
    feature_scale: torch.Tensor,
    hidden_weight: torch.Tensor,
    hidden_bias: torch.Tensor,
    output_weight: torch.Tensor,
    output_bias: torch.Tensor,
    dimension: int,
    components: int,
    defensive_weight: float,
    local_dimension: int,
    config: AmortizedResidualTrainingConfig,
    training_seed: int,
    training_cost: BaselineCostLedger,
) -> FrozenAmortizedResidualGenerator:
    payload = _generator_payload(
        feature_mean=tuple(float(x) for x in feature_mean),
        feature_scale=tuple(float(x) for x in feature_scale),
        hidden_weight=tuple(tuple(float(x) for x in row) for row in hidden_weight),
        hidden_bias=tuple(float(x) for x in hidden_bias),
        output_weight=tuple(tuple(float(x) for x in row) for row in output_weight),
        output_bias=tuple(float(x) for x in output_bias),
        dimension=dimension,
        components=components,
        defensive_weight=defensive_weight,
        local_dimension=local_dimension,
        maximum_mean_norm=config.maximum_mean_norm,
        positive_floor=config.positive_floor,
        training_seed=training_seed,
        training_cost=asdict(training_cost),
    )
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return FrozenAmortizedResidualGenerator(
        schema="npi.g11.v12-amortized-residual-generator.v1",
        feature_mean=payload["feature_mean"],  # type: ignore[arg-type]
        feature_scale=payload["feature_scale"],  # type: ignore[arg-type]
        hidden_weight=payload["hidden_weight"],  # type: ignore[arg-type]
        hidden_bias=payload["hidden_bias"],  # type: ignore[arg-type]
        output_weight=payload["output_weight"],  # type: ignore[arg-type]
        output_bias=payload["output_bias"],  # type: ignore[arg-type]
        dimension=dimension,
        components=components,
        defensive_weight=defensive_weight,
        local_dimension=local_dimension,
        maximum_mean_norm=config.maximum_mean_norm,
        positive_floor=config.positive_floor,
        training_seed=training_seed,
        training_cost=training_cost,
        frozen=True,
        emits_exact_likelihood=True,
        sha256=digest,
    )


def _decode(
    raw: torch.Tensor,
    *,
    dimension: int,
    components: int,
    defensive_weight: float,
    local_dimension: int,
    maximum_mean_norm: float,
    positive_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    price_dimension = dimension - local_dimension
    shifted = components - 1
    direction_raw = raw[:, :price_dimension]
    cursor = price_dimension
    mean_raw = raw[:, cursor : cursor + shifted * dimension].reshape(
        -1, shifted, dimension
    )
    cursor += shifted * dimension
    weight_logits = raw[:, cursor : cursor + shifted]
    price = torch.nn.functional.softplus(direction_raw) + positive_floor
    price = price / torch.linalg.vector_norm(price, dim=1).unsqueeze(1)
    direction = torch.cat(
        (torch.zeros((raw.shape[0], local_dimension), dtype=torch.float64), price), dim=1
    )
    projected = mean_raw - torch.sum(
        mean_raw * direction.unsqueeze(1), dim=2, keepdim=True
    ) * direction.unsqueeze(1)
    norms = torch.linalg.vector_norm(projected, dim=2, keepdim=True)
    factors = torch.clamp(maximum_mean_norm / torch.clamp(norms, min=1e-300), max=1.0)
    means = projected * factors
    weights = (1.0 - defensive_weight) * torch.softmax(weight_logits, dim=1)
    return direction, means, weights


def fit_amortized_residual_generator(
    *,
    task_features: torch.Tensor,
    teacher_proposals: Sequence[FrozenResidualTransport],
    local_dimension: int,
    training_seed: int,
    config: AmortizedResidualTrainingConfig | None = None,
) -> FrozenAmortizedResidualGenerator:
    config = config or AmortizedResidualTrainingConfig()
    if task_features.ndim != 2 or task_features.shape[0] != len(teacher_proposals):
        raise ValueError("one feature row is required per teacher")
    if len(teacher_proposals) < 2 or task_features.dtype != torch.float64:
        raise ValueError("at least two float64 teachers are required")
    first = teacher_proposals[0]
    dimension, components = first.dimension, first.components
    defensive = float(first.component_weights[0])
    if not 0 <= local_dimension < dimension:
        raise ValueError("invalid local dimension")
    for teacher in teacher_proposals:
        if teacher.dimension != dimension or teacher.components != components:
            raise ValueError("teacher proposal shapes differ")
        if not math.isclose(float(teacher.component_weights[0]), defensive, abs_tol=1e-12):
            raise ValueError("teacher defensive weights differ")
        direction = torch.tensor(teacher.direction, dtype=torch.float64)
        if float(torch.amax(torch.abs(direction[:local_dimension]))) > 1e-10 or bool(
            (direction[local_dimension:] <= 0.0).any()
        ):
            raise ValueError("teacher direction violates positive price support")
    feature_mean = torch.mean(task_features, dim=0)
    feature_scale = torch.std(task_features, dim=0, unbiased=False)
    feature_scale = torch.where(feature_scale > 1e-12, feature_scale, torch.ones_like(feature_scale))
    normalized = (task_features - feature_mean) / feature_scale
    target_directions = torch.stack(
        [torch.tensor(teacher.direction, dtype=torch.float64) for teacher in teacher_proposals]
    )
    canonical = [_canonical_shifted(teacher) for teacher in teacher_proposals]
    target_means = torch.stack([item[0] for item in canonical])
    target_weights = torch.stack([item[1] for item in canonical])
    price_dimension = dimension - local_dimension
    output_dimension = price_dimension + (components - 1) * dimension + components - 1
    generator = torch.Generator().manual_seed(training_seed)
    hidden_weight = torch.nn.Parameter(
        0.05 * torch.randn((task_features.shape[1], config.hidden_features), dtype=torch.float64, generator=generator)
    )
    hidden_bias = torch.nn.Parameter(torch.zeros(config.hidden_features, dtype=torch.float64))
    output_weight = torch.nn.Parameter(
        0.05 * torch.randn((config.hidden_features, output_dimension), dtype=torch.float64, generator=generator)
    )
    output_bias = torch.nn.Parameter(torch.zeros(output_dimension, dtype=torch.float64))
    optimizer = torch.optim.Adam(
        (hidden_weight, hidden_bias, output_weight, output_bias), lr=config.learning_rate
    )
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    for _ in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        hidden = torch.tanh(normalized @ hidden_weight + hidden_bias)
        raw = hidden @ output_weight + output_bias
        directions, means, weights = _decode(
            raw,
            dimension=dimension,
            components=components,
            defensive_weight=defensive,
            local_dimension=local_dimension,
            maximum_mean_norm=config.maximum_mean_norm,
            positive_floor=config.positive_floor,
        )
        loss = config.direction_loss_weight * torch.mean((directions - target_directions) ** 2)
        loss += config.mean_loss_weight * torch.mean((means - target_means) ** 2)
        loss += config.mixture_loss_weight * torch.mean((weights - target_weights) ** 2)
        if not torch.isfinite(loss):
            raise FloatingPointError("amortized training loss became nonfinite")
        loss.backward()
        optimizer.step()
    work = config.epochs * len(teacher_proposals) * (
        task_features.shape[1] * config.hidden_features
        + config.hidden_features * output_dimension
    )
    cost = BaselineCostLedger(
        training_samples=len(teacher_proposals),
        optimizer_steps=config.epochs,
        hyperparameter_trials=1,
        algorithmic_work_units=float(work),
        wall_seconds=time.perf_counter() - started_wall,
        cpu_seconds=time.process_time() - started_cpu,
        peak_memory_bytes=process_peak_resident_memory_bytes(),
        measurement_mode="standardized_hardware_wall",
    )
    return _freeze_generator(
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        hidden_weight=hidden_weight.detach(),
        hidden_bias=hidden_bias.detach(),
        output_weight=output_weight.detach(),
        output_bias=output_bias.detach(),
        dimension=dimension,
        components=components,
        defensive_weight=defensive,
        local_dimension=local_dimension,
        config=config,
        training_seed=training_seed,
        training_cost=cost,
    )


def emit_residual_transport(
    generator: FrozenAmortizedResidualGenerator,
    *,
    task_id: str,
    task_features: torch.Tensor,
) -> FrozenResidualTransport:
    if task_features.shape != (len(generator.feature_mean),):
        raise ValueError("emission features have the wrong shape")
    feature = (
        task_features - torch.tensor(generator.feature_mean, dtype=torch.float64)
    ) / torch.tensor(generator.feature_scale, dtype=torch.float64)
    hidden = torch.tanh(
        feature @ torch.tensor(generator.hidden_weight, dtype=torch.float64)
        + torch.tensor(generator.hidden_bias, dtype=torch.float64)
    )
    raw = (
        hidden @ torch.tensor(generator.output_weight, dtype=torch.float64)
        + torch.tensor(generator.output_bias, dtype=torch.float64)
    ).unsqueeze(0)
    direction, shifted_means, shifted_weights = _decode(
        raw,
        dimension=generator.dimension,
        components=generator.components,
        defensive_weight=generator.defensive_weight,
        local_dimension=generator.local_dimension,
        maximum_mean_norm=generator.maximum_mean_norm,
        positive_floor=generator.positive_floor,
    )
    means = torch.cat(
        (torch.zeros((1, generator.dimension), dtype=torch.float64), shifted_means[0]),
        dim=0,
    )
    weights = torch.cat(
        (torch.tensor([generator.defensive_weight], dtype=torch.float64), shifted_weights[0])
    )
    spec = ResidualGaussianMixtureSpec(
        direction=direction[0], means=means, weights=weights
    )
    return freeze_residual_transport(
        task_id=task_id,
        spec=spec,
        training_seed=generator.training_seed,
        training_objective="amortized_task_conditioned_exact_residual_emission",
        training_cost=generator.training_cost,
    )
