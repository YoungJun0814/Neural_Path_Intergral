"""Exact defensive low-rank coupling flows on Gaussian residual hyperplanes."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Literal

import torch

from src.path_integral.baseline_framework import BaselineCostLedger
from src.path_integral.provenance import process_peak_resident_memory_bytes
from src.path_integral.residual_coupling_transport import ResidualHouseholderCoordinates

PartitionStyle = Literal["alternating", "ordered_blocks"]


@dataclass(frozen=True)
class FrozenLowRankCouplingLayer:
    active_indices: tuple[int, ...]
    transformed_indices: tuple[int, ...]
    scale_left: tuple[tuple[float, ...], ...]
    scale_right: tuple[tuple[float, ...], ...]
    scale_bias: tuple[float, ...]
    shift_left: tuple[tuple[float, ...], ...]
    shift_right: tuple[tuple[float, ...], ...]
    shift_bias: tuple[float, ...]

    @property
    def rank(self) -> int:
        return len(self.scale_right)


@dataclass(frozen=True)
class FrozenLowRankResidualFlow:
    schema: str
    task_id: str
    direction: tuple[float, ...]
    defensive_weight: float
    maximum_log_scale: float
    partition_style: PartitionStyle
    layers: tuple[FrozenLowRankCouplingLayer, ...]
    training_seed: int
    training_cost: BaselineCostLedger
    exact_likelihood: bool
    self_normalized: bool
    frozen: bool
    sha256: str

    @property
    def dimension(self) -> int:
        return len(self.direction)

    @property
    def residual_dimension(self) -> int:
        return self.dimension - 1

    @property
    def rank(self) -> int:
        return self.layers[0].rank


@dataclass(frozen=True)
class LowRankResidualFlowTrainingConfig:
    layers: int = 4
    rank: int = 8
    epochs: int = 50
    learning_rate: float = 0.01
    defensive_weight: float = 0.1
    maximum_log_scale: float = 1.0
    l2_penalty: float = 1e-6
    partition_style: PartitionStyle = "alternating"

    def __post_init__(self) -> None:
        integers = (self.layers, self.rank, self.epochs)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in integers
        ):
            raise ValueError("low-rank flow counts must be positive integers")
        positive = (self.learning_rate, self.maximum_log_scale)
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("low-rank flow scales must be finite and positive")
        if not 0.0 < self.defensive_weight < 1.0:
            raise ValueError("defensive weight must lie in (0, 1)")
        if not math.isfinite(self.l2_penalty) or self.l2_penalty < 0.0:
            raise ValueError("L2 penalty must be finite and nonnegative")
        if self.partition_style not in {"alternating", "ordered_blocks"}:
            raise ValueError("unsupported low-rank partition style")


def _partition(
    dimension: int, layer: int, style: PartitionStyle
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if dimension < 2:
        raise ValueError("coupling coordinates require dimension at least two")
    if style == "alternating":
        active = tuple(index for index in range(dimension) if index % 2 == layer % 2)
        transformed = tuple(index for index in range(dimension) if index % 2 != layer % 2)
    else:
        split = (dimension + 1) // 2
        prefix = tuple(range(split))
        suffix = tuple(range(split, dimension))
        active, transformed = (prefix, suffix) if layer % 2 == 0 else (suffix, prefix)
    if not active or not transformed or set(active) & set(transformed):
        raise AssertionError("invalid exact coupling partition")
    if set(active) | set(transformed) != set(range(dimension)):
        raise AssertionError("coupling partition does not cover every coordinate")
    return active, transformed


def _matrix(value: tuple[tuple[float, ...], ...]) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float64)


def _vector(value: tuple[float, ...]) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float64)


def _condition(
    source: torch.Tensor,
    *,
    left: torch.Tensor,
    right: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return (source @ left) @ right + bias


def low_rank_flow_forward(
    base: torch.Tensor, proposal: FrozenLowRankResidualFlow
) -> tuple[torch.Tensor, torch.Tensor]:
    if base.ndim != 2 or base.shape[1] != proposal.residual_dimension:
        raise ValueError("base coordinates have the wrong shape")
    value = base.clone()
    log_det = torch.zeros(base.shape[0], dtype=torch.float64)
    for layer in proposal.layers:
        active = list(layer.active_indices)
        transformed = list(layer.transformed_indices)
        source = value[:, active]
        log_scale = proposal.maximum_log_scale * torch.tanh(
            _condition(
                source,
                left=_matrix(layer.scale_left),
                right=_matrix(layer.scale_right),
                bias=_vector(layer.scale_bias),
            )
        )
        shift = _condition(
            source,
            left=_matrix(layer.shift_left),
            right=_matrix(layer.shift_right),
            bias=_vector(layer.shift_bias),
        )
        updated = value.clone()
        updated[:, transformed] = value[:, transformed] * torch.exp(log_scale) + shift
        value = updated
        log_det += torch.sum(log_scale, dim=1)
    return value, log_det


def low_rank_flow_inverse(
    output: torch.Tensor, proposal: FrozenLowRankResidualFlow
) -> tuple[torch.Tensor, torch.Tensor]:
    if output.ndim != 2 or output.shape[1] != proposal.residual_dimension:
        raise ValueError("flow output coordinates have the wrong shape")
    value = output.clone()
    forward_log_det = torch.zeros(output.shape[0], dtype=torch.float64)
    for layer in reversed(proposal.layers):
        active = list(layer.active_indices)
        transformed = list(layer.transformed_indices)
        source = value[:, active]
        log_scale = proposal.maximum_log_scale * torch.tanh(
            _condition(
                source,
                left=_matrix(layer.scale_left),
                right=_matrix(layer.scale_right),
                bias=_vector(layer.scale_bias),
            )
        )
        shift = _condition(
            source,
            left=_matrix(layer.shift_left),
            right=_matrix(layer.shift_right),
            bias=_vector(layer.shift_bias),
        )
        updated = value.clone()
        updated[:, transformed] = (value[:, transformed] - shift) * torch.exp(-log_scale)
        value = updated
        forward_log_det += torch.sum(log_scale, dim=1)
    return value, forward_log_det


def low_rank_flow_log_q_over_p(
    residual: torch.Tensor, proposal: FrozenLowRankResidualFlow
) -> torch.Tensor:
    coordinate_map = ResidualHouseholderCoordinates.build(
        torch.tensor(proposal.direction, dtype=torch.float64)
    )
    output = coordinate_map.to_coordinates(residual)
    base, forward_log_det = low_rank_flow_inverse(output, proposal)
    log_flow_over_p = (
        -0.5 * (torch.sum(base.square(), dim=1) - torch.sum(output.square(), dim=1))
        - forward_log_det
    )
    delta = proposal.defensive_weight
    return torch.logaddexp(
        torch.full_like(log_flow_over_p, math.log(delta)),
        math.log1p(-delta) + log_flow_over_p,
    )


@dataclass(frozen=True)
class LowRankResidualFlowSample:
    residual: torch.Tensor
    labels: torch.Tensor
    likelihood: torch.Tensor
    maximum_projection_error: float
    maximum_likelihood_bound_violation: float


def sample_low_rank_residual_flow(
    proposal: FrozenLowRankResidualFlow,
    sample_count: int,
    *,
    gaussian_seed: int,
    label_seed: int,
) -> LowRankResidualFlowSample:
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
        raise ValueError("sample count must be a positive integer")
    if not proposal.exact_likelihood or proposal.self_normalized or not proposal.frozen:
        raise ValueError("sampling requires a frozen exact ordinary low-rank flow")
    seeds = (gaussian_seed, label_seed)
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("low-rank flow seeds must be nonnegative integers")
    if gaussian_seed == label_seed:
        raise ValueError("Gaussian and label seeds must be disjoint")
    coordinate_map = ResidualHouseholderCoordinates.build(
        torch.tensor(proposal.direction, dtype=torch.float64)
    )
    base = torch.randn(
        (sample_count, proposal.residual_dimension),
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(gaussian_seed),
    )
    labels = (
        torch.rand(sample_count, generator=torch.Generator().manual_seed(label_seed))
        >= proposal.defensive_weight
    )
    flowed, _ = low_rank_flow_forward(base, proposal)
    coordinates = torch.where(labels.unsqueeze(1), flowed, base)
    residual = coordinate_map.from_coordinates(coordinates)
    log_q_over_p = low_rank_flow_log_q_over_p(residual, proposal)
    likelihood = torch.exp(-log_q_over_p)
    bound = 1.0 / proposal.defensive_weight
    violation = max(0.0, float(torch.amax(likelihood)) - bound)
    projection = float(
        torch.amax(torch.abs(residual @ torch.tensor(proposal.direction, dtype=torch.float64)))
    )
    if not torch.isfinite(likelihood).all() or violation > 1e-9:
        raise FloatingPointError("low-rank defensive likelihood is invalid")
    return LowRankResidualFlowSample(
        residual=residual,
        labels=labels,
        likelihood=likelihood,
        maximum_projection_error=projection,
        maximum_likelihood_bound_violation=violation,
    )


def _payload(
    *,
    task_id: str,
    direction: tuple[float, ...],
    defensive_weight: float,
    maximum_log_scale: float,
    partition_style: PartitionStyle,
    layers: tuple[FrozenLowRankCouplingLayer, ...],
    training_seed: int,
    training_cost: BaselineCostLedger,
) -> dict[str, object]:
    return {
        "schema": "npi.g11.v13-frozen-low-rank-residual-flow.v1",
        "task_id": task_id,
        "direction": direction,
        "defensive_weight": defensive_weight,
        "maximum_log_scale": maximum_log_scale,
        "partition_style": partition_style,
        "layers": [asdict(layer) for layer in layers],
        "training_seed": training_seed,
        "training_cost": asdict(training_cost),
        "exact_likelihood": True,
        "self_normalized": False,
        "frozen": True,
    }


def freeze_low_rank_residual_flow(
    *,
    task_id: str,
    direction: torch.Tensor,
    defensive_weight: float,
    maximum_log_scale: float,
    partition_style: PartitionStyle,
    layers: tuple[FrozenLowRankCouplingLayer, ...],
    training_seed: int,
    training_cost: BaselineCostLedger,
) -> FrozenLowRankResidualFlow:
    mapping = ResidualHouseholderCoordinates.build(direction)
    if isinstance(training_seed, bool) or not isinstance(training_seed, int) or training_seed < 0:
        raise ValueError("low-rank training seed must be a nonnegative integer")
    if not task_id.strip() or not 0.0 < defensive_weight < 1.0 or not layers:
        raise ValueError("invalid frozen low-rank flow identity")
    if not math.isfinite(maximum_log_scale) or maximum_log_scale <= 0.0:
        raise ValueError("maximum log scale must be finite and positive")
    ranks: set[int] = set()
    for layer in layers:
        active, transformed = layer.active_indices, layer.transformed_indices
        rank = layer.rank
        ranks.add(rank)
        expected = (
            (len(layer.scale_left), len(active)),
            (len(layer.scale_right), rank),
            (len(layer.scale_bias), len(transformed)),
            (len(layer.shift_left), len(active)),
            (len(layer.shift_right), rank),
            (len(layer.shift_bias), len(transformed)),
        )
        if any(actual != target for actual, target in expected):
            raise ValueError("low-rank layer tensor shapes are inconsistent")
        if set(active) | set(transformed) != set(range(mapping.coordinate_dimension)):
            raise ValueError("low-rank layer does not cover residual coordinates")
    if len(ranks) != 1 or next(iter(ranks)) < 1:
        raise ValueError("all layers must share one positive rank")
    payload = _payload(
        task_id=task_id,
        direction=tuple(float(x) for x in direction),
        defensive_weight=defensive_weight,
        maximum_log_scale=maximum_log_scale,
        partition_style=partition_style,
        layers=layers,
        training_seed=training_seed,
        training_cost=training_cost,
    )
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return FrozenLowRankResidualFlow(
        schema="npi.g11.v13-frozen-low-rank-residual-flow.v1",
        task_id=task_id,
        direction=tuple(float(x) for x in direction),
        defensive_weight=defensive_weight,
        maximum_log_scale=maximum_log_scale,
        partition_style=partition_style,
        layers=layers,
        training_seed=training_seed,
        training_cost=training_cost,
        exact_likelihood=True,
        self_normalized=False,
        frozen=True,
        sha256=digest,
    )


def fit_low_rank_residual_flow(
    *,
    task_id: str,
    residual: torch.Tensor,
    direction: torch.Tensor,
    training_seed: int,
    log_training_weights: torch.Tensor | None = None,
    config: LowRankResidualFlowTrainingConfig | None = None,
) -> FrozenLowRankResidualFlow:
    """Fit an exact flow; normalized weights affect training only."""

    config = config or LowRankResidualFlowTrainingConfig()
    coordinate_map = ResidualHouseholderCoordinates.build(direction)
    coordinates = coordinate_map.to_coordinates(residual)
    sample_count, dimension = coordinates.shape
    if sample_count < 2:
        raise ValueError("low-rank flow training requires at least two residuals")
    if log_training_weights is None:
        weights = torch.full((sample_count,), 1.0 / sample_count, dtype=torch.float64)
    else:
        if log_training_weights.shape != (sample_count,) or bool(
            torch.isnan(log_training_weights).any()
        ):
            raise ValueError("low-rank flow training weights are invalid")
        weights = torch.softmax(log_training_weights, dim=0).detach()
    raw_layers: list[
        tuple[
            tuple[int, ...],
            tuple[int, ...],
            torch.nn.Parameter,
            torch.nn.Parameter,
            torch.nn.Parameter,
            torch.nn.Parameter,
            torch.nn.Parameter,
            torch.nn.Parameter,
        ]
    ] = []
    parameters: list[torch.nn.Parameter] = []
    generator = torch.Generator().manual_seed(training_seed)
    for layer_index in range(config.layers):
        active, transformed = _partition(dimension, layer_index, config.partition_style)
        rank = min(config.rank, len(active), len(transformed))
        scale_left = torch.nn.Parameter(
            0.02 * torch.randn((len(active), rank), dtype=torch.float64, generator=generator)
        )
        scale_right = torch.nn.Parameter(
            0.02 * torch.randn((rank, len(transformed)), dtype=torch.float64, generator=generator)
        )
        scale_bias = torch.nn.Parameter(torch.zeros(len(transformed), dtype=torch.float64))
        shift_left = torch.nn.Parameter(
            0.02 * torch.randn((len(active), rank), dtype=torch.float64, generator=generator)
        )
        shift_right = torch.nn.Parameter(
            0.02 * torch.randn((rank, len(transformed)), dtype=torch.float64, generator=generator)
        )
        shift_bias = torch.nn.Parameter(torch.zeros(len(transformed), dtype=torch.float64))
        raw_layers.append(
            (
                active,
                transformed,
                scale_left,
                scale_right,
                scale_bias,
                shift_left,
                shift_right,
                shift_bias,
            )
        )
        parameters.extend(
            (
                scale_left,
                scale_right,
                scale_bias,
                shift_left,
                shift_right,
                shift_bias,
            )
        )
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    for _ in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        value = coordinates.clone()
        inverse_log_det = torch.zeros(sample_count, dtype=torch.float64)
        for active, transformed, sl, sr, sb, tl, tr, tb in reversed(raw_layers):
            source = value[:, list(active)]
            log_scale = config.maximum_log_scale * torch.tanh((source @ sl) @ sr + sb)
            shift = (source @ tl) @ tr + tb
            updated = value.clone()
            updated[:, list(transformed)] = (value[:, list(transformed)] - shift) * torch.exp(
                -log_scale
            )
            value = updated
            inverse_log_det -= torch.sum(log_scale, dim=1)
        log_flow_over_p = (
            -0.5 * (torch.sum(value.square(), dim=1) - torch.sum(coordinates.square(), dim=1))
            + inverse_log_det
        )
        log_mix_over_p = torch.logaddexp(
            torch.full_like(log_flow_over_p, math.log(config.defensive_weight)),
            math.log1p(-config.defensive_weight) + log_flow_over_p,
        )
        penalty = config.l2_penalty * sum(
            torch.mean(parameter.square()) for parameter in parameters
        )
        loss = -torch.sum(weights * log_mix_over_p) + penalty
        if not torch.isfinite(loss):
            raise FloatingPointError("low-rank flow training loss became nonfinite")
        loss.backward()
        optimizer.step()
    frozen_layers = tuple(
        FrozenLowRankCouplingLayer(
            active_indices=active,
            transformed_indices=transformed,
            scale_left=tuple(tuple(float(x) for x in row) for row in sl.detach()),
            scale_right=tuple(tuple(float(x) for x in row) for row in sr.detach()),
            scale_bias=tuple(float(x) for x in sb.detach()),
            shift_left=tuple(tuple(float(x) for x in row) for row in tl.detach()),
            shift_right=tuple(tuple(float(x) for x in row) for row in tr.detach()),
            shift_bias=tuple(float(x) for x in tb.detach()),
        )
        for active, transformed, sl, sr, sb, tl, tr, tb in raw_layers
    )
    conditioner_work_per_sample = sum(
        2 * layer.rank * (len(layer.active_indices) + len(layer.transformed_indices))
        for layer in frozen_layers
    )
    work = sample_count * config.epochs * conditioner_work_per_sample
    training_cost = BaselineCostLedger(
        training_samples=sample_count,
        optimizer_steps=config.epochs,
        hyperparameter_trials=1,
        algorithmic_work_units=float(work),
        wall_seconds=time.perf_counter() - started_wall,
        cpu_seconds=time.process_time() - started_cpu,
        peak_memory_bytes=process_peak_resident_memory_bytes(),
        measurement_mode="standardized_hardware_wall",
    )
    return freeze_low_rank_residual_flow(
        task_id=task_id,
        direction=direction,
        defensive_weight=config.defensive_weight,
        maximum_log_scale=config.maximum_log_scale,
        partition_style=config.partition_style,
        layers=frozen_layers,
        training_seed=training_seed,
        training_cost=training_cost,
    )
