"""Exact defensive coupling flows on a Gaussian residual hyperplane.

The map uses Householder coordinates for ``e^perp`` and triangular affine
couplings in ``R^(d-1)``.  A fixed natural component makes the final proposal
defensive, so its ordinary importance weight is exactly bounded by ``1/delta``.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass

import torch

from src.path_integral.baseline_framework import BaselineCostLedger
from src.path_integral.provenance import process_peak_resident_memory_bytes


@dataclass(frozen=True)
class ResidualHouseholderCoordinates:
    direction: torch.Tensor
    reflector: torch.Tensor | None
    tolerance: float = 1e-10

    @classmethod
    def build(
        cls, direction: torch.Tensor, *, tolerance: float = 1e-10
    ) -> ResidualHouseholderCoordinates:
        if direction.ndim != 1 or direction.numel() < 2:
            raise ValueError("direction must have dimension at least two")
        if direction.device.type != "cpu" or direction.dtype != torch.float64:
            raise ValueError("direction must be CPU float64")
        if not torch.isfinite(direction).all():
            raise ValueError("direction must be finite")
        if abs(float(torch.linalg.vector_norm(direction)) - 1.0) > tolerance:
            raise ValueError("direction must be unit norm")
        last = torch.zeros_like(direction)
        last[-1] = 1.0
        difference = last - direction
        norm = float(torch.linalg.vector_norm(difference))
        reflector = None if norm <= tolerance else difference / norm
        return cls(direction=direction.clone(), reflector=reflector, tolerance=tolerance)

    @property
    def ambient_dimension(self) -> int:
        return int(self.direction.numel())

    @property
    def coordinate_dimension(self) -> int:
        return self.ambient_dimension - 1

    def _reflect(self, values: torch.Tensor) -> torch.Tensor:
        if self.reflector is None:
            return values
        return values - 2.0 * (values @ self.reflector).unsqueeze(1) * self.reflector

    def from_coordinates(self, coordinates: torch.Tensor) -> torch.Tensor:
        if coordinates.ndim != 2 or coordinates.shape[1] != self.coordinate_dimension:
            raise ValueError("residual coordinates have the wrong shape")
        padded = torch.cat(
            (coordinates, torch.zeros((coordinates.shape[0], 1), dtype=torch.float64)),
            dim=1,
        )
        residual = self._reflect(padded)
        error = float(torch.amax(torch.abs(residual @ self.direction)))
        if error > self.tolerance * max(1.0, float(torch.amax(torch.abs(residual)))):
            raise FloatingPointError("Householder output left the residual hyperplane")
        return residual

    def to_coordinates(self, residual: torch.Tensor) -> torch.Tensor:
        if residual.ndim != 2 or residual.shape[1] != self.ambient_dimension:
            raise ValueError("residual has the wrong shape")
        projection = float(torch.amax(torch.abs(residual @ self.direction)))
        if projection > self.tolerance * max(1.0, float(torch.amax(torch.abs(residual)))):
            raise ValueError("input is not in the residual hyperplane")
        reflected = self._reflect(residual)
        if float(torch.amax(torch.abs(reflected[:, -1]))) > self.tolerance * max(
            1.0, float(torch.amax(torch.abs(reflected)))
        ):
            raise FloatingPointError("Householder inverse did not remove last coordinate")
        return reflected[:, :-1]


@dataclass(frozen=True)
class FrozenCouplingLayer:
    active_indices: tuple[int, ...]
    transformed_indices: tuple[int, ...]
    scale_weight: tuple[tuple[float, ...], ...]
    scale_bias: tuple[float, ...]
    shift_weight: tuple[tuple[float, ...], ...]
    shift_bias: tuple[float, ...]


@dataclass(frozen=True)
class FrozenResidualCouplingTransport:
    schema: str
    task_id: str
    direction: tuple[float, ...]
    defensive_weight: float
    maximum_log_scale: float
    layers: tuple[FrozenCouplingLayer, ...]
    training_seed: int
    training_cost: BaselineCostLedger
    exact_likelihood: bool
    self_normalized: bool
    frozen: bool
    sha256: str

    @property
    def dimension(self) -> int:
        return len(self.direction)


def _flow_payload(
    *,
    task_id: str,
    direction: tuple[float, ...],
    defensive_weight: float,
    maximum_log_scale: float,
    layers: tuple[FrozenCouplingLayer, ...],
    training_seed: int,
    training_cost: BaselineCostLedger,
) -> dict[str, object]:
    return {
        "schema": "npi.g11.v12-frozen-residual-coupling-flow.v1",
        "task_id": task_id,
        "direction": direction,
        "defensive_weight": defensive_weight,
        "maximum_log_scale": maximum_log_scale,
        "layers": [asdict(layer) for layer in layers],
        "training_seed": training_seed,
        "training_cost": asdict(training_cost),
        "exact_likelihood": True,
        "self_normalized": False,
        "frozen": True,
    }


def freeze_residual_coupling_transport(
    *,
    task_id: str,
    direction: torch.Tensor,
    defensive_weight: float,
    maximum_log_scale: float,
    layers: tuple[FrozenCouplingLayer, ...],
    training_seed: int,
    training_cost: BaselineCostLedger,
) -> FrozenResidualCouplingTransport:
    ResidualHouseholderCoordinates.build(direction)
    if not task_id.strip() or not 0.0 < defensive_weight < 1.0:
        raise ValueError("invalid flow identity or defensive weight")
    if not math.isfinite(maximum_log_scale) or maximum_log_scale <= 0.0:
        raise ValueError("maximum log scale must be positive")
    if not layers:
        raise ValueError("at least one coupling layer is required")
    payload = _flow_payload(
        task_id=task_id,
        direction=tuple(float(x) for x in direction),
        defensive_weight=defensive_weight,
        maximum_log_scale=maximum_log_scale,
        layers=layers,
        training_seed=training_seed,
        training_cost=training_cost,
    )
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return FrozenResidualCouplingTransport(
        schema="npi.g11.v12-frozen-residual-coupling-flow.v1",
        task_id=task_id,
        direction=tuple(float(x) for x in direction),
        defensive_weight=defensive_weight,
        maximum_log_scale=maximum_log_scale,
        layers=layers,
        training_seed=training_seed,
        training_cost=training_cost,
        exact_likelihood=True,
        self_normalized=False,
        frozen=True,
        sha256=digest,
    )


def _layer_tensors(
    layer: FrozenCouplingLayer,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(layer.scale_weight, dtype=torch.float64),
        torch.tensor(layer.scale_bias, dtype=torch.float64),
        torch.tensor(layer.shift_weight, dtype=torch.float64),
        torch.tensor(layer.shift_bias, dtype=torch.float64),
    )


def _forward_coordinates(
    base: torch.Tensor, proposal: FrozenResidualCouplingTransport
) -> tuple[torch.Tensor, torch.Tensor]:
    value = base.clone()
    log_det = torch.zeros(base.shape[0], dtype=torch.float64)
    for layer in proposal.layers:
        active = list(layer.active_indices)
        transformed = list(layer.transformed_indices)
        sw, sb, tw, tb = _layer_tensors(layer)
        source = value[:, active]
        log_scale = proposal.maximum_log_scale * torch.tanh(source @ sw + sb)
        shift = source @ tw + tb
        value[:, transformed] = value[:, transformed] * torch.exp(log_scale) + shift
        log_det += torch.sum(log_scale, dim=1)
    return value, log_det


def _inverse_coordinates(
    output: torch.Tensor, proposal: FrozenResidualCouplingTransport
) -> tuple[torch.Tensor, torch.Tensor]:
    value = output.clone()
    forward_log_det = torch.zeros(output.shape[0], dtype=torch.float64)
    for layer in reversed(proposal.layers):
        active = list(layer.active_indices)
        transformed = list(layer.transformed_indices)
        sw, sb, tw, tb = _layer_tensors(layer)
        source = value[:, active]
        log_scale = proposal.maximum_log_scale * torch.tanh(source @ sw + sb)
        shift = source @ tw + tb
        value[:, transformed] = (value[:, transformed] - shift) * torch.exp(-log_scale)
        forward_log_det += torch.sum(log_scale, dim=1)
    return value, forward_log_det


def residual_coupling_log_q_over_p(
    residual: torch.Tensor, proposal: FrozenResidualCouplingTransport
) -> torch.Tensor:
    coordinates = ResidualHouseholderCoordinates.build(
        torch.tensor(proposal.direction, dtype=torch.float64)
    ).to_coordinates(residual)
    base, forward_log_det = _inverse_coordinates(coordinates, proposal)
    log_flow_over_p = -0.5 * (
        torch.sum(base.square(), dim=1) - torch.sum(coordinates.square(), dim=1)
    ) - forward_log_det
    delta = proposal.defensive_weight
    return torch.logaddexp(
        torch.full_like(log_flow_over_p, math.log(delta)),
        math.log1p(-delta) + log_flow_over_p,
    )


@dataclass(frozen=True)
class ResidualCouplingSample:
    residual: torch.Tensor
    labels: torch.Tensor
    likelihood: torch.Tensor


def sample_residual_coupling_transport(
    proposal: FrozenResidualCouplingTransport,
    sample_count: int,
    *,
    gaussian_seed: int,
    label_seed: int,
) -> ResidualCouplingSample:
    if sample_count < 1 or gaussian_seed == label_seed:
        raise ValueError("invalid sample count or colliding seeds")
    coordinate_map = ResidualHouseholderCoordinates.build(
        torch.tensor(proposal.direction, dtype=torch.float64)
    )
    base = torch.randn(
        (sample_count, coordinate_map.coordinate_dimension),
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(gaussian_seed),
    )
    labels = (
        torch.rand(sample_count, generator=torch.Generator().manual_seed(label_seed))
        >= proposal.defensive_weight
    )
    flowed, _ = _forward_coordinates(base, proposal)
    coordinates = torch.where(labels.unsqueeze(1), flowed, base)
    residual = coordinate_map.from_coordinates(coordinates)
    likelihood = torch.exp(-residual_coupling_log_q_over_p(residual, proposal))
    if float(torch.amax(likelihood)) > (1.0 / proposal.defensive_weight) * (1.0 + 1e-10):
        raise FloatingPointError("defensive flow likelihood bound failed")
    return ResidualCouplingSample(residual=residual, labels=labels, likelihood=likelihood)


@dataclass(frozen=True)
class ResidualCouplingTrainingConfig:
    layers: int = 4
    epochs: int = 100
    learning_rate: float = 0.01
    defensive_weight: float = 0.1
    maximum_log_scale: float = 1.0
    l2_penalty: float = 1e-6


def fit_residual_coupling_transport(
    *,
    task_id: str,
    residual: torch.Tensor,
    direction: torch.Tensor,
    log_unnormalized_target_over_sampling: torch.Tensor,
    training_seed: int,
    config: ResidualCouplingTrainingConfig | None = None,
) -> FrozenResidualCouplingTransport:
    """Fit an exact flow with normalized weights used only inside training."""

    config = config or ResidualCouplingTrainingConfig()
    coordinates = ResidualHouseholderCoordinates.build(direction).to_coordinates(residual)
    if log_unnormalized_target_over_sampling.shape != (residual.shape[0],):
        raise ValueError("training log weights have the wrong shape")
    if bool(torch.isnan(log_unnormalized_target_over_sampling).any()):
        raise ValueError("training log weights contain NaN")
    weights = torch.softmax(log_unnormalized_target_over_sampling, dim=0).detach()
    dimension = coordinates.shape[1]
    if dimension < 2:
        raise ValueError("coupling flow requires residual dimension at least two")
    generator = torch.Generator().manual_seed(training_seed)
    raw: list[tuple[tuple[int, ...], tuple[int, ...], torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter]] = []
    parameters: list[torch.nn.Parameter] = []
    for index in range(config.layers):
        active = tuple(i for i in range(dimension) if i % 2 == index % 2)
        transformed = tuple(i for i in range(dimension) if i % 2 != index % 2)
        sw = torch.nn.Parameter(0.01 * torch.randn((len(active), len(transformed)), dtype=torch.float64, generator=generator))
        sb = torch.nn.Parameter(torch.zeros(len(transformed), dtype=torch.float64))
        tw = torch.nn.Parameter(0.01 * torch.randn((len(active), len(transformed)), dtype=torch.float64, generator=generator))
        tb = torch.nn.Parameter(torch.zeros(len(transformed), dtype=torch.float64))
        raw.append((active, transformed, sw, sb, tw, tb))
        parameters.extend((sw, sb, tw, tb))
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    for _ in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        value = coordinates.clone()
        inverse_log_det = torch.zeros(coordinates.shape[0], dtype=torch.float64)
        for active, transformed, sw, sb, tw, tb in reversed(raw):
            source = value[:, list(active)]
            scale = config.maximum_log_scale * torch.tanh(source @ sw + sb)
            shift = source @ tw + tb
            value[:, list(transformed)] = (value[:, list(transformed)] - shift) * torch.exp(-scale)
            inverse_log_det -= torch.sum(scale, dim=1)
        log_flow_over_p = -0.5 * (
            torch.sum(value.square(), dim=1) - torch.sum(coordinates.square(), dim=1)
        ) + inverse_log_det
        log_mix_over_p = torch.logaddexp(
            torch.full_like(log_flow_over_p, math.log(config.defensive_weight)),
            math.log1p(-config.defensive_weight) + log_flow_over_p,
        )
        penalty = config.l2_penalty * sum(torch.mean(parameter.square()) for parameter in parameters)
        loss = -torch.sum(weights * log_mix_over_p) + penalty
        if not torch.isfinite(loss):
            raise FloatingPointError("coupling-flow training loss became nonfinite")
        loss.backward()
        optimizer.step()
    layers = tuple(
        FrozenCouplingLayer(
            active_indices=active,
            transformed_indices=transformed,
            scale_weight=tuple(tuple(float(x) for x in row) for row in sw.detach()),
            scale_bias=tuple(float(x) for x in sb.detach()),
            shift_weight=tuple(tuple(float(x) for x in row) for row in tw.detach()),
            shift_bias=tuple(float(x) for x in tb.detach()),
        )
        for active, transformed, sw, sb, tw, tb in raw
    )
    work = residual.shape[0] * config.epochs * config.layers * dimension * dimension
    cost = BaselineCostLedger(
        training_samples=residual.shape[0],
        optimizer_steps=config.epochs,
        hyperparameter_trials=1,
        algorithmic_work_units=float(work),
        wall_seconds=time.perf_counter() - started_wall,
        cpu_seconds=time.process_time() - started_cpu,
        peak_memory_bytes=process_peak_resident_memory_bytes(),
        measurement_mode="standardized_hardware_wall",
    )
    return freeze_residual_coupling_transport(
        task_id=task_id,
        direction=direction,
        defensive_weight=config.defensive_weight,
        maximum_log_scale=config.maximum_log_scale,
        layers=layers,
        training_seed=training_seed,
        training_cost=cost,
    )
