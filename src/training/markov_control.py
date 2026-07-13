"""Per-instance Markov neural control for the Heston G2 benchmark."""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.physics_engine import MarketSimulator

ObjectiveName = Literal["scaled_second_moment", "log_second_moment", "entropy_stress"]
ArchitectureName = Literal["affine", "mlp"]
FeatureMapName = Literal["linear", "log"]


class MarkovianHestonControl(nn.Module):
    """Bounded state/time feedback ``u(t, S_t, v_t)`` for one rare event.

    Inputs are dimensionless and centered around the benchmark task.  The
    running-average argument is accepted only for simulator compatibility and
    is intentionally ignored: this is the Markov baseline, not a path-memory
    controller.
    """

    def __init__(
        self,
        *,
        initial_spot: float,
        barrier: float,
        maturity: float,
        variance_scale: float,
        hidden_dim: int = 32,
        n_layers: int = 2,
        architecture: ArchitectureName = "mlp",
        feature_map: FeatureMapName = "log",
        control_bound: float = 8.0,
        initial_constant: float = 0.0,
    ) -> None:
        super().__init__()
        if initial_spot <= 0.0 or barrier <= 0.0 or maturity <= 0.0:
            raise ValueError("initial_spot, barrier, and maturity must be positive")
        if variance_scale <= 0.0 or control_bound <= 0.0:
            raise ValueError("variance_scale and control_bound must be positive")
        if architecture not in ("affine", "mlp"):
            raise ValueError("architecture must be 'affine' or 'mlp'")
        if feature_map not in ("linear", "log"):
            raise ValueError("feature_map must be 'linear' or 'log'")
        if n_layers <= 0 or hidden_dim <= 0:
            raise ValueError("hidden_dim and n_layers must be positive")
        self.initial_spot = float(initial_spot)
        self.barrier = float(barrier)
        self.maturity = float(maturity)
        self.variance_scale = float(variance_scale)
        self.control_bound = float(control_bound)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.architecture = architecture
        self.feature_map = feature_map

        # In an affine model S/S0 and S/K are collinear functions of S. Keep
        # only S/K to avoid a non-identifiable redundant coefficient. The MLP
        # retains both nonlinear coordinates.
        input_dim = 3 if architecture == "affine" else 4
        self.features: nn.Module
        if architecture == "affine":
            self.features = nn.Identity()
            self.output = nn.Linear(input_dim, 1)
        else:
            layers: list[nn.Module] = []
            for layer_index in range(n_layers):
                layers.append(nn.Linear(input_dim if layer_index == 0 else hidden_dim, hidden_dim))
                layers.append(nn.SiLU())
            self.features = nn.Sequential(*layers)
            self.output = nn.Linear(hidden_dim, 1)
        self.initialize_constant(initial_constant)

    def initialize_constant(self, constant: float) -> None:
        """Initialize exactly at a constant proposal while retaining trainability."""
        if abs(constant) >= self.control_bound:
            raise ValueError("initial constant must lie strictly inside the control bound")
        nn.init.zeros_(self.output.weight)
        normalized = constant / self.control_bound
        bias = 0.5 * math.log((1.0 + normalized) / (1.0 - normalized))
        nn.init.constant_(self.output.bias, bias)

    def forward(
        self,
        time: float | torch.Tensor,
        spot: torch.Tensor,
        variance: torch.Tensor,
        _average_spot: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.architecture == "affine" and self.feature_map == "linear":
            weights = self.output.weight[0]
            time_feature = (
                time.to(device=spot.device, dtype=spot.dtype) / self.maturity
                if torch.is_tensor(time)
                else float(time) / self.maturity
            )
            raw = (
                (weights[0] / self.barrier) * spot
                + (weights[1] / self.variance_scale) * variance
                + weights[2] * time_feature
                + self.output.bias[0]
                - weights[0]
                - weights[1]
            )
            return self.control_bound * torch.tanh(raw)
        time_tensor = (
            time.to(device=spot.device, dtype=spot.dtype)
            if torch.is_tensor(time)
            else torch.tensor(float(time), device=spot.device, dtype=spot.dtype)
        ).expand_as(spot)
        safe_spot = torch.clamp(spot, min=1e-12)
        safe_variance = torch.clamp(variance, min=1e-10)
        if self.architecture == "affine":
            price_feature = torch.log(safe_spot / self.barrier)
            variance_feature = torch.log(safe_variance / self.variance_scale)
            weights = self.output.weight[0]
            raw = (
                weights[0] * price_feature
                + weights[1] * variance_feature
                + weights[2] * (time_tensor / self.maturity)
                + self.output.bias[0]
            )
            return self.control_bound * torch.tanh(raw)
        if self.feature_map == "linear":
            inputs = torch.stack(
                [
                    safe_spot / self.initial_spot - 1.0,
                    safe_spot / self.barrier - 1.0,
                    safe_variance / self.variance_scale - 1.0,
                    time_tensor / self.maturity,
                ],
                dim=-1,
            )
        else:
            inputs = torch.stack(
                [
                    torch.log(safe_spot / self.initial_spot),
                    torch.log(safe_spot / self.barrier),
                    torch.log(safe_variance / self.variance_scale),
                    time_tensor / self.maturity,
                ],
                dim=-1,
            )
        raw = self.output(self.features(inputs)).squeeze(-1)
        return self.control_bound * torch.tanh(raw)

    def inference_control_fn(
        self,
    ) -> Callable[[float, torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor]:
        """Return a numerically equivalent low-overhead frozen callable."""
        if self.architecture != "affine" or self.feature_map != "linear":
            return self.forward
        with torch.no_grad():
            weights = self.output.weight[0]
            spot_coefficient = float(weights[0] / self.barrier)
            variance_coefficient = float(weights[1] / self.variance_scale)
            time_coefficient = float(weights[2] / self.maturity)
            intercept = float(self.output.bias[0] - weights[0] - weights[1])
            bound = self.control_bound

        def frozen_control(
            time: float,
            spot: torch.Tensor,
            variance: torch.Tensor,
            _average_spot: torch.Tensor | None = None,
        ) -> torch.Tensor:
            raw = (
                spot_coefficient * spot
                + variance_coefficient * variance
                + time_coefficient * time
                + intercept
            )
            return bound * torch.tanh(raw)

        return frozen_control

    def configuration(self) -> dict[str, float | int | str]:
        return {
            "initial_spot": self.initial_spot,
            "barrier": self.barrier,
            "maturity": self.maturity,
            "variance_scale": self.variance_scale,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "architecture": self.architecture,
            "feature_map": self.feature_map,
            "control_bound": self.control_bound,
        }


def markov_control_state_sha256(control: MarkovianHestonControl) -> str:
    """Hash tensor names, shapes, dtypes, and bytes in stable sorted order."""
    digest = hashlib.sha256()
    for name, tensor in sorted(control.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def save_markovian_control_checkpoint(
    path: str | Path,
    control: MarkovianHestonControl,
    *,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Save a versioned checkpoint and return its stable state hash."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    state_hash = markov_control_state_sha256(control)
    torch.save(
        {
            "schema_version": 1,
            "model_class": "MarkovianHestonControl",
            "configuration": control.configuration(),
            "state_dict": control.state_dict(),
            "state_sha256": state_hash,
            "metadata": metadata or {},
        },
        checkpoint_path,
    )
    return state_hash


def load_markovian_control_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[MarkovianHestonControl, dict[str, Any]]:
    """Load and integrity-check checkpoint schema version 1."""
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported Markov control checkpoint schema")
    if payload.get("model_class") != "MarkovianHestonControl":
        raise ValueError("checkpoint model class mismatch")
    configuration = payload.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("checkpoint configuration is missing")
    control = MarkovianHestonControl(initial_constant=0.0, **configuration).to(device)
    control.load_state_dict(payload["state_dict"])
    expected_hash = payload.get("state_sha256")
    actual_hash = markov_control_state_sha256(control)
    if expected_hash != actual_hash:
        raise ValueError("checkpoint state hash mismatch")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata must be a mapping")
    return control, metadata


@dataclass(frozen=True)
class MarkovObjectiveDiagnostics:
    loss: torch.Tensor
    log_second_moment: torch.Tensor
    event_fraction_under_proposal: torch.Tensor
    estimate: torch.Tensor
    kl_estimate: torch.Tensor


def markov_control_objective(
    simulator: MarketSimulator,
    control: MarkovianHestonControl,
    *,
    spot: float,
    variance: float,
    maturity: float,
    dt: float,
    barrier: float,
    num_paths: int,
    objective: ObjectiveName,
    reference_probability: float,
    entropy_temperature: float = 2.0,
    entropy_kl_weight: float = 0.05,
) -> MarkovObjectiveDiagnostics:
    """Simulate one differentiable batch and compute a selected objective."""
    if reference_probability <= 0.0:
        raise ValueError("reference_probability must be positive")
    brownian_increments: list[torch.Tensor] = []

    def observe_brownian(_time: float, increment: torch.Tensor) -> None:
        brownian_increments.append(increment.detach())

    paths, variance_paths, log_weight, _barrier, _average = simulator.simulate_controlled(
        S0=spot,
        v0=variance,
        T=maturity,
        dt=dt,
        num_paths=num_paths,
        control_fn=control,
        brownian_observer=observe_brownian,
    )
    terminal = paths[:, -1]
    event = terminal <= barrier
    event_fraction = event.float().mean()
    if not event.any():
        raise RuntimeError("training batch contains no target events; use a stronger warm start")

    negative_infinity = torch.full_like(log_weight, -torch.inf)
    log_squared_contribution = torch.where(event, 2.0 * log_weight, negative_infinity)
    log_second_moment = torch.logsumexp(log_squared_contribution, dim=0) - math.log(num_paths)
    estimate = torch.mean(event.to(log_weight.dtype) * torch.exp(log_weight))
    kl_estimate = -log_weight.mean()

    if objective in ("scaled_second_moment", "log_second_moment"):
        # A hard indicator has no valid ordinary pathwise derivative at its
        # moving boundary. Use the exact likelihood-ratio gradient instead.
        # At fixed sampled trajectory, ∇log q_theta is
        # Σ_k ∇u_theta(t_k, X_k) ΔW_k^Q. States and Brownian increments are
        # detached so autograd differentiates q, not the trajectory map.
        if len(brownian_increments) != paths.shape[1] - 1:
            raise RuntimeError("Brownian diagnostic length does not match the simulated grid")
        score_log_q = torch.zeros_like(log_weight)
        step_dt = maturity / len(brownian_increments)
        for step, brownian_increment in enumerate(brownian_increments):
            time = step * step_dt
            fixed_spot = paths[:, step].detach()
            fixed_variance = variance_paths[:, step].detach()
            fixed_control = control(time, fixed_spot, fixed_variance, None)
            score_log_q = score_log_q + fixed_control * brownian_increment

        squared_contribution = event.to(log_weight.dtype) * torch.exp(2.0 * log_weight)
        if objective == "scaled_second_moment":
            objective_value = torch.exp(log_second_moment - 2.0 * math.log(reference_probability))
            gradient_surrogate = -torch.mean(squared_contribution.detach() * score_log_q) / (
                reference_probability**2
            )
        else:
            objective_value = log_second_moment
            normalized_terms = torch.softmax(log_squared_contribution, dim=0).detach()
            gradient_surrogate = -torch.sum(normalized_terms * score_log_q)
        # Report the true objective value while taking the unbiased score
        # gradient supplied by gradient_surrogate.
        loss = gradient_surrogate - gradient_surrogate.detach() + objective_value.detach()
    elif objective == "entropy_stress":
        if entropy_temperature <= 0.0 or entropy_kl_weight < 0.0:
            raise ValueError("entropy temperature must be positive and KL weight nonnegative")
        reward = F.softplus((barrier - terminal) / entropy_temperature).mean()
        loss = -reward + entropy_kl_weight * kl_estimate
    else:
        raise ValueError(f"unknown objective: {objective}")
    return MarkovObjectiveDiagnostics(
        loss=loss,
        log_second_moment=log_second_moment,
        event_fraction_under_proposal=event_fraction,
        estimate=estimate,
        kl_estimate=kl_estimate,
    )


@dataclass(frozen=True)
class MarkovTrainingEpoch:
    epoch: int
    train_seed: int
    loss: float
    train_log_second_moment: float
    train_event_fraction: float
    validation_log_second_moment: float | None


@dataclass(frozen=True)
class MarkovTrainingResult:
    objective: ObjectiveName
    best_validation_log_second_moment: float
    history: tuple[MarkovTrainingEpoch, ...]


def train_markovian_control(
    simulator: MarketSimulator,
    control: MarkovianHestonControl,
    *,
    spot: float,
    variance: float,
    maturity: float,
    dt: float,
    barrier: float,
    reference_probability: float,
    objective: ObjectiveName,
    train_seeds: tuple[int, ...],
    validation_seeds: tuple[int, ...],
    epochs: int = 50,
    paths_per_batch: int = 5_000,
    validation_paths: int = 10_000,
    learning_rate: float = 1e-3,
    validate_every: int = 5,
    gradient_clip: float = 5.0,
) -> MarkovTrainingResult:
    """Train on fixed seeds and select checkpoints only on validation seeds."""
    if not train_seeds or not validation_seeds:
        raise ValueError("train and validation seeds must be nonempty")
    if set(train_seeds) & set(validation_seeds):
        raise ValueError("train and validation seeds must be disjoint")
    if epochs <= 0 or paths_per_batch <= 0 or validation_paths <= 0:
        raise ValueError("epochs and path counts must be positive")
    if validate_every <= 0:
        raise ValueError("validate_every must be positive")

    optimizer = torch.optim.Adam(control.parameters(), lr=learning_rate)
    best_metric = math.inf
    best_state = copy.deepcopy(control.state_dict())
    history: list[MarkovTrainingEpoch] = []

    for epoch in range(1, epochs + 1):
        root_index = (epoch - 1) % len(train_seeds)
        stream_index = (epoch - 1) // len(train_seeds)
        # Root seeds define the training split, while every epoch receives a
        # distinct deterministic substream. Resetting the exact same root each
        # cycle would repeatedly fit the same Brownian trajectories.
        train_seed = int((train_seeds[root_index] + 1_000_003 * stream_index) % (2**63 - 1))
        torch.manual_seed(train_seed)
        control.train()
        optimizer.zero_grad()
        diagnostics = markov_control_objective(
            simulator,
            control,
            spot=spot,
            variance=variance,
            maturity=maturity,
            dt=dt,
            barrier=barrier,
            num_paths=paths_per_batch,
            objective=objective,
            reference_probability=reference_probability,
        )
        diagnostics.loss.backward()
        torch.nn.utils.clip_grad_norm_(control.parameters(), gradient_clip)
        optimizer.step()

        validation_metric: float | None = None
        if epoch % validate_every == 0 or epoch == epochs:
            control.eval()
            metrics: list[float] = []
            with torch.no_grad():
                for seed in validation_seeds:
                    torch.manual_seed(seed)
                    validation = markov_control_objective(
                        simulator,
                        control,
                        spot=spot,
                        variance=variance,
                        maturity=maturity,
                        dt=dt,
                        barrier=barrier,
                        num_paths=validation_paths,
                        objective="log_second_moment",
                        reference_probability=reference_probability,
                    )
                    metrics.append(float(validation.log_second_moment))
            validation_metric = float(sum(metrics) / len(metrics))
            if validation_metric < best_metric:
                best_metric = validation_metric
                best_state = copy.deepcopy(control.state_dict())

        history.append(
            MarkovTrainingEpoch(
                epoch=epoch,
                train_seed=train_seed,
                loss=float(diagnostics.loss.detach()),
                train_log_second_moment=float(diagnostics.log_second_moment.detach()),
                train_event_fraction=float(diagnostics.event_fraction_under_proposal.detach()),
                validation_log_second_moment=validation_metric,
            )
        )

    control.load_state_dict(best_state)
    return MarkovTrainingResult(
        objective=objective,
        best_validation_log_second_moment=best_metric,
        history=tuple(history),
    )
