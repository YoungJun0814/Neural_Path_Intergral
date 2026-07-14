"""Two-driver Heston feedback control for the G1 path-integral gate.

The module keeps three mathematically different training stages separate:

* soft path-integral (PI) stochastic control under the current proposal;
* feedback PICE, an off-policy reverse-KL projection on target coordinates;
* hard-event second-moment (J2) refinement with a score-function gradient.

All controls use the independent Brownian basis implemented by
``MarketSimulator.simulate_controlled_two_driver``.  In particular,
``dW_target = dW_proposal + u dt`` and the simulator returns ``log(dP/dQ)``.
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

from src.evaluation.heston_reference import HestonReferenceParams
from src.path_integral.heston_oracle import HestonOracleNumerics, heston_soft_oracle_control
from src.path_integral.potentials import terminal_left_tail_potential
from src.physics_engine import MarketSimulator, TwoDriverHestonPaths

FeedbackArchitecture = Literal["affine", "mlp"]


class TwoDriverHestonControl(nn.Module):
    """Bounded Markov feedback in the independent Heston Brownian basis.

    The three inputs ``log(S/K)``, ``log(v/v_scale)`` and ``t/T`` are
    dimensionless.  Unlike the historical one-driver controller, the feature
    map does not contain two affinely redundant spot coordinates.
    """

    def __init__(
        self,
        *,
        barrier: float,
        maturity: float,
        variance_scale: float,
        architecture: FeedbackArchitecture = "mlp",
        hidden_dim: int = 32,
        n_layers: int = 2,
        control_bound: float | Sequence[float] = 8.0,
        initial_control: Sequence[float] = (0.0, 0.0),
    ) -> None:
        super().__init__()
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (barrier, maturity, variance_scale)
        ):
            raise ValueError("barrier, maturity, and variance_scale must be finite and positive")
        if architecture not in ("affine", "mlp"):
            raise ValueError("architecture must be 'affine' or 'mlp'")
        if hidden_dim <= 0 or n_layers <= 0:
            raise ValueError("hidden_dim and n_layers must be positive")

        if isinstance(control_bound, (float, int)):
            bounds = (float(control_bound), float(control_bound))
        else:
            bounds = tuple(float(value) for value in control_bound)
        initial = tuple(float(value) for value in initial_control)
        if len(bounds) != 2 or len(initial) != 2:
            raise ValueError("control_bound and initial_control must contain two coordinates")
        if not all(math.isfinite(value) and value > 0.0 for value in bounds):
            raise ValueError("control bounds must be finite and positive")
        if not all(math.isfinite(value) for value in initial):
            raise ValueError("initial controls must be finite")
        if any(abs(value) >= bound for value, bound in zip(initial, bounds, strict=True)):
            raise ValueError("initial controls must lie strictly inside their bounds")

        self.barrier = float(barrier)
        self.maturity = float(maturity)
        self.variance_scale = float(variance_scale)
        self.architecture = architecture
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.register_buffer("control_bounds", torch.tensor(bounds, dtype=torch.float32))

        if architecture == "affine":
            self.features: nn.Module = nn.Identity()
            self.output = nn.Linear(3, 2)
        else:
            layers: list[nn.Module] = []
            for index in range(n_layers):
                layers.append(nn.Linear(3 if index == 0 else hidden_dim, hidden_dim))
                layers.append(nn.SiLU())
            self.features = nn.Sequential(*layers)
            self.output = nn.Linear(hidden_dim, 2)
        self.initialize_constant(initial)

    def initialize_constant(self, control: Sequence[float]) -> None:
        """Initialize the network to an exact constant two-driver control."""
        values = tuple(float(value) for value in control)
        if len(values) != 2:
            raise ValueError("control must contain exactly two coordinates")
        bounds = tuple(float(value) for value in self.control_bounds.detach().cpu())
        if any(abs(value) >= bound for value, bound in zip(values, bounds, strict=True)):
            raise ValueError("constant controls must lie strictly inside their bounds")
        # Only the final layer is zeroed.  Keeping the hidden feature map at
        # its ordinary random initialization gives the output weights a
        # nonzero learning signal on the first update while the represented
        # initial policy remains exactly constant.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        normalized = torch.tensor(
            [value / bound for value, bound in zip(values, bounds, strict=True)],
            dtype=self.output.bias.dtype,
            device=self.output.bias.device,
        )
        with torch.no_grad():
            self.output.bias.copy_(torch.atanh(normalized))

    def forward(
        self,
        time: float | torch.Tensor,
        spot: torch.Tensor,
        variance: torch.Tensor,
        _average_spot: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if spot.shape != variance.shape:
            raise ValueError("spot and variance must have identical shapes")
        if not spot.is_floating_point() or not variance.is_floating_point():
            raise TypeError("spot and variance must be floating point")
        safe_spot = torch.clamp(spot, min=torch.finfo(spot.dtype).tiny)
        safe_variance = torch.clamp(variance, min=1e-10)
        time_tensor = (
            time.to(device=spot.device, dtype=spot.dtype)
            if torch.is_tensor(time)
            else torch.as_tensor(float(time), device=spot.device, dtype=spot.dtype)
        ).expand_as(spot)
        inputs = torch.stack(
            (
                torch.log(safe_spot / self.barrier),
                torch.log(safe_variance / self.variance_scale),
                time_tensor / self.maturity,
            ),
            dim=-1,
        )
        raw = self.output(self.features(inputs))
        bounds = self.control_bounds.to(device=raw.device, dtype=raw.dtype)
        return bounds * torch.tanh(raw)

    def frozen_copy(self) -> TwoDriverHestonControl:
        """Return an evaluation-only behavior policy with no shared parameters."""
        result = copy.deepcopy(self)
        result.eval()
        for parameter in result.parameters():
            parameter.requires_grad_(False)
        return result

    def configuration(self) -> dict[str, float | int | str | tuple[float, float]]:
        """Return the architecture fields required for a safe checkpoint reload."""
        return {
            "barrier": self.barrier,
            "maturity": self.maturity,
            "variance_scale": self.variance_scale,
            "architecture": self.architecture,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "control_bound": tuple(
                float(value) for value in self.control_bounds.detach().cpu()
            ),
        }


def two_driver_control_state_sha256(control: TwoDriverHestonControl) -> str:
    """Hash names, shapes, dtypes, and bytes of a two-driver policy state."""
    digest = hashlib.sha256()
    for name, tensor in sorted(control.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def save_two_driver_control_checkpoint(
    path: str | Path,
    control: TwoDriverHestonControl,
    *,
    metadata: dict[str, object] | None = None,
) -> str:
    """Save a versioned feedback checkpoint and return its state hash."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    state_hash = two_driver_control_state_sha256(control)
    torch.save(
        {
            "schema_version": 1,
            "model_class": "TwoDriverHestonControl",
            "configuration": control.configuration(),
            "state_dict": control.state_dict(),
            "state_sha256": state_hash,
            "metadata": metadata or {},
        },
        destination,
    )
    return state_hash


def load_two_driver_control_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[TwoDriverHestonControl, dict[str, object]]:
    """Load and integrity-check a schema-version-1 feedback checkpoint."""
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported two-driver feedback checkpoint schema")
    if payload.get("model_class") != "TwoDriverHestonControl":
        raise ValueError("checkpoint model class mismatch")
    configuration = payload.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("checkpoint configuration is missing")
    control = TwoDriverHestonControl(initial_control=(0.0, 0.0), **configuration).to(device)
    control.load_state_dict(payload["state_dict"])
    if payload.get("state_sha256") != two_driver_control_state_sha256(control):
        raise ValueError("checkpoint state hash mismatch")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata must be a mapping")
    return control, metadata


@dataclass(frozen=True)
class HestonOracleDataset:
    """Finite deterministic state grid and matching two-driver oracle controls."""

    time: torch.Tensor
    spot: torch.Tensor
    variance: torch.Tensor
    control: torch.Tensor
    maximum_gradient_discrepancy: float

    def validate(self) -> None:
        paths = self.time.shape[0]
        if self.time.shape != (paths,) or self.spot.shape != (paths,):
            raise ValueError("oracle state tensors must have shape (samples,)")
        if self.variance.shape != (paths,) or self.control.shape != (paths, 2):
            raise ValueError("oracle variance/control shapes are inconsistent")
        tensors = (self.time, self.spot, self.variance, self.control)
        if not all(tensor.is_floating_point() for tensor in tensors):
            raise TypeError("oracle tensors must be floating point")
        if not all(torch.isfinite(tensor).all() for tensor in tensors):
            raise ValueError("oracle dataset must be finite")
        if paths < 1 or not math.isfinite(self.maximum_gradient_discrepancy):
            raise ValueError("oracle dataset must be nonempty with a finite diagnostic")


def build_heston_oracle_dataset(
    *,
    times: Sequence[float],
    spots: Sequence[float],
    variances: Sequence[float],
    maturity: float,
    barrier: float,
    temperature: float,
    params: HestonReferenceParams,
    numerics: HestonOracleNumerics | None = None,
    dtype: torch.dtype = torch.float32,
) -> HestonOracleDataset:
    """Evaluate the deterministic soft Heston oracle on a Cartesian grid."""
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("dtype must be floating point")
    rows: list[tuple[float, float, float, float, float]] = []
    maximum_discrepancy = 0.0
    for time in times:
        if not math.isfinite(time) or not 0.0 <= time < maturity:
            raise ValueError("oracle times must be finite and lie in [0, maturity)")
        for spot in spots:
            for variance in variances:
                oracle = heston_soft_oracle_control(
                    spot=float(spot),
                    variance=float(variance),
                    remaining_time=maturity - float(time),
                    barrier=barrier,
                    temperature=temperature,
                    params=params,
                    numerics=numerics,
                )
                maximum_discrepancy = max(
                    maximum_discrepancy,
                    oracle.gradient.log_spot_error_estimate,
                    oracle.gradient.variance_error_estimate,
                )
                rows.append(
                    (
                        float(time),
                        float(spot),
                        float(variance),
                        oracle.control_1,
                        oracle.control_2,
                    )
                )
    values = torch.tensor(rows, dtype=dtype)
    dataset = HestonOracleDataset(
        time=values[:, 0],
        spot=values[:, 1],
        variance=values[:, 2],
        control=values[:, 3:5],
        maximum_gradient_discrepancy=maximum_discrepancy,
    )
    dataset.validate()
    return dataset


@dataclass(frozen=True)
class OracleAlignment:
    rmse: float
    normalized_rmse: float
    mean_cosine: float
    sign_agreement: float


def oracle_alignment(
    control: TwoDriverHestonControl,
    dataset: HestonOracleDataset,
) -> OracleAlignment:
    """Measure vector-field recovery without using a publication seed."""
    dataset.validate()
    control.eval()
    device = next(control.parameters()).device
    dtype = next(control.parameters()).dtype
    with torch.no_grad():
        prediction = control(
            dataset.time.to(device=device, dtype=dtype),
            dataset.spot.to(device=device, dtype=dtype),
            dataset.variance.to(device=device, dtype=dtype),
            None,
        ).cpu().double()
    target = dataset.control.cpu().double()
    error = prediction - target
    rmse = float(torch.sqrt(torch.mean(error.square())))
    scale = float(torch.sqrt(torch.mean(target.square())).clamp_min(1e-12))
    cosine = torch.nn.functional.cosine_similarity(prediction, target, dim=-1, eps=1e-12)
    active = torch.abs(target) > 1e-8
    sign_agreement = float(torch.mean((torch.sign(prediction[active]) == torch.sign(target[active])).double()))
    return OracleAlignment(
        rmse=rmse,
        normalized_rmse=rmse / scale,
        mean_cosine=float(torch.mean(cosine)),
        sign_agreement=sign_agreement,
    )


def fit_heston_oracle_distillation(
    control: TwoDriverHestonControl,
    dataset: HestonOracleDataset,
    *,
    epochs: int = 500,
    learning_rate: float = 3e-3,
) -> tuple[float, ...]:
    """Supervise the feedback field on a deterministic oracle grid."""
    dataset.validate()
    if epochs <= 0 or not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("epochs and learning_rate must be positive")
    device = next(control.parameters()).device
    dtype = next(control.parameters()).dtype
    time = dataset.time.to(device=device, dtype=dtype)
    spot = dataset.spot.to(device=device, dtype=dtype)
    variance = dataset.variance.to(device=device, dtype=dtype)
    target = dataset.control.to(device=device, dtype=dtype)
    target_scale = torch.sqrt(torch.mean(target.square(), dim=0)).clamp_min(0.1)
    optimizer = torch.optim.Adam(control.parameters(), lr=learning_rate)
    history: list[float] = []
    control.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        prediction = control(time, spot, variance, None)
        loss = torch.mean(((prediction - target) / target_scale).square())
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return tuple(history)


@dataclass(frozen=True)
class SoftPIObjective:
    loss: torch.Tensor
    potential_mean: torch.Tensor
    energy_mean: torch.Tensor
    soft_estimate: torch.Tensor
    proposal_event_fraction: torch.Tensor


def soft_pi_objective(
    simulator: MarketSimulator,
    control: TwoDriverHestonControl,
    *,
    spot: float,
    variance: float,
    maturity: float,
    dt: float,
    barrier: float,
    temperature: float,
    num_paths: int,
) -> SoftPIObjective:
    r"""Return ``E_Q[Phi(X_T) + 1/2 int ||u||^2 dt]`` and diagnostics.

    This is the Gibbs variational/path-integral stochastic-control objective
    for ``g=exp(-Phi)``.  It is a soft proposal-shaping objective, not the
    hard-event second moment.
    """
    result = simulator.simulate_controlled_two_driver(
        S0=spot,
        v0=variance,
        T=maturity,
        dt=dt,
        num_paths=num_paths,
        control_fn=control,
        record_brownian=False,
        dtype=next(control.parameters()).dtype,
    )
    potential = terminal_left_tail_potential(result.spot[:, -1], barrier, temperature)
    loss = potential.double().mean() + 0.5 * result.control_energy.mean()
    log_contribution = -potential.double() + result.log_likelihood
    soft_estimate = torch.exp(torch.logsumexp(log_contribution, dim=0) - math.log(num_paths))
    return SoftPIObjective(
        loss=loss,
        potential_mean=potential.detach().mean(),
        energy_mean=result.control_energy.detach().mean(),
        soft_estimate=soft_estimate.detach(),
        proposal_event_fraction=(result.spot[:, -1] <= barrier).double().mean().detach(),
    )


def _prefix_running_average(spot_paths: torch.Tensor, step: int) -> torch.Tensor:
    if step == 0:
        return spot_paths[:, 0]
    return torch.mean(spot_paths[:, :step], dim=1)


def candidate_log_density_on_target_paths(
    control: TwoDriverHestonControl,
    paths: TwoDriverHestonPaths,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Replay a candidate and return ``log(dQ_theta/dP)`` on target paths.

    States and target increments are treated as canonical data.  The replay is
    causal because the control at step ``i`` only sees state index ``i``.
    """
    target = paths.target_brownian_increments
    if target is None:
        raise ValueError("target Brownian increments were not recorded")
    steps = target.shape[1]
    candidate_controls: list[torch.Tensor] = []
    detached_spot = paths.spot.detach()
    detached_variance = paths.variance.detach()
    for step in range(steps):
        candidate_controls.append(
            control(
                step * paths.step_dt,
                detached_spot[:, step],
                detached_variance[:, step],
                _prefix_running_average(detached_spot, step),
            )
        )
    controls = torch.stack(candidate_controls, dim=1)
    target_for_density = target.detach().to(dtype=controls.dtype)
    log_density = torch.sum(controls * target_for_density, dim=(1, 2))
    log_density = log_density - 0.5 * paths.step_dt * torch.sum(
        controls.square(), dim=(1, 2)
    )
    return log_density, controls


@dataclass(frozen=True)
class FeedbackPICEObjective:
    loss: torch.Tensor
    effective_sample_size: torch.Tensor
    effective_sample_fraction: torch.Tensor
    weighted_log_density: torch.Tensor
    soft_estimate: torch.Tensor


def feedback_pice_objective(
    simulator: MarketSimulator,
    candidate: TwoDriverHestonControl,
    *,
    behavior_control: Callable[[float, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    | None,
    spot: float,
    variance: float,
    maturity: float,
    dt: float,
    barrier: float,
    temperature: float,
    num_paths: int,
) -> FeedbackPICEObjective:
    r"""Return a self-normalized feedback PICE training loss.

    Behavior paths are converted to target coordinates before the candidate
    density is evaluated.  Self-normalization is confined to this training
    projection; the returned soft estimate uses ordinary likelihood weights.
    """
    with torch.no_grad():
        paths = simulator.simulate_controlled_two_driver(
            S0=spot,
            v0=variance,
            T=maturity,
            dt=dt,
            num_paths=num_paths,
            control_fn=behavior_control,
            record_brownian=True,
            dtype=next(candidate.parameters()).dtype,
        )
        potential = terminal_left_tail_potential(paths.spot[:, -1], barrier, temperature)
        log_tilted_behavior_weight = -potential.double() + paths.log_likelihood
        normalized_weight = torch.softmax(log_tilted_behavior_weight, dim=0)
        effective_sample_size = torch.reciprocal(torch.sum(normalized_weight.square()))
        soft_estimate = torch.exp(
            torch.logsumexp(log_tilted_behavior_weight, dim=0) - math.log(num_paths)
        )
    candidate_log_density, _controls = candidate_log_density_on_target_paths(candidate, paths)
    normalized_for_loss = normalized_weight.to(dtype=candidate_log_density.dtype).detach()
    weighted_log_density = torch.sum(normalized_for_loss * candidate_log_density)
    return FeedbackPICEObjective(
        loss=-weighted_log_density,
        effective_sample_size=effective_sample_size.detach(),
        effective_sample_fraction=(effective_sample_size / num_paths).detach(),
        weighted_log_density=weighted_log_density.detach(),
        soft_estimate=soft_estimate.detach(),
    )


@dataclass(frozen=True)
class HardJ2Objective:
    loss: torch.Tensor
    log_second_moment: torch.Tensor
    estimate: torch.Tensor
    proposal_event_fraction: torch.Tensor
    contribution_ess: torch.Tensor


def hard_j2_objective(
    simulator: MarketSimulator,
    control: TwoDriverHestonControl,
    *,
    spot: float,
    variance: float,
    maturity: float,
    dt: float,
    barrier: float,
    num_paths: int,
) -> HardJ2Objective:
    r"""Hard-event ``log J2`` with the exact two-driver score gradient.

    For ``J2=E_Q[1_A L^2]`` the fixed-target-path derivative is
    ``-E_Q[1_A L^2 grad log q]``.  The hard indicator is never pathwise
    differentiated.
    """
    # The hard indicator admits no ordinary pathwise gradient.  Sampling is
    # therefore deliberately graph-free; the only gradient below is the
    # fixed-target-path score replay.
    with torch.no_grad():
        paths = simulator.simulate_controlled_two_driver(
            S0=spot,
            v0=variance,
            T=maturity,
            dt=dt,
            num_paths=num_paths,
            control_fn=control,
            record_brownian=True,
            dtype=next(control.parameters()).dtype,
        )
    proposal = paths.proposal_brownian_increments
    if proposal is None:
        raise RuntimeError("proposal Brownian increments were not recorded")
    event = (paths.spot[:, -1] <= barrier).detach()
    if not bool(event.any()):
        raise RuntimeError("J2 batch contains no hard events; strengthen the warm start")

    negative_infinity = torch.full_like(paths.log_likelihood, -torch.inf)
    log_terms = torch.where(event, 2.0 * paths.log_likelihood, negative_infinity)
    log_second_moment = torch.logsumexp(log_terms, dim=0) - math.log(num_paths)
    normalized_terms = torch.softmax(log_terms, dim=0).detach()

    detached_spot = paths.spot.detach()
    detached_variance = paths.variance.detach()
    score_terms: list[torch.Tensor] = []
    for step in range(proposal.shape[1]):
        replayed = control(
            step * paths.step_dt,
            detached_spot[:, step],
            detached_variance[:, step],
            _prefix_running_average(detached_spot, step),
        )
        score_terms.append(torch.sum(replayed * proposal[:, step, :].detach(), dim=-1))
    score_log_q = torch.stack(score_terms, dim=1).sum(dim=1)
    gradient_surrogate = -torch.sum(
        normalized_terms.to(dtype=score_log_q.dtype) * score_log_q
    )
    loss = gradient_surrogate - gradient_surrogate.detach() + log_second_moment.detach()

    contribution = event.double() * torch.exp(paths.log_likelihood)
    contribution_sum = contribution.sum()
    contribution_ess = contribution_sum.square() / contribution.square().sum().clamp_min(1e-300)
    return HardJ2Objective(
        loss=loss,
        log_second_moment=log_second_moment.detach(),
        estimate=contribution.mean().detach(),
        proposal_event_fraction=event.double().mean(),
        contribution_ess=contribution_ess.detach(),
    )


@dataclass(frozen=True)
class FeedbackTrainingRecord:
    stage: str
    update: int
    loss: float
    diagnostic: float


def train_soft_pi_stage(
    simulator: MarketSimulator,
    control: TwoDriverHestonControl,
    *,
    updates: int,
    learning_rate: float,
    seed: int,
    gradient_clip: float = 5.0,
    **objective_kwargs: float | int,
) -> tuple[FeedbackTrainingRecord, ...]:
    """Optimize a soft PI stage with deterministic, distinct RNG substreams."""
    return _train_feedback_stage(
        control,
        updates=updates,
        learning_rate=learning_rate,
        seed=seed,
        gradient_clip=gradient_clip,
        stage="pi",
        objective=lambda: soft_pi_objective(simulator, control, **objective_kwargs),
        diagnostic=lambda value: float(value.soft_estimate),
    )


def train_feedback_pice_stage(
    simulator: MarketSimulator,
    control: TwoDriverHestonControl,
    *,
    updates: int,
    learning_rate: float,
    seed: int,
    gradient_clip: float = 5.0,
    behavior_refresh: int = 10,
    **objective_kwargs: float | int,
) -> tuple[FeedbackTrainingRecord, ...]:
    """Optimize PICE against periodically frozen behavior policies."""
    if behavior_refresh <= 0:
        raise ValueError("behavior_refresh must be positive")
    behavior = control.frozen_copy()
    optimizer = torch.optim.Adam(control.parameters(), lr=learning_rate)
    records: list[FeedbackTrainingRecord] = []
    control.train()
    for update in range(1, updates + 1):
        if update > 1 and (update - 1) % behavior_refresh == 0:
            behavior = control.frozen_copy()
        torch.manual_seed(_substream_seed(seed, update))
        optimizer.zero_grad()
        result = feedback_pice_objective(
            simulator,
            control,
            behavior_control=behavior,
            **objective_kwargs,
        )
        result.loss.backward()
        torch.nn.utils.clip_grad_norm_(control.parameters(), gradient_clip)
        optimizer.step()
        records.append(
            FeedbackTrainingRecord(
                stage="pice",
                update=update,
                loss=float(result.loss.detach()),
                diagnostic=float(result.effective_sample_fraction),
            )
        )
    return tuple(records)


def train_hard_j2_stage(
    simulator: MarketSimulator,
    control: TwoDriverHestonControl,
    *,
    updates: int,
    learning_rate: float,
    seed: int,
    gradient_clip: float = 5.0,
    **objective_kwargs: float | int,
) -> tuple[FeedbackTrainingRecord, ...]:
    """Optimize the hard-event second moment after PI/PICE warm starts."""
    return _train_feedback_stage(
        control,
        updates=updates,
        learning_rate=learning_rate,
        seed=seed,
        gradient_clip=gradient_clip,
        stage="j2",
        objective=lambda: hard_j2_objective(simulator, control, **objective_kwargs),
        diagnostic=lambda value: float(value.proposal_event_fraction),
    )


def _substream_seed(root: int, update: int) -> int:
    return int((root + 1_000_003 * (update - 1)) % (2**63 - 1))


def _train_feedback_stage(
    control: TwoDriverHestonControl,
    *,
    updates: int,
    learning_rate: float,
    seed: int,
    gradient_clip: float,
    stage: str,
    objective: Callable[[], SoftPIObjective | HardJ2Objective],
    diagnostic: Callable[[SoftPIObjective | HardJ2Objective], float],
) -> tuple[FeedbackTrainingRecord, ...]:
    if updates <= 0 or not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("updates and learning_rate must be positive")
    if not math.isfinite(gradient_clip) or gradient_clip <= 0.0:
        raise ValueError("gradient_clip must be finite and positive")
    optimizer = torch.optim.Adam(control.parameters(), lr=learning_rate)
    records: list[FeedbackTrainingRecord] = []
    control.train()
    for update in range(1, updates + 1):
        torch.manual_seed(_substream_seed(seed, update))
        optimizer.zero_grad()
        result = objective()
        result.loss.backward()
        torch.nn.utils.clip_grad_norm_(control.parameters(), gradient_clip)
        optimizer.step()
        records.append(
            FeedbackTrainingRecord(
                stage=stage,
                update=update,
                loss=float(result.loss.detach()),
                diagnostic=diagnostic(result),
            )
        )
    return tuple(records)


def numpy_oracle_arrays(dataset: HestonOracleDataset) -> dict[str, np.ndarray]:
    """Return detached NumPy arrays for reproducible report serialization."""
    dataset.validate()
    return {
        "time": dataset.time.detach().cpu().numpy(),
        "spot": dataset.spot.detach().cpu().numpy(),
        "variance": dataset.variance.detach().cpu().numpy(),
        "control": dataset.control.detach().cpu().numpy(),
    }
