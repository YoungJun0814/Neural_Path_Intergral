"""Sequential PI/PICE/J2 training objectives for the stateful VFO controller."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

from src.path_integral.controllers import VFOBranchDiagnostics, VolterraFollmerOperator
from src.physics_engine import RBergomiSimulator, TwoDriverRBergomiPaths

VFOObjectiveName = Literal["pi", "pice", "j2"]
VFOEventType = Literal["terminal", "down_barrier"]


def _event_value(
    paths: TwoDriverRBergomiPaths,
    event_type: VFOEventType,
) -> torch.Tensor:
    if event_type == "terminal":
        return paths.spot[:, -1]
    if event_type == "down_barrier":
        return paths.running_minimum[:, -1]
    raise ValueError("event_type must be 'terminal' or 'down_barrier'")


def _soft_potential(
    terminal_spot: torch.Tensor,
    *,
    barrier: float,
    scale: float,
) -> torch.Tensor:
    if not math.isfinite(barrier) or barrier <= 0.0:
        raise ValueError("barrier must be positive")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be positive")
    return torch.nn.functional.softplus((terminal_spot - barrier) / scale)


@dataclass(frozen=True)
class VFOSoftPIObjective:
    loss: torch.Tensor
    soft_estimate: torch.Tensor
    potential_mean: torch.Tensor
    energy_mean: torch.Tensor
    branch: VFOBranchDiagnostics


def vfo_soft_pi_objective(
    simulator: RBergomiSimulator,
    control: VolterraFollmerOperator,
    *,
    spot: float,
    maturity: float,
    dt: float,
    num_paths: int,
    barrier: float,
    soft_scale: float,
    event_type: VFOEventType = "terminal",
) -> VFOSoftPIObjective:
    paths = simulator.simulate_controlled_two_driver(
        S0=spot,
        T=maturity,
        dt=dt,
        num_paths=num_paths,
        control_fn=control,
        record_augmented=False,
        dtype=next(control.parameters()).dtype,
    )
    potential = _soft_potential(
        _event_value(paths, event_type), barrier=barrier, scale=soft_scale
    )
    loss = potential.double().mean() + 0.5 * paths.control_energy.mean()
    log_contribution = -potential.double() + paths.log_likelihood
    soft_estimate = torch.exp(
        torch.logsumexp(log_contribution, dim=0) - math.log(num_paths)
    )
    return VFOSoftPIObjective(
        loss=loss,
        soft_estimate=soft_estimate.detach(),
        potential_mean=potential.detach().mean(),
        energy_mean=paths.control_energy.detach().mean(),
        branch=control.branch_diagnostics(),
    )


def replay_vfo_on_target_paths(
    control: VolterraFollmerOperator,
    paths: TwoDriverRBergomiPaths,
) -> torch.Tensor:
    """Causally replay a candidate VFO on a fixed canonical target path."""
    target = paths.target_brownian_increments
    if target is None:
        raise ValueError("target Brownian increments must be recorded")
    batch, steps, _drivers = target.shape
    dtype = next(control.parameters()).dtype
    device = next(control.parameters()).device
    if paths.spot.device != device:
        raise ValueError("candidate and recorded paths must share a device")
    control.reset_for_simulation(batch_size=batch, device=device, dtype=dtype)
    controls: list[torch.Tensor] = []
    for step in range(steps):
        controls.append(
            control(
                step * paths.step_dt,
                paths.spot[:, step].detach().to(dtype=dtype),
                paths.variance[:, step].detach().to(dtype=dtype),
                paths.volterra[:, step].detach().to(dtype=dtype),
                paths.running_minimum[:, step].detach().to(dtype=dtype),
            )
        )
        control.observe_target_increment(
            target[:, step, 0].detach().to(dtype=dtype), paths.step_dt
        )
    return torch.stack(controls, dim=1)


@dataclass(frozen=True)
class VFOPICEObjective:
    loss: torch.Tensor
    effective_sample_fraction: torch.Tensor
    soft_estimate: torch.Tensor
    branch: VFOBranchDiagnostics


def vfo_pice_objective(
    simulator: RBergomiSimulator,
    candidate: VolterraFollmerOperator,
    *,
    behavior: VolterraFollmerOperator,
    spot: float,
    maturity: float,
    dt: float,
    num_paths: int,
    barrier: float,
    soft_scale: float,
    event_type: VFOEventType = "terminal",
) -> VFOPICEObjective:
    with torch.no_grad():
        paths = simulator.simulate_controlled_two_driver(
            S0=spot,
            T=maturity,
            dt=dt,
            num_paths=num_paths,
            control_fn=behavior,
            record_augmented=True,
            dtype=next(candidate.parameters()).dtype,
        )
        potential = _soft_potential(
            _event_value(paths, event_type), barrier=barrier, scale=soft_scale
        )
        log_weight = -potential.double() + paths.log_likelihood
        normalized_weight = torch.softmax(log_weight, dim=0)
        ess = torch.reciprocal(torch.sum(normalized_weight.square()))
        soft_estimate = torch.exp(torch.logsumexp(log_weight, dim=0) - math.log(num_paths))
    candidate_controls = replay_vfo_on_target_paths(candidate, paths)
    assert paths.target_brownian_increments is not None
    target = paths.target_brownian_increments.detach().to(dtype=candidate_controls.dtype)
    log_density = torch.sum(candidate_controls * target, dim=(1, 2))
    log_density = log_density - 0.5 * paths.step_dt * torch.sum(
        candidate_controls.square(), dim=(1, 2)
    )
    loss = -torch.sum(normalized_weight.to(dtype=log_density.dtype).detach() * log_density)
    return VFOPICEObjective(
        loss=loss,
        effective_sample_fraction=(ess / num_paths).detach(),
        soft_estimate=soft_estimate.detach(),
        branch=candidate.branch_diagnostics(),
    )


@dataclass(frozen=True)
class VFOHardJ2Objective:
    loss: torch.Tensor
    log_second_moment: torch.Tensor
    estimate: torch.Tensor
    event_fraction: torch.Tensor
    contribution_ess_fraction: torch.Tensor
    branch: VFOBranchDiagnostics


def vfo_hard_j2_objective(
    simulator: RBergomiSimulator,
    control: VolterraFollmerOperator,
    *,
    spot: float,
    maturity: float,
    dt: float,
    num_paths: int,
    barrier: float,
    event_type: VFOEventType = "terminal",
) -> VFOHardJ2Objective:
    with torch.no_grad():
        paths = simulator.simulate_controlled_two_driver(
            S0=spot,
            T=maturity,
            dt=dt,
            num_paths=num_paths,
            control_fn=control,
            record_augmented=True,
            dtype=next(control.parameters()).dtype,
        )
    assert paths.proposal_brownian_increments is not None
    event = (_event_value(paths, event_type) <= barrier).detach()
    if not bool(event.any()):
        raise RuntimeError("VFO J2 batch contains no hard events")
    negative_infinity = torch.full_like(paths.log_likelihood, -torch.inf)
    log_terms = torch.where(event, 2.0 * paths.log_likelihood, negative_infinity)
    log_second_moment = torch.logsumexp(log_terms, dim=0) - math.log(num_paths)
    normalized = torch.softmax(log_terms, dim=0).detach()
    replayed = replay_vfo_on_target_paths(control, paths)
    proposal = paths.proposal_brownian_increments.detach().to(dtype=replayed.dtype)
    score_log_q = torch.sum(replayed * proposal, dim=(1, 2))
    surrogate = -torch.sum(normalized.to(dtype=score_log_q.dtype) * score_log_q)
    loss = surrogate - surrogate.detach() + log_second_moment.detach()
    contribution = event.double() * torch.exp(paths.log_likelihood)
    ess = contribution.sum().square() / contribution.square().sum().clamp_min(1e-300)
    return VFOHardJ2Objective(
        loss=loss,
        log_second_moment=log_second_moment.detach(),
        estimate=contribution.mean().detach(),
        event_fraction=event.double().mean(),
        contribution_ess_fraction=(ess / num_paths).detach(),
        branch=control.branch_diagnostics(),
    )


@dataclass(frozen=True)
class VFOTrainingRecord:
    update: int
    stage: str
    objective: str
    loss: float
    diagnostic: float
    structural_gate: float
    residual_gate: float
    residual_energy_fraction: float
    takeover_alarm: bool


def train_vfo_stage(
    simulator: RBergomiSimulator,
    control: VolterraFollmerOperator,
    *,
    stage: Literal["instant", "structural", "residual", "joint"],
    objective: VFOObjectiveName,
    updates: int,
    learning_rate: float,
    seed: int,
    gradient_clip: float = 5.0,
    behavior_refresh: int = 10,
    **objective_kwargs: float | int,
) -> tuple[VFOTrainingRecord, ...]:
    if updates <= 0 or learning_rate <= 0.0 or gradient_clip <= 0.0:
        raise ValueError("updates, learning_rate, and gradient_clip must be positive")
    control.set_stage(stage)
    parameters = [parameter for parameter in control.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("VFO stage has no trainable parameters")
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    behavior = control.frozen_copy()
    records: list[VFOTrainingRecord] = []
    for update in range(1, updates + 1):
        torch.manual_seed(int((seed + 1_000_003 * (update - 1)) % (2**63 - 1)))
        optimizer.zero_grad()
        if objective == "pi":
            result = vfo_soft_pi_objective(
                simulator, control, **objective_kwargs
            )
            diagnostic = float(result.soft_estimate)
        elif objective == "pice":
            if update > 1 and (update - 1) % behavior_refresh == 0:
                behavior = control.frozen_copy()
            result = vfo_pice_objective(
                simulator, control, behavior=behavior, **objective_kwargs
            )
            diagnostic = float(result.effective_sample_fraction)
        elif objective == "j2":
            result = vfo_hard_j2_objective(
                simulator, control, **objective_kwargs
            )
            diagnostic = float(result.event_fraction)
        else:
            raise ValueError("unknown VFO objective")
        result.loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, gradient_clip)
        optimizer.step()
        branch = result.branch
        takeover = branch.residual_energy_fraction > 0.90
        records.append(
            VFOTrainingRecord(
                update=update,
                stage=stage,
                objective=objective,
                loss=float(result.loss.detach()),
                diagnostic=diagnostic,
                structural_gate=branch.structural_gate,
                residual_gate=branch.residual_gate,
                residual_energy_fraction=branch.residual_energy_fraction,
                takeover_alarm=takeover,
            )
        )
    return tuple(records)
