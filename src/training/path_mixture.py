"""Mode-specialized PI/PICE and exact mixture-weight J2 objectives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

from src.path_integral.controllers import LeanRBergomiControl
from src.path_integral.mixture import all_expert_log_q_over_p, log_mixture_q_over_p
from src.path_integral.rbergomi_mixture import (
    RBergomiMixtureSample,
    replay_rbergomi_control_on_target_paths,
    simulate_rbergomi_mixture,
)
from src.physics_engine import RBergomiSimulator

MixtureMode = Literal["left", "right", "union"]


def terminal_mode_potential(
    terminal_spot: torch.Tensor,
    *,
    lower_threshold: float,
    upper_threshold: float,
    scale: float,
    mode: MixtureMode,
) -> torch.Tensor:
    """Return ``-log(G_soft)`` for a left, right, or two-tail event."""
    values = (lower_threshold, upper_threshold, scale)
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("thresholds and scale must be finite and positive")
    if lower_threshold >= upper_threshold:
        raise ValueError("lower_threshold must be smaller than upper_threshold")
    left = torch.nn.functional.softplus((terminal_spot - lower_threshold) / scale)
    right = torch.nn.functional.softplus((upper_threshold - terminal_spot) / scale)
    if mode == "left":
        return left
    if mode == "right":
        return right
    if mode != "union":
        raise ValueError("mode must be 'left', 'right', or 'union'")
    left_payoff = torch.exp(-left)
    right_payoff = torch.exp(-right)
    union_payoff = 1.0 - (1.0 - left_payoff) * (1.0 - right_payoff)
    return -torch.log(union_payoff.clamp_min(torch.finfo(terminal_spot.dtype).tiny))


@dataclass(frozen=True)
class LeanPIObjective:
    loss: torch.Tensor
    soft_estimate: torch.Tensor
    potential_mean: torch.Tensor
    energy_mean: torch.Tensor


def lean_soft_pi_objective(
    simulator: RBergomiSimulator,
    control: LeanRBergomiControl,
    *,
    spot: float,
    maturity: float,
    dt: float,
    num_paths: int,
    lower_threshold: float,
    upper_threshold: float,
    soft_scale: float,
    mode: MixtureMode,
) -> LeanPIObjective:
    paths = simulator.simulate_controlled_two_driver(
        S0=spot,
        T=maturity,
        dt=dt,
        num_paths=num_paths,
        control_fn=control,
        record_augmented=False,
        dtype=next(control.parameters()).dtype,
    )
    potential = terminal_mode_potential(
        paths.spot[:, -1],
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
        scale=soft_scale,
        mode=mode,
    )
    loss = potential.double().mean() + 0.5 * paths.control_energy.mean()
    log_contribution = -potential.double() + paths.log_likelihood
    soft_estimate = torch.exp(torch.logsumexp(log_contribution, dim=0) - math.log(num_paths))
    return LeanPIObjective(
        loss=loss,
        soft_estimate=soft_estimate.detach(),
        potential_mean=potential.detach().mean(),
        energy_mean=paths.control_energy.detach().mean(),
    )


@dataclass(frozen=True)
class LeanPICEObjective:
    loss: torch.Tensor
    effective_sample_fraction: torch.Tensor
    soft_estimate: torch.Tensor


def lean_pice_objective(
    simulator: RBergomiSimulator,
    candidate: LeanRBergomiControl,
    *,
    behavior: LeanRBergomiControl,
    spot: float,
    maturity: float,
    dt: float,
    num_paths: int,
    lower_threshold: float,
    upper_threshold: float,
    soft_scale: float,
    mode: MixtureMode,
) -> LeanPICEObjective:
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
        potential = terminal_mode_potential(
            paths.spot[:, -1],
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
            scale=soft_scale,
            mode=mode,
        )
        log_weight = -potential.double() + paths.log_likelihood
        normalized = torch.softmax(log_weight, dim=0)
        ess = torch.reciprocal(torch.sum(normalized.square()))
        soft_estimate = torch.exp(torch.logsumexp(log_weight, dim=0) - math.log(num_paths))
    controls = replay_rbergomi_control_on_target_paths(candidate, paths)
    assert paths.target_brownian_increments is not None
    component = all_expert_log_q_over_p(
        controls[:, None], paths.target_brownian_increments, paths.step_dt
    )[:, 0]
    loss = -torch.sum(normalized.to(component.dtype).detach() * component)
    return LeanPICEObjective(
        loss=loss,
        effective_sample_fraction=(ess / num_paths).detach(),
        soft_estimate=soft_estimate.detach(),
    )


def mixture_weights_from_logits(
    logits: torch.Tensor,
    *,
    minimum_weight: float,
) -> torch.Tensor:
    """Map logits to simplex weights with a strict per-component floor."""
    if logits.ndim != 1 or logits.numel() < 1:
        raise ValueError("logits must be a nonempty one-dimensional tensor")
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("logits must be finite and floating point")
    components = logits.numel()
    if (
        not math.isfinite(minimum_weight)
        or minimum_weight < 0.0
        or minimum_weight * components >= 1.0
    ):
        raise ValueError("minimum_weight must be nonnegative with K * floor < 1")
    free_mass = 1.0 - components * minimum_weight
    return minimum_weight + free_mass * torch.softmax(logits, dim=0)


@dataclass(frozen=True)
class MixtureWeightJ2Objective:
    loss: torch.Tensor
    log_second_moment: torch.Tensor
    weights: torch.Tensor
    event_fraction: torch.Tensor


def mixture_weight_j2_objective(
    sample: RBergomiMixtureSample,
    logits: torch.Tensor,
    *,
    lower_threshold: float,
    upper_threshold: float,
    minimum_weight: float,
) -> MixtureWeightJ2Objective:
    if lower_threshold >= upper_threshold:
        raise ValueError("lower_threshold must be smaller than upper_threshold")
    weights = mixture_weights_from_logits(logits, minimum_weight=minimum_weight)
    candidate_log = log_mixture_q_over_p(sample.component_log_q_over_p.detach(), weights)
    event = (sample.paths.spot[:, -1] <= lower_threshold) | (
        sample.paths.spot[:, -1] >= upper_threshold
    )
    if not bool(event.any()):
        raise RuntimeError("mixture J2 batch contains no hard events")
    negative_infinity = torch.full_like(candidate_log, -torch.inf)
    log_terms = torch.where(
        event,
        -sample.log_mixture_q_over_p.detach() - candidate_log,
        negative_infinity,
    )
    log_second_moment = torch.logsumexp(log_terms, dim=0) - math.log(event.numel())
    return MixtureWeightJ2Objective(
        loss=log_second_moment,
        log_second_moment=log_second_moment.detach(),
        weights=weights.detach(),
        event_fraction=event.double().mean().detach(),
    )


@dataclass(frozen=True)
class LeanTrainingRecord:
    update: int
    objective: str
    loss: float
    diagnostic: float


@dataclass(frozen=True)
class MixtureWeightTrainingRecord:
    update: int
    log_second_moment: float
    event_fraction: float
    weights: tuple[float, ...]


def train_lean_pi_pice(
    simulator: RBergomiSimulator,
    control: LeanRBergomiControl,
    *,
    spot: float,
    maturity: float,
    dt: float,
    num_paths: int,
    lower_threshold: float,
    upper_threshold: float,
    soft_scale: float,
    mode: MixtureMode,
    pi_updates: int,
    pice_updates: int,
    pi_learning_rate: float,
    pice_learning_rate: float,
    gradient_clip: float,
    seed: int,
    behavior_refresh: int = 5,
) -> list[LeanTrainingRecord]:
    if pi_updates < 0 or pice_updates < 0 or pi_updates + pice_updates == 0:
        raise ValueError("at least one nonnegative PI/PICE update is required")
    if behavior_refresh <= 0:
        raise ValueError("behavior_refresh must be positive")
    torch.manual_seed(seed)
    records: list[LeanTrainingRecord] = []
    optimizer = torch.optim.Adam(control.parameters(), lr=pi_learning_rate)
    for update in range(1, pi_updates + 1):
        optimizer.zero_grad(set_to_none=True)
        pi_result = lean_soft_pi_objective(
            simulator,
            control,
            spot=spot,
            maturity=maturity,
            dt=dt,
            num_paths=num_paths,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
            soft_scale=soft_scale,
            mode=mode,
        )
        pi_result.loss.backward()
        torch.nn.utils.clip_grad_norm_(control.parameters(), gradient_clip)
        optimizer.step()
        records.append(
            LeanTrainingRecord(
                update=update,
                objective="pi",
                loss=float(pi_result.loss.detach()),
                diagnostic=float(pi_result.soft_estimate),
            )
        )
    behavior = control.frozen_copy()
    optimizer = torch.optim.Adam(control.parameters(), lr=pice_learning_rate)
    for update in range(1, pice_updates + 1):
        if update > 1 and (update - 1) % behavior_refresh == 0:
            behavior = control.frozen_copy()
        optimizer.zero_grad(set_to_none=True)
        pice_result = lean_pice_objective(
            simulator,
            control,
            behavior=behavior,
            spot=spot,
            maturity=maturity,
            dt=dt,
            num_paths=num_paths,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
            soft_scale=soft_scale,
            mode=mode,
        )
        pice_result.loss.backward()
        torch.nn.utils.clip_grad_norm_(control.parameters(), gradient_clip)
        optimizer.step()
        records.append(
            LeanTrainingRecord(
                update=update,
                objective="pice",
                loss=float(pice_result.loss.detach()),
                diagnostic=float(pice_result.effective_sample_fraction),
            )
        )
    return records


def train_mixture_weight_j2(
    simulator: RBergomiSimulator,
    controls: list[LeanRBergomiControl],
    behavior_weights: torch.Tensor,
    *,
    spot: float,
    maturity: float,
    dt: float,
    num_paths: int,
    lower_threshold: float,
    upper_threshold: float,
    minimum_weight: float,
    updates: int,
    learning_rate: float,
    gradient_clip: float,
    seed: int,
) -> tuple[torch.Tensor, list[MixtureWeightTrainingRecord]]:
    """Optimize only mixture logits with experts and behavior mixture frozen."""
    if len(controls) != behavior_weights.numel():
        raise ValueError("controls and behavior_weights must have the same length")
    if updates <= 0 or num_paths <= 0:
        raise ValueError("updates and num_paths must be positive")
    if learning_rate <= 0.0 or gradient_clip <= 0.0:
        raise ValueError("learning_rate and gradient_clip must be positive")
    frozen = [control.frozen_copy() for control in controls]
    reference = next(controls[0].parameters())
    weights = behavior_weights.to(device=reference.device, dtype=reference.dtype)
    logits = torch.nn.Parameter(torch.log(weights))
    optimizer = torch.optim.Adam([logits], lr=learning_rate)
    records: list[MixtureWeightTrainingRecord] = []
    torch.manual_seed(seed)
    for update in range(1, updates + 1):
        with torch.no_grad():
            sample = simulate_rbergomi_mixture(
                simulator,
                frozen,
                weights,
                spot=spot,
                maturity=maturity,
                dt=dt,
                num_paths=num_paths,
                dtype=reference.dtype,
                label_generator=torch.Generator(device="cpu").manual_seed(seed + 100_000 + update),
            )
        optimizer.zero_grad(set_to_none=True)
        objective = mixture_weight_j2_objective(
            sample,
            logits,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
            minimum_weight=minimum_weight,
        )
        objective.loss.backward()
        torch.nn.utils.clip_grad_norm_([logits], gradient_clip)
        optimizer.step()
        current = mixture_weights_from_logits(logits.detach(), minimum_weight=minimum_weight)
        records.append(
            MixtureWeightTrainingRecord(
                update=update,
                log_second_moment=float(objective.log_second_moment),
                event_fraction=float(objective.event_fraction),
                weights=tuple(float(value) for value in current),
            )
        )
    return (
        mixture_weights_from_logits(logits.detach(), minimum_weight=minimum_weight),
        records,
    )
