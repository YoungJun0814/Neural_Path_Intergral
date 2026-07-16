"""Conditional desirability and Brownian-moment regression for SDV."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from src.path_integral import (
    ConstantTwoDriverControl,
    DownsideExcursionTask,
    SpectralDoobVolterraControl,
    TimePiecewiseTwoDriverControl,
    simulate_rbergomi_mixture,
)
from src.physics_engine import RBergomiSimulator, TwoDriverRBergomiPaths


@dataclass(frozen=True)
class SDVRegressionObjective:
    loss: torch.Tensor
    desirability_loss: torch.Tensor
    moment_loss: torch.Tensor
    anchor_loss: torch.Tensor
    behavior_ess_fraction: torch.Tensor
    soft_target_mean: torch.Tensor
    predicted_desirability_mean: torch.Tensor
    maximum_selected_replay_error: float


@dataclass(frozen=True)
class SDVTrainingRecord:
    update: int
    loss: float
    desirability_loss: float
    moment_loss: float
    anchor_loss: float
    behavior_ess_fraction: float
    soft_target_mean: float
    predicted_desirability_mean: float
    gradient_norm: float
    maximum_selected_replay_error: float


def replay_sdv_outputs_on_target_paths(
    control: SpectralDoobVolterraControl,
    paths: TwoDriverRBergomiPaths,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replay pre-increment desirability and control on a canonical target path."""
    target = paths.target_brownian_increments
    if target is None:
        raise ValueError("target Brownian increments must be recorded")
    batch, steps, drivers = target.shape
    if drivers != 2:
        raise ValueError("SDV requires two target Brownian drivers")
    reference = next(control.parameters())
    if paths.spot.device != reference.device:
        raise ValueError("SDV and target paths must share a device")
    control.reset_for_simulation(
        batch_size=batch, device=reference.device, dtype=reference.dtype
    )
    controls: list[torch.Tensor] = []
    desirabilities: list[torch.Tensor] = []
    for step in range(steps):
        value = control(
            step * paths.step_dt,
            paths.spot[:, step].detach().to(dtype=reference.dtype),
            paths.variance[:, step].detach().to(dtype=reference.dtype),
            paths.volterra[:, step].detach().to(dtype=reference.dtype),
            paths.running_minimum[:, step].detach().to(dtype=reference.dtype),
        )
        controls.append(value)
        desirabilities.append(control.last_desirability)
        control.observe_target_increment(
            target[:, step, 0].detach().to(dtype=reference.dtype), paths.step_dt
        )
    return torch.stack(desirabilities, dim=1), torch.stack(controls, dim=1)


def sdv_regression_objective(
    simulator: RBergomiSimulator,
    control: SpectralDoobVolterraControl,
    task: DownsideExcursionTask,
    *,
    spot: float,
    maturity: float,
    dt: float,
    num_paths: int,
    natural_behavior_mass: float = 0.20,
    moment_loss_weight: float = 1.0,
    anchor_loss_weight: float = 0.01,
    label_seed: int = 0,
) -> SDVRegressionObjective:
    """Estimate the finite-grid conditional-moment projection under exact weights.

    Normalized weights are used only to estimate a training regression risk.
    Final probability estimators never self-normalize.
    """
    if not 0.0 < natural_behavior_mass < 1.0:
        raise ValueError("natural_behavior_mass must lie in (0, 1)")
    if num_paths <= 1 or moment_loss_weight < 0.0 or anchor_loss_weight < 0.0:
        raise ValueError("invalid SDV batch size or loss weights")
    reference = next(control.parameters())
    natural = ConstantTwoDriverControl(0.0, 0.0)
    anchor = TimePiecewiseTwoDriverControl(
        tuple((float(pair[0]), float(pair[1])) for pair in control.anchor_values),
        maturity=maturity,
    )
    behavior_weights = torch.tensor(
        (natural_behavior_mass, 1.0 - natural_behavior_mass), dtype=reference.dtype
    )
    with torch.no_grad():
        paths = simulate_rbergomi_mixture(
            simulator,
            (natural, anchor),
            behavior_weights,
            spot=spot,
            maturity=maturity,
            dt=dt,
            num_paths=num_paths,
            dtype=reference.dtype,
            label_generator=torch.Generator().manual_seed(label_seed),
        )
        soft_target = task.soft_payoff(paths.paths.spot, paths.paths.step_dt).to(
            dtype=reference.dtype
        )
        normalized_weight = torch.softmax(paths.mixture_log_likelihood, dim=0).to(
            dtype=reference.dtype
        )
        behavior_ess = torch.reciprocal(torch.sum(normalized_weight.square()))
    desirability, candidate_control = replay_sdv_outputs_on_target_paths(
        control, paths.paths
    )
    assert paths.paths.target_brownian_increments is not None
    target_increment = paths.paths.target_brownian_increments.detach().to(
        dtype=reference.dtype
    )
    sqrt_dt = math.sqrt(paths.paths.step_dt)
    target_h = soft_target[:, None].expand_as(desirability)
    h_per_path = torch.mean((desirability - target_h).square(), dim=1)
    desirability_loss = torch.sum(normalized_weight.detach() * h_per_path)

    predicted_moment = sqrt_dt * desirability[:, :, None] * candidate_control
    target_moment = soft_target[:, None, None] * target_increment / sqrt_dt
    moment_per_path = torch.mean((predicted_moment - target_moment).square(), dim=(1, 2))
    moment_loss = torch.sum(normalized_weight.detach() * moment_per_path)

    times = (
        torch.arange(candidate_control.shape[1], device=reference.device, dtype=reference.dtype)
        * paths.paths.step_dt
    )
    anchor_by_step = control.anchor_at(times).unsqueeze(0)
    anchor_per_path = torch.mean(
        (candidate_control - anchor_by_step).square(), dim=(1, 2)
    )
    anchor_loss = torch.mean((1.0 - soft_target).detach() * anchor_per_path)
    loss = (
        desirability_loss
        + moment_loss_weight * moment_loss
        + anchor_loss_weight * anchor_loss
    )
    return SDVRegressionObjective(
        loss=loss,
        desirability_loss=desirability_loss.detach(),
        moment_loss=moment_loss.detach(),
        anchor_loss=anchor_loss.detach(),
        behavior_ess_fraction=(behavior_ess / num_paths).detach(),
        soft_target_mean=torch.sum(normalized_weight * soft_target).detach(),
        predicted_desirability_mean=torch.sum(
            normalized_weight[:, None] * desirability.detach()
        )
        / desirability.shape[1],
        maximum_selected_replay_error=paths.maximum_selected_replay_error,
    )


def train_sdv_regression(
    simulator: RBergomiSimulator,
    control: SpectralDoobVolterraControl,
    task: DownsideExcursionTask,
    *,
    updates: int,
    learning_rate: float,
    seed: int,
    gradient_clip: float = 5.0,
    **objective_kwargs: Any,
) -> tuple[SDVTrainingRecord, ...]:
    """Train SDV without differentiating through the rBergomi simulator."""
    if updates <= 0 or learning_rate <= 0.0 or gradient_clip <= 0.0:
        raise ValueError("SDV optimization settings must be positive")
    parameters = [parameter for parameter in control.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    records: list[SDVTrainingRecord] = []
    for update in range(1, updates + 1):
        update_seed = int((seed + 1_000_003 * (update - 1)) % (2**63 - 1))
        torch.manual_seed(update_seed)
        optimizer.zero_grad()
        result = sdv_regression_objective(
            simulator,
            control,
            task,
            label_seed=update_seed + 17,
            **objective_kwargs,
        )
        result.loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip)
        optimizer.step()
        records.append(
            SDVTrainingRecord(
                update=update,
                loss=float(result.loss.detach()),
                desirability_loss=float(result.desirability_loss),
                moment_loss=float(result.moment_loss),
                anchor_loss=float(result.anchor_loss),
                behavior_ess_fraction=float(result.behavior_ess_fraction),
                soft_target_mean=float(result.soft_target_mean),
                predicted_desirability_mean=float(result.predicted_desirability_mean),
                gradient_norm=float(gradient_norm),
                maximum_selected_replay_error=result.maximum_selected_replay_error,
            )
        )
    return tuple(records)
