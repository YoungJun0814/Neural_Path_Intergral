"""Exact all-expert replay for finite-grid rBergomi proposal mixtures."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, cast

import torch

from src.path_integral.mixture import (
    all_expert_log_q_over_p,
    log_mixture_q_over_p,
    sample_mixture_labels,
    selected_component_log_p_over_q,
)
from src.path_integral.rbergomi_fft import simulate_rbergomi_fft
from src.physics_engine import RBergomiSimulator, TwoDriverRBergomiPaths

RBergomiControl = Callable[
    [float, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
]
SimulationEngine = Literal["reference", "fft"]


@dataclass(frozen=True)
class RBergomiMixtureSample:
    """Paths and both valid multiple-importance-sampling likelihood schemes."""

    paths: TwoDriverRBergomiPaths
    labels: torch.Tensor
    weights: torch.Tensor
    all_expert_controls: torch.Tensor
    component_log_q_over_p: torch.Tensor
    log_mixture_q_over_p: torch.Tensor
    mixture_log_likelihood: torch.Tensor
    selected_component_log_likelihood: torch.Tensor
    maximum_selected_replay_error: float


def replay_rbergomi_control_on_target_paths(
    control: RBergomiControl,
    paths: TwoDriverRBergomiPaths,
    *,
    path_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Causally replay a stateless or stateful controller on canonical paths."""
    target = paths.target_brownian_increments
    if target is None:
        raise ValueError("target Brownian increments must be recorded")
    full_batch, steps, drivers = target.shape
    if drivers != 2:
        raise ValueError("rBergomi target paths must contain two Brownian drivers")
    if path_mask is None:
        path_mask = torch.ones(full_batch, device=target.device, dtype=torch.bool)
    if (
        path_mask.ndim != 1
        or path_mask.shape[0] != full_batch
        or path_mask.device != target.device
        or path_mask.dtype != torch.bool
    ):
        raise ValueError("path_mask must be a boolean tensor matching the path batch")
    batch = int(path_mask.sum())
    if batch == 0:
        return torch.empty(
            (0, steps, drivers), device=target.device, dtype=target.dtype
        )
    reset_memory = getattr(control, "reset_for_simulation", None)
    if callable(reset_memory):
        reset_memory(batch_size=batch, device=target.device, dtype=target.dtype)
    controls: list[torch.Tensor] = []
    for step in range(steps):
        if bool(getattr(control, "uses_running_minimum", False)):
            stateful_control = cast(Callable[..., torch.Tensor], control)
            value = stateful_control(
                step * paths.step_dt,
                paths.spot[path_mask, step],
                paths.variance[path_mask, step],
                paths.volterra[path_mask, step],
                paths.running_minimum[path_mask, step],
            )
        else:
            value = control(
                step * paths.step_dt,
                paths.spot[path_mask, step],
                paths.variance[path_mask, step],
                paths.volterra[path_mask, step],
            )
        if value.shape != (batch, 2):
            raise ValueError("replayed controller must return shape (batch, 2)")
        if value.device != target.device or value.dtype != target.dtype:
            raise ValueError("replayed control must match target increment device and dtype")
        if not torch.isfinite(value).all():
            raise ValueError("replayed control must be finite")
        controls.append(value)
        observe_increment = getattr(control, "observe_target_increment", None)
        if callable(observe_increment):
            observe_increment(target[path_mask, step, 0], paths.step_dt)
    return torch.stack(controls, dim=1)


def _concatenate_paths(parts: Sequence[TwoDriverRBergomiPaths]) -> TwoDriverRBergomiPaths:
    if not parts:
        raise ValueError("at least one nonempty component sample is required")
    step_dt = parts[0].step_dt
    if any(abs(part.step_dt - step_dt) > 1e-15 for part in parts[1:]):
        raise ValueError("component samples must share the same time grid")

    def required(field: str) -> torch.Tensor:
        values = [getattr(part, field) for part in parts]
        if any(value is None for value in values):
            raise ValueError(f"component sample is missing required field {field}")
        return torch.cat(values, dim=0)  # type: ignore[arg-type]

    return TwoDriverRBergomiPaths(
        spot=required("spot"),
        variance=required("variance"),
        volterra=required("volterra"),
        running_minimum=required("running_minimum"),
        log_likelihood=required("log_likelihood"),
        control_energy=required("control_energy"),
        step_dt=step_dt,
        proposal_brownian_increments=required("proposal_brownian_increments"),
        target_brownian_increments=required("target_brownian_increments"),
        proposal_local_integrals=required("proposal_local_integrals"),
        target_local_integrals=required("target_local_integrals"),
        controls=required("controls"),
    )


def simulate_rbergomi_mixture(
    simulator: RBergomiSimulator,
    controls: Sequence[RBergomiControl],
    weights: torch.Tensor,
    *,
    spot: float,
    maturity: float,
    dt: float,
    num_paths: int,
    dtype: torch.dtype = torch.float64,
    label_generator: torch.Generator | None = None,
    engine: SimulationEngine = "reference",
) -> RBergomiMixtureSample:
    """Sample a randomized expert mixture and evaluate its exact marginal density."""
    if not controls:
        raise ValueError("at least one expert control is required")
    if engine not in ("reference", "fft"):
        raise ValueError("engine must be 'reference' or 'fft'")
    labels_draw = sample_mixture_labels(
        weights, num_paths, generator=label_generator
    )
    parts: list[TwoDriverRBergomiPaths] = []
    grouped_labels: list[torch.Tensor] = []
    for expert_index, control in enumerate(controls):
        count = int(torch.sum(labels_draw == expert_index))
        if count == 0:
            continue
        if engine == "reference":
            part = simulator.simulate_controlled_two_driver(
                S0=spot,
                T=maturity,
                dt=dt,
                num_paths=count,
                control_fn=control,
                record_augmented=True,
                dtype=dtype,
            )
        else:
            part = simulate_rbergomi_fft(
                simulator,
                S0=spot,
                T=maturity,
                dt=dt,
                num_paths=count,
                control_fn=control,
                dtype=dtype,
            )
        parts.append(part)
        grouped_labels.append(
            torch.full(
                (count,),
                expert_index,
                device=simulator.device,
                dtype=torch.long,
            )
        )
    paths = _concatenate_paths(parts)
    labels = torch.cat(grouped_labels, dim=0)
    if paths.controls is None:
        raise ValueError("selected component controls must be recorded")
    batch, steps, drivers = paths.controls.shape
    replayed = torch.empty(
        (batch, len(controls), steps, drivers),
        device=paths.controls.device,
        dtype=paths.controls.dtype,
    )
    for expert_index, control in enumerate(controls):
        deterministic = bool(getattr(control, "is_deterministic_time_control", False))
        evaluator = getattr(control, "deterministic_schedule", None)
        if deterministic and callable(evaluator):
            times = torch.arange(
                steps,
                device=paths.controls.device,
                dtype=paths.controls.dtype,
            ) * paths.step_dt
            schedule = cast(torch.Tensor, evaluator(times))
            if (
                schedule.shape != (steps, drivers)
                or schedule.device != paths.controls.device
                or schedule.dtype != paths.controls.dtype
                or not torch.isfinite(schedule).all()
            ):
                raise ValueError("deterministic expert returned an invalid schedule")
            replayed[:, expert_index] = schedule.unsqueeze(0)
            continue
        selected = labels == expert_index
        replayed[selected, expert_index] = paths.controls[selected]
        unselected = ~selected
        replayed[unselected, expert_index] = replay_rbergomi_control_on_target_paths(
            control, paths, path_mask=unselected
        )
    assert paths.target_brownian_increments is not None
    component_log = all_expert_log_q_over_p(
        replayed, paths.target_brownian_increments, paths.step_dt
    )
    resolved_weights = weights.to(device=component_log.device, dtype=component_log.dtype)
    mixture_log = log_mixture_q_over_p(component_log, resolved_weights)
    selected_log_likelihood = selected_component_log_p_over_q(component_log, labels)
    replay_error = torch.max(torch.abs(selected_log_likelihood - paths.log_likelihood))
    return RBergomiMixtureSample(
        paths=paths,
        labels=labels,
        weights=resolved_weights,
        all_expert_controls=replayed,
        component_log_q_over_p=component_log,
        log_mixture_q_over_p=mixture_log,
        mixture_log_likelihood=-mixture_log,
        selected_component_log_likelihood=selected_log_likelihood,
        maximum_selected_replay_error=float(replay_error.detach()),
    )
