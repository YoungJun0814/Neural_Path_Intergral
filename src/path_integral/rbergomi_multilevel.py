"""Exact defensive mixtures for adjacent-grid rBergomi corrections."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from src.path_integral.mixture import (
    all_expert_log_q_over_p,
    log_mixture_q_over_p,
    sample_mixture_labels,
    selected_component_log_p_over_q,
)
from src.path_integral.rbergomi_coupling import (
    CoupledRBergomiPaths,
    RBergomiControl,
    RBergomiLevelPaths,
    simulate_coupled_rbergomi_adjacent,
)
from src.path_integral.rbergomi_mixture import replay_rbergomi_control_on_target_paths
from src.physics_engine import RBergomiSimulator, TwoDriverRBergomiPaths


@dataclass(frozen=True)
class CoupledRBergomiMixtureSample:
    """A coupled sample with exact balance-mixture likelihoods."""

    paths: CoupledRBergomiPaths
    labels: torch.Tensor
    weights: torch.Tensor
    all_expert_controls: torch.Tensor
    component_log_q_over_p: torch.Tensor
    log_mixture_q_over_p: torch.Tensor
    mixture_log_likelihood: torch.Tensor
    selected_component_log_likelihood: torch.Tensor
    maximum_selected_replay_error: float


def _required_cat(parts: Sequence[object], field: str) -> torch.Tensor:
    values = [getattr(part, field) for part in parts]
    if any(value is None for value in values):
        raise ValueError(f"component sample is missing required field {field}")
    return torch.cat(values, dim=0)  # type: ignore[arg-type]


def _concatenate_levels(parts: Sequence[RBergomiLevelPaths]) -> RBergomiLevelPaths:
    if not parts:
        raise ValueError("at least one level sample is required")
    step_dt = parts[0].step_dt
    if any(abs(part.step_dt - step_dt) > 1e-15 for part in parts[1:]):
        raise ValueError("level samples must share a time grid")
    return RBergomiLevelPaths(
        spot=_required_cat(parts, "spot"),
        variance=_required_cat(parts, "variance"),
        volterra=_required_cat(parts, "volterra"),
        running_minimum=_required_cat(parts, "running_minimum"),
        step_dt=step_dt,
        target_brownian_increments=_required_cat(parts, "target_brownian_increments"),
        target_local_integrals=_required_cat(parts, "target_local_integrals"),
    )


def _concatenate_coupled(
    parts: Sequence[CoupledRBergomiPaths],
) -> CoupledRBergomiPaths:
    if not parts:
        raise ValueError("at least one coupled component sample is required")
    return CoupledRBergomiPaths(
        fine=_concatenate_levels([part.fine for part in parts]),
        coarse=_concatenate_levels([part.coarse for part in parts]),
        log_likelihood=_required_cat(parts, "log_likelihood"),
        control_energy=_required_cat(parts, "control_energy"),
        proposal_fine_brownian_increments=_required_cat(
            parts, "proposal_fine_brownian_increments"
        ),
        target_fine_brownian_increments=_required_cat(
            parts, "target_fine_brownian_increments"
        ),
        proposal_fine_local_integrals=_required_cat(
            parts, "proposal_fine_local_integrals"
        ),
        target_fine_local_integrals=_required_cat(parts, "target_fine_local_integrals"),
        proposal_coarse_local_integrals=_required_cat(
            parts, "proposal_coarse_local_integrals"
        ),
        target_coarse_local_integrals=_required_cat(
            parts, "target_coarse_local_integrals"
        ),
        fine_controls=_required_cat(parts, "fine_controls"),
    )


def _as_replay_paths(paths: CoupledRBergomiPaths) -> TwoDriverRBergomiPaths:
    """Adapt the fine marginal to the existing causal replay implementation."""
    target = paths.target_fine_brownian_increments
    if target is None:
        raise ValueError("target fine Brownian increments must be recorded")
    return TwoDriverRBergomiPaths(
        spot=paths.fine.spot,
        variance=paths.fine.variance,
        volterra=paths.fine.volterra,
        running_minimum=paths.fine.running_minimum,
        log_likelihood=paths.log_likelihood,
        control_energy=paths.control_energy,
        step_dt=paths.fine.step_dt,
        proposal_brownian_increments=paths.proposal_fine_brownian_increments,
        target_brownian_increments=target,
        proposal_local_integrals=paths.proposal_fine_local_integrals,
        target_local_integrals=paths.target_fine_local_integrals,
        controls=paths.fine_controls,
    )


def simulate_coupled_rbergomi_mixture(
    simulator: RBergomiSimulator,
    controls: Sequence[RBergomiControl],
    weights: torch.Tensor,
    *,
    spot: float,
    maturity: float,
    fine_steps: int,
    num_paths: int,
    dtype: torch.dtype = torch.float64,
    label_generator: torch.Generator | None = None,
) -> CoupledRBergomiMixtureSample:
    """Sample a mixture and evaluate every expert on one fine target path.

    The signed multilevel payoff must be multiplied by
    ``exp(mixture_log_likelihood)``.  Component likelihoods are returned only
    for diagnostics; they must not be assigned separately to fine and coarse
    terms.
    """
    if not controls:
        raise ValueError("at least one expert control is required")
    labels_draw = sample_mixture_labels(weights, num_paths, generator=label_generator)
    parts: list[CoupledRBergomiPaths] = []
    grouped_labels: list[torch.Tensor] = []
    for expert_index, control in enumerate(controls):
        count = int(torch.sum(labels_draw == expert_index))
        if count == 0:
            continue
        parts.append(
            simulate_coupled_rbergomi_adjacent(
                simulator,
                S0=spot,
                T=maturity,
                fine_steps=fine_steps,
                num_paths=count,
                control_fn=control,
                record_augmented=True,
                dtype=dtype,
            )
        )
        grouped_labels.append(
            torch.full(
                (count,), expert_index, device=simulator.device, dtype=torch.long
            )
        )
    paths = _concatenate_coupled(parts)
    labels = torch.cat(grouped_labels, dim=0)
    selected_controls = paths.fine_controls
    if selected_controls is None:
        raise ValueError("selected component controls must be recorded")
    batch, steps, drivers = selected_controls.shape
    replayed = torch.empty(
        (batch, len(controls), steps, drivers),
        device=selected_controls.device,
        dtype=selected_controls.dtype,
    )
    replay_paths = _as_replay_paths(paths)
    for expert_index, control in enumerate(controls):
        selected = labels == expert_index
        replayed[selected, expert_index] = selected_controls[selected]
        unselected = ~selected
        replayed[unselected, expert_index] = replay_rbergomi_control_on_target_paths(
            control, replay_paths, path_mask=unselected
        )
    target = paths.target_fine_brownian_increments
    if target is None:
        raise ValueError("target fine Brownian increments must be recorded")
    component_log = all_expert_log_q_over_p(replayed, target, paths.fine.step_dt)
    resolved_weights = weights.to(device=component_log.device, dtype=component_log.dtype)
    mixture_log = log_mixture_q_over_p(component_log, resolved_weights)
    selected_log_likelihood = selected_component_log_p_over_q(component_log, labels)
    replay_error = torch.max(
        torch.abs(selected_log_likelihood - paths.log_likelihood)
    )
    return CoupledRBergomiMixtureSample(
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
