"""Cross-entropy fitting for adjacent-grid rBergomi disagreement events."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from src.path_integral import DownsideExcursionTask, TimePiecewiseTwoDriverControl
from src.path_integral.rbergomi_coupling import simulate_coupled_rbergomi_adjacent
from src.physics_engine import RBergomiSimulator
from src.training.rbergomi_piecewise_cem import (
    PiecewiseValues,
    _as_values,
    _segment_sufficient_statistics,
)


@dataclass(frozen=True)
class CorrectionCEMIteration:
    """One likelihood-corrected MLE update for |H_f-H_c|."""

    iteration: int
    control_before: PiecewiseValues
    control_candidate: PiecewiseValues
    control_after: PiecewiseValues
    proposal_disagreement_fraction: float
    target_disagreement_probability: float
    signed_correction_estimate: float
    disagreement_weight_ess: float
    weighted_second_moment: float
    maximum_parameter_change: float


@dataclass(frozen=True)
class CorrectionCEMResult:
    control: PiecewiseValues
    converged: bool
    history: tuple[CorrectionCEMIteration, ...]


def fit_rbergomi_correction_cem(
    simulator: RBergomiSimulator,
    task: DownsideExcursionTask,
    *,
    spot: float,
    maturity: float,
    fine_steps: int,
    initial_control: PiecewiseValues,
    num_paths: int,
    seed: int,
    max_iterations: int = 8,
    smoothing: float = 0.60,
    min_disagreement_paths: int = 64,
    control_bound: float = 8.0,
    parameter_tolerance: float = 0.05,
    stable_repetitions: int = 2,
) -> CorrectionCEMResult:
    r"""Fit a fine-grid drift to the adjacent-level disagreement measure.

    At iteration ``k`` the sufficient statistics are weighted by
    ``|H_f-H_c| dP/dQ_k``.  This is the likelihood-corrected Gaussian MLE for
    the normalized disagreement measure.  The signed correction is used only
    for estimation diagnostics, never as an MLE weight.
    """
    if not initial_control or any(len(value) != 2 for value in initial_control):
        raise ValueError("initial_control must contain two-driver segments")
    if not all(math.isfinite(entry) for value in initial_control for entry in value):
        raise ValueError("initial controls must be finite")
    if fine_steps < 2 or fine_steps % 2 != 0:
        raise ValueError("fine_steps must be a positive even integer")
    if num_paths <= 0 or max_iterations <= 0:
        raise ValueError("num_paths and max_iterations must be positive")
    if not 0.0 < smoothing <= 1.0:
        raise ValueError("smoothing must lie in (0, 1]")
    if min_disagreement_paths <= 0 or min_disagreement_paths > num_paths:
        raise ValueError("min_disagreement_paths is outside its valid range")
    if control_bound <= 0.0 or parameter_tolerance <= 0.0:
        raise ValueError("control_bound and parameter_tolerance must be positive")
    if stable_repetitions <= 0:
        raise ValueError("stable_repetitions must be positive")

    control = torch.tensor(initial_control, dtype=torch.float64)
    segments = len(initial_control)
    history: list[CorrectionCEMIteration] = []
    stable = 0
    torch.manual_seed(seed)
    for iteration in range(max_iterations):
        proposal = TimePiecewiseTwoDriverControl(
            _as_values(control), maturity=maturity
        )
        paths = simulate_coupled_rbergomi_adjacent(
            simulator,
            S0=spot,
            T=maturity,
            fine_steps=fine_steps,
            num_paths=num_paths,
            control_fn=proposal,
            record_augmented=True,
            dtype=torch.float64,
        )
        target = paths.target_fine_brownian_increments
        if target is None:
            raise RuntimeError("correction CEM requires recorded target increments")
        fine_event = task.hard_event(paths.fine.spot, paths.fine.step_dt)
        coarse_event = task.hard_event(paths.coarse.spot, paths.coarse.step_dt)
        signed = fine_event.to(torch.float64) - coarse_event.to(torch.float64)
        disagreement = signed != 0.0
        count = int(disagreement.sum())
        if count < min_disagreement_paths:
            raise RuntimeError(
                "too few adjacent-level disagreements for a stable CEM update: "
                f"observed {count}, required {min_disagreement_paths}"
            )
        log_weights = paths.log_likelihood[disagreement]
        normalized = torch.softmax(log_weights, dim=0)
        sufficient = _segment_sufficient_statistics(
            target, segments=segments, step_dt=paths.fine.step_dt
        )
        candidate = torch.sum(
            normalized[:, None, None] * sufficient[disagreement], dim=0
        ).cpu()
        updated = (1.0 - smoothing) * control + smoothing * candidate
        updated = torch.clamp(updated, min=-control_bound, max=control_bound)
        likelihood = torch.exp(paths.log_likelihood)
        contribution = signed * likelihood
        disagreement_contribution = disagreement.to(torch.float64) * likelihood
        maximum_change = float(torch.max(torch.abs(updated - control)))
        history.append(
            CorrectionCEMIteration(
                iteration=iteration,
                control_before=_as_values(control),
                control_candidate=_as_values(candidate),
                control_after=_as_values(updated),
                proposal_disagreement_fraction=float(disagreement.double().mean()),
                target_disagreement_probability=float(
                    disagreement_contribution.mean()
                ),
                signed_correction_estimate=float(contribution.mean()),
                disagreement_weight_ess=float(
                    torch.reciprocal(torch.sum(normalized.square()))
                ),
                weighted_second_moment=float(torch.mean(contribution.square())),
                maximum_parameter_change=maximum_change,
            )
        )
        control = updated
        if maximum_change <= parameter_tolerance:
            stable += 1
            if stable >= stable_repetitions:
                return CorrectionCEMResult(
                    control=_as_values(control), converged=True, history=tuple(history)
                )
        else:
            stable = 0
    return CorrectionCEMResult(
        control=_as_values(control), converged=False, history=tuple(history)
    )
