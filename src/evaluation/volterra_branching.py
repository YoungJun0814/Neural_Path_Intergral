"""Boundary-measurable allocation and evaluation for Volterra bridge branches."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from src.path_integral.path_functionals import DownsideExcursionTask
from src.path_integral.rbergomi_branching import (
    RBergomiCoarseTrunks,
    refine_rbergomi_coarse_trunks,
    subset_rbergomi_coarse_trunks,
)
from src.physics_engine import RBergomiSimulator


@dataclass(frozen=True)
class BoundaryBranchingPolicy:
    """Two-tier branch allocation measurable from a completed coarse path."""

    hit_band: float
    occupation_band: float
    high_branches: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.hit_band) or self.hit_band < 0.0:
            raise ValueError("hit_band must be finite and nonnegative")
        if not math.isfinite(self.occupation_band) or self.occupation_band < 0.0:
            raise ValueError("occupation_band must be finite and nonnegative")
        if self.high_branches < 1:
            raise ValueError("high_branches must be positive")


@dataclass(frozen=True)
class BoundaryPolicyCalibration:
    policy: BoundaryBranchingPolicy
    baseline_work_proxy: float
    selected_work_proxy: float
    work_ratio: float
    selected_fraction: float
    single_path_variance: float
    mean_branches: float


@dataclass(frozen=True)
class AdaptiveBranchedCorrection:
    contributions: torch.Tensor
    branch_counts: torch.Tensor
    likelihood_parent_means: torch.Tensor
    first_branch_log_likelihood: torch.Tensor
    control_energy: float
    elapsed_refinement_seconds: float
    mean_branches: float
    selected_fraction: float
    maximum_constraint_error: float
    branch_disagreement_fraction: float


def coarse_boundary_margins(
    trunks: RBergomiCoarseTrunks, task: DownsideExcursionTask
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized absolute hit and occupation margins from coarse paths."""
    running_minimum, occupation, _hit = task.prefix_state(trunks.spot, 2.0 * trunks.fine_dt)
    hit_margin = torch.abs(running_minimum[:, -1] - task.hit_barrier) / task.hit_scale
    occupation_margin = (
        torch.abs(occupation[:, -1] - task.minimum_occupation) / task.occupation_scale
    )
    return hit_margin, occupation_margin


def boundary_branch_counts(
    trunks: RBergomiCoarseTrunks,
    task: DownsideExcursionTask,
    policy: BoundaryBranchingPolicy,
) -> torch.Tensor:
    """Allocate before fine residuals are sampled, preserving exactness."""
    hit_margin, occupation_margin = coarse_boundary_margins(trunks, task)
    selected = (hit_margin <= policy.hit_band) | (occupation_margin <= policy.occupation_band)
    return torch.where(
        selected,
        torch.full_like(hit_margin, policy.high_branches, dtype=torch.long),
        torch.ones_like(hit_margin, dtype=torch.long),
    )


def calibrate_boundary_branching_policy(
    trunks: RBergomiCoarseTrunks,
    task: DownsideExcursionTask,
    branch_contributions: torch.Tensor,
    *,
    hit_bands: Sequence[float],
    occupation_bands: Sequence[float],
    high_branch_candidates: Sequence[int],
    trunk_work: float | None = None,
    refinement_work_per_branch: float | None = None,
) -> BoundaryPolicyCalibration:
    """Minimize an explicit variance-times-innovation-work proxy on development data."""
    parents = trunks.spot.shape[0]
    if (
        branch_contributions.ndim != 2
        or branch_contributions.shape[0] != parents
        or branch_contributions.shape[1] < 2
    ):
        raise ValueError("branch_contributions must have shape (parents, max_branches>=2)")
    if (
        not branch_contributions.is_floating_point()
        or not torch.isfinite(branch_contributions).all()
    ):
        raise ValueError("branch_contributions must be finite floating point")
    if not hit_bands or not occupation_bands or not high_branch_candidates:
        raise ValueError("calibration grids must be nonempty")
    maximum_branches = branch_contributions.shape[1]
    if any(value < 2 or value > maximum_branches for value in high_branch_candidates):
        raise ValueError("high branch candidates must lie in [2, max_branches]")
    resolved_trunk_work = float(trunks.fine_steps // 2) if trunk_work is None else float(trunk_work)
    resolved_refinement_work = (
        float(trunks.fine_steps)
        if refinement_work_per_branch is None
        else float(refinement_work_per_branch)
    )
    if resolved_trunk_work < 0.0 or resolved_refinement_work <= 0.0:
        raise ValueError("work coefficients are outside their valid ranges")
    baseline_variance = float(branch_contributions[:, 0].var(unbiased=True))
    baseline_work = baseline_variance * (resolved_trunk_work + resolved_refinement_work)
    best: BoundaryPolicyCalibration | None = None
    for high_branches in high_branch_candidates:
        for hit_band in hit_bands:
            for occupation_band in occupation_bands:
                policy = BoundaryBranchingPolicy(
                    hit_band=float(hit_band),
                    occupation_band=float(occupation_band),
                    high_branches=int(high_branches),
                )
                counts = boundary_branch_counts(trunks, task, policy)
                selected = counts > 1
                contribution = branch_contributions[:, 0].clone()
                if bool(selected.any()):
                    contribution[selected] = branch_contributions[selected, :high_branches].mean(
                        dim=1
                    )
                variance = float(contribution.var(unbiased=True))
                mean_branches = float(counts.double().mean())
                work = variance * (resolved_trunk_work + resolved_refinement_work * mean_branches)
                candidate = BoundaryPolicyCalibration(
                    policy=policy,
                    baseline_work_proxy=baseline_work,
                    selected_work_proxy=work,
                    work_ratio=baseline_work / work if work > 0.0 else math.inf,
                    selected_fraction=float(selected.double().mean()),
                    single_path_variance=variance,
                    mean_branches=mean_branches,
                )
                if best is None or candidate.selected_work_proxy < best.selected_work_proxy:
                    best = candidate
    if best is None:
        raise RuntimeError("no boundary policy candidate was evaluated")
    return best


def evaluate_adaptive_branched_correction(
    simulator: RBergomiSimulator,
    trunks: RBergomiCoarseTrunks,
    task: DownsideExcursionTask,
    policy: BoundaryBranchingPolicy,
) -> AdaptiveBranchedCorrection:
    """Evaluate an exact boundary-band branch allocation."""
    counts = boundary_branch_counts(trunks, task, policy)
    return evaluate_variable_branched_correction(simulator, trunks, task, branch_counts=counts)


def evaluate_variable_branched_correction(
    simulator: RBergomiSimulator,
    trunks: RBergomiCoarseTrunks,
    task: DownsideExcursionTask,
    *,
    branch_counts: torch.Tensor,
) -> AdaptiveBranchedCorrection:
    """Evaluate any positive coarse-measurable integer branch allocation."""
    parents = trunks.spot.shape[0]
    if (
        branch_counts.ndim != 1
        or branch_counts.shape[0] != parents
        or branch_counts.device != trunks.spot.device
    ):
        raise ValueError("branch_counts must match the parent batch and device")
    if branch_counts.dtype != torch.long:
        raise TypeError("branch_counts must have dtype torch.long")
    if bool((branch_counts < 1).any()):
        raise ValueError("all branch counts must be positive")
    counts = branch_counts
    contributions = torch.empty(parents, device=trunks.spot.device, dtype=torch.float64)
    likelihood_parent_means = torch.empty_like(contributions)
    first_branch_log_likelihood = torch.empty_like(contributions)
    total_disagreements = 0
    total_branches = 0
    maximum_constraint_error = 0.0
    control_energy: float | None = None
    start = time.perf_counter()
    coarse_event_full = task.hard_event(trunks.spot, 2.0 * trunks.fine_dt).to(torch.float64)
    for branch_count in torch.unique(counts, sorted=True).tolist():
        resolved_count = int(branch_count)
        selected = counts == resolved_count
        selected_trunks = subset_rbergomi_coarse_trunks(trunks, selected)
        refined = refine_rbergomi_coarse_trunks(simulator, selected_trunks, branches=resolved_count)
        selected_parents = int(selected.sum())
        fine_event = (
            task.hard_event(
                refined.fine_spot.reshape(-1, trunks.fine_steps + 1),
                trunks.fine_dt,
            )
            .to(torch.float64)
            .reshape(selected_parents, resolved_count)
        )
        coarse_event = coarse_event_full[selected]
        likelihood = torch.exp(refined.log_likelihood)
        branch_values = (fine_event - coarse_event[:, None]) * likelihood
        contributions[selected] = branch_values.mean(dim=1)
        likelihood_parent_means[selected] = likelihood.mean(dim=1)
        first_branch_log_likelihood[selected] = refined.log_likelihood[:, 0]
        if control_energy is None:
            control_energy = refined.control_energy
        elif not math.isclose(control_energy, refined.control_energy, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("deterministic control energy changed across branch groups")
        total_disagreements += int((fine_event != coarse_event[:, None]).sum())
        total_branches += fine_event.numel()
        maximum_constraint_error = max(
            maximum_constraint_error, refined.conditional_constraint_error
        )
    elapsed = time.perf_counter() - start
    return AdaptiveBranchedCorrection(
        contributions=contributions,
        branch_counts=counts,
        likelihood_parent_means=likelihood_parent_means,
        first_branch_log_likelihood=first_branch_log_likelihood,
        control_energy=0.0 if control_energy is None else control_energy,
        elapsed_refinement_seconds=elapsed,
        mean_branches=float(counts.double().mean()),
        selected_fraction=float((counts > 1).double().mean()),
        maximum_constraint_error=maximum_constraint_error,
        branch_disagreement_fraction=total_disagreements / total_branches,
    )
