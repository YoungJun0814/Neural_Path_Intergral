"""Exact terminal conditioning baseline for the finite-grid rBergomi law."""

from __future__ import annotations

import math

import torch

from src.path_integral.baseline_framework import (
    BaselineCostLedger,
    FrozenBaselineProposal,
    freeze_baseline_proposal,
    sample_baseline_proposal,
)
from src.path_integral.path_functionals import TerminalThresholdTask

from .rbergomi_common import BaselineUnitBatch, RBergomiBaselineProblem


def freeze_conditional_rbergomi_proposal(
    problem: RBergomiBaselineProblem,
    *,
    training_seed: int,
) -> FrozenBaselineProposal:
    """Freeze the natural law of the two local BLP normals per time cell.

    Conditional on these coordinates, terminal log-price is a scalar Gaussian.
    Barrier and occupation events deliberately use a different baseline because
    their conditional laws are not one-dimensional terminal Gaussian CDFs.
    """

    if not isinstance(problem.task, TerminalThresholdTask):
        raise ValueError("conditional rBergomi is exact only for terminal tasks")
    return freeze_baseline_proposal(
        method="conditional_rbergomi",
        task_id=problem.task_id,
        dimension=problem.local_dimension,
        training_seed=training_seed,
        training_cost=BaselineCostLedger(),
        conditional_integral="analytic_gaussian_cdf",
    )


def evaluate_conditional_terminal_units(
    problem: RBergomiBaselineProblem,
    proposal: FrozenBaselineProposal,
    *,
    sample_count: int,
    seed: int,
) -> BaselineUnitBatch:
    """Integrate the independent price driver analytically, path by path."""

    if not isinstance(problem.task, TerminalThresholdTask):
        raise ValueError("conditional rBergomi is exact only for terminal tasks")
    if proposal.method != "conditional_rbergomi":
        raise ValueError("a conditional rBergomi proposal is required")
    if proposal.task_id != problem.task_id or proposal.dimension != problem.local_dimension:
        raise ValueError("proposal does not match the conditional rBergomi problem")
    local = sample_baseline_proposal(proposal, sample_count=sample_count, seed=seed)
    paths = problem.simulate_local(local)
    conditional_variance = (
        (1.0 - problem.rho**2) * problem.step_dt * torch.sum(paths.variance[:, :-1], dim=1)
    )
    if not torch.isfinite(conditional_variance).all() or bool((conditional_variance <= 0.0).any()):
        raise FloatingPointError("conditional terminal variance must be finite and positive")
    standardized = (math.log(problem.task.level) - paths.log_spot[:, -1]) / torch.sqrt(
        conditional_variance
    )
    values = torch.special.ndtr(standardized)
    if not torch.isfinite(values).all() or bool((values < 0.0).any()) or bool((values > 1.0).any()):
        raise FloatingPointError("conditional terminal CDF is invalid")
    return BaselineUnitBatch(
        unit_contributions=values,
        raw_sample_count=sample_count,
        likelihood_evaluations=0,
        cdf_calls=sample_count,
        quadrature_calls=0,
    )
