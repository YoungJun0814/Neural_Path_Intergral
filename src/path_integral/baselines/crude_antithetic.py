"""Crude, antithetic, and exact-likelihood latent IS evaluation."""

from __future__ import annotations

from typing import Literal

import torch

from src.path_integral.baseline_framework import (
    BaselineCostLedger,
    FrozenBaselineProposal,
    evaluate_baseline_log_q_over_p,
    freeze_baseline_proposal,
    ordinary_is_contributions,
    sample_baseline_proposal,
)

from .rbergomi_common import BaselineUnitBatch, RBergomiBaselineProblem


def freeze_crude_or_antithetic_proposal(
    problem: RBergomiBaselineProblem,
    *,
    method: Literal["crude_mc", "antithetic_mc"],
    training_seed: int,
) -> FrozenBaselineProposal:
    """Freeze either natural iid Gaussian sampling or exact antithetic pairs."""

    if method not in {"crude_mc", "antithetic_mc"}:
        raise ValueError("method must be crude_mc or antithetic_mc")
    return freeze_baseline_proposal(
        method=method,
        task_id=problem.task_id,
        dimension=problem.latent_dimension,
        training_seed=training_seed,
        training_cost=BaselineCostLedger(),
    )


def evaluate_latent_is_units(
    problem: RBergomiBaselineProblem,
    proposal: FrozenBaselineProposal,
    *,
    sample_count: int,
    seed: int,
) -> BaselineUnitBatch:
    """Evaluate an ordinary-mean IS baseline on the common latent path law."""

    if proposal.task_id != problem.task_id:
        raise ValueError("proposal and rBergomi problem task IDs differ")
    if proposal.dimension != problem.latent_dimension:
        raise ValueError("proposal dimension does not match the rBergomi latent law")
    if proposal.method in {"conditional_rbergomi", "smoothing_rqmc"}:
        raise ValueError("conditional and RQMC baselines require their dedicated evaluator")
    latent = sample_baseline_proposal(
        proposal,
        sample_count=sample_count,
        seed=seed,
    )
    paths = problem.simulate_latent(latent)
    event = problem.hard_event(paths).to(torch.float64)
    log_q_over_p = evaluate_baseline_log_q_over_p(latent, proposal)
    contributions = ordinary_is_contributions(event, log_q_over_p)
    if proposal.method == "antithetic_mc":
        if sample_count % 2:
            raise ValueError("antithetic baseline requires an even sample count")
        units = contributions.reshape(-1, 2).mean(dim=1)
    else:
        units = contributions
    likelihoods = (
        0 if proposal.family in {"target_gaussian", "rqmc_target_gaussian"} else sample_count
    )
    return BaselineUnitBatch(
        unit_contributions=units,
        raw_sample_count=sample_count,
        likelihood_evaluations=likelihoods,
        cdf_calls=0,
        quadrature_calls=0,
    )
