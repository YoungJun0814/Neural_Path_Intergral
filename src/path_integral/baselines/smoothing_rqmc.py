"""Randomized QMC with exact rank-one Gaussian path smoothing."""

from __future__ import annotations

import torch

from src.path_integral.baseline_framework import (
    BaselineCostLedger,
    FrozenBaselineProposal,
    freeze_baseline_proposal,
    sample_baseline_proposal,
)
from src.path_integral.gaussian_smoothing import positive_flat_direction
from src.path_integral.rbergomi_dcs_mlmc import scalar_task_threshold
from src.path_integral.rbergomi_smoothing import affine_rbergomi_log_spot

from .rbergomi_common import BaselineUnitBatch, RBergomiBaselineProblem


def freeze_smoothing_rqmc_proposal(
    problem: RBergomiBaselineProblem,
    *,
    training_seed: int,
) -> FrozenBaselineProposal:
    """Freeze a scrambled-Sobol target proposal with analytic path smoothing."""

    return freeze_baseline_proposal(
        method="smoothing_rqmc",
        task_id=problem.task_id,
        dimension=problem.latent_dimension,
        training_seed=training_seed,
        training_cost=BaselineCostLedger(),
        conditional_integral="analytic_gaussian_cdf",
    )


def evaluate_smoothing_rqmc_units(
    problem: RBergomiBaselineProblem,
    proposal: FrozenBaselineProposal,
    *,
    randomizations: int,
    points_per_randomization: int,
    seed: int,
    projection_tolerance: float = 1e-12,
) -> BaselineUnitBatch:
    """Return one independent estimator per Owen-scrambled Sobol randomization.

    The price-driver component parallel to a positive unit vector is removed
    before simulation and then integrated exactly.  Hence each Sobol point has
    dimension ``3N`` but its parallel coordinate is analytically marginalized;
    independent scrambles, rather than Sobol points, are the inferential units.
    """

    if proposal.method != "smoothing_rqmc":
        raise ValueError("a smoothing RQMC proposal is required")
    if proposal.task_id != problem.task_id or proposal.dimension != problem.latent_dimension:
        raise ValueError("proposal does not match the smoothing RQMC problem")
    if (
        isinstance(randomizations, bool)
        or not isinstance(randomizations, int)
        or randomizations < 1
    ):
        raise ValueError("randomizations must be a positive integer")
    if (
        isinstance(points_per_randomization, bool)
        or not isinstance(points_per_randomization, int)
        or points_per_randomization < 1
        or points_per_randomization & (points_per_randomization - 1)
    ):
        raise ValueError("points per RQMC randomization must be a power of two")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if projection_tolerance < 0.0:
        raise ValueError("projection tolerance must be nonnegative")

    direction = positive_flat_direction(problem.steps, device="cpu", dtype=torch.float64)
    unit_values: list[torch.Tensor] = []
    for randomization in range(randomizations):
        latent = sample_baseline_proposal(
            proposal,
            sample_count=points_per_randomization,
            seed=seed + randomization,
        )
        price = latent[:, problem.local_dimension :]
        coordinate = price @ direction
        residual = price - coordinate.unsqueeze(1) * direction.unsqueeze(0)
        projected = torch.cat((latent[:, : problem.local_dimension], residual), dim=1)
        paths = problem.simulate_latent(projected)
        increments = paths.proposal_brownian_increments
        if increments is None:
            raise AssertionError("BLP simulator omitted proposal Brownian increments")
        affine = affine_rbergomi_log_spot(
            spot=paths.spot,
            log_spot=paths.log_spot,
            variance=paths.variance,
            proposal_fine_brownian_increments=increments,
            fine_step_dt=paths.step_dt,
            rho=problem.rho,
            direction=direction,
        )
        if float(torch.max(torch.abs(affine.coordinate))) > projection_tolerance:
            raise FloatingPointError("RQMC smoothing residual is not orthogonal")
        threshold = scalar_task_threshold(
            affine.intercept,
            affine.slope,
            step_dt=paths.step_dt,
            task=problem.task,
        )
        conditional = torch.special.ndtr(threshold)
        if (
            not torch.isfinite(conditional).all()
            or bool((conditional < 0.0).any())
            or bool((conditional > 1.0).any())
        ):
            raise FloatingPointError("smoothed RQMC conditional value is invalid")
        unit_values.append(torch.mean(conditional))
    units = torch.stack(unit_values)
    raw = randomizations * points_per_randomization
    return BaselineUnitBatch(
        unit_contributions=units,
        raw_sample_count=raw,
        likelihood_evaluations=0,
        cdf_calls=raw,
        quadrature_calls=0,
    )
