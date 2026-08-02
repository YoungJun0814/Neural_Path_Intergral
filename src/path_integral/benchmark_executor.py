"""Common train-frozen, pilot-frozen, ordinary-mean baseline execution."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from src.path_integral.baseline_framework import (
    BaselineAllocationPlan,
    BaselineCostLedger,
    BaselineEstimateArtifact,
    BaselineLifecycleAudit,
    FrozenBaselineProposal,
    audit_baseline_lifecycle,
    finalize_baseline_estimate,
    plan_baseline_allocation,
)
from src.path_integral.baselines import (
    BaselineUnitBatch,
    RBergomiBaselineProblem,
    evaluate_conditional_terminal_units,
    evaluate_latent_is_units,
    evaluate_smoothing_rqmc_units,
)
from src.path_integral.provenance import process_peak_resident_memory_bytes


@dataclass(frozen=True)
class BaselineExecutionRequest:
    """Frozen pilot/final request in independent inferential units."""

    pilot_units: int
    target_estimator_variance: float
    pilot_seed: int
    final_seed: int
    minimum_final_units: int = 2
    maximum_final_units: int = 10**9
    rqmc_points_per_randomization: int = 1
    minimum_nonzero_pilot_units: int = 0
    pilot_variance_safety_factor: float = 1.0

    def __post_init__(self) -> None:
        integers = (
            self.pilot_units,
            self.pilot_seed,
            self.final_seed,
            self.minimum_final_units,
            self.maximum_final_units,
            self.rqmc_points_per_randomization,
            self.minimum_nonzero_pilot_units,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in integers
        ):
            raise ValueError("execution counts and seeds must be nonnegative integers")
        if self.pilot_units < 2 or self.minimum_final_units < 2:
            raise ValueError("pilot and minimum final units must be at least two")
        if self.maximum_final_units < self.minimum_final_units:
            raise ValueError("maximum final units must not be below the minimum")
        if (
            not math.isfinite(self.target_estimator_variance)
            or self.target_estimator_variance <= 0.0
        ):
            raise ValueError("target estimator variance must be finite and positive")
        points = self.rqmc_points_per_randomization
        if points < 1 or points & (points - 1):
            raise ValueError("RQMC points per randomization must be a power of two")
        if self.minimum_nonzero_pilot_units > self.pilot_units:
            raise ValueError("minimum nonzero pilot units cannot exceed pilot units")
        if (
            not math.isfinite(self.pilot_variance_safety_factor)
            or self.pilot_variance_safety_factor < 1.0
        ):
            raise ValueError("pilot variance safety factor must be finite and at least one")


class PilotSupportError(RuntimeError):
    """Raised before final sampling when a rare-event pilot has no support."""

    def __init__(self, *, observed_nonzero_units: int, required_nonzero_units: int) -> None:
        self.observed_nonzero_units = observed_nonzero_units
        self.required_nonzero_units = required_nonzero_units
        super().__init__(
            "pilot has insufficient nonzero inferential units: "
            f"observed {observed_nonzero_units}, required {required_nonzero_units}"
        )


@dataclass(frozen=True)
class BaselineExecutionArtifact:
    """One fully audited pilot-to-final baseline execution."""

    proposal: FrozenBaselineProposal
    plan: BaselineAllocationPlan
    estimate: BaselineEstimateArtifact
    audit: BaselineLifecycleAudit
    pilot_unit_count: int
    pilot_mean: float
    pilot_variance: float
    planning_variance: float
    pilot_nonzero_unit_count: int
    pilot_variance_safety_factor: float
    pilot_cost: BaselineCostLedger

    def __post_init__(self) -> None:
        if self.pilot_unit_count < 2:
            raise ValueError("artifact requires at least two pilot units")
        if not all(
            math.isfinite(value)
            for value in (
                self.pilot_mean,
                self.pilot_variance,
                self.planning_variance,
                self.pilot_variance_safety_factor,
            )
        ):
            raise ValueError("pilot moments must be finite")
        if self.pilot_variance < 0.0 or self.planning_variance < self.pilot_variance:
            raise ValueError("pilot variance must be nonnegative")
        if not 0 <= self.pilot_nonzero_unit_count <= self.pilot_unit_count:
            raise ValueError("invalid nonzero pilot-unit count")
        if self.pilot_variance_safety_factor < 1.0:
            raise ValueError("pilot variance safety factor must be at least one")
        if not self.audit.passed:
            raise ValueError("baseline execution artifact requires a passing lifecycle audit")


def _raw_count(proposal: FrozenBaselineProposal, units: int, rqmc_points: int) -> int:
    if proposal.method == "antithetic_mc":
        return 2 * units
    if proposal.method == "smoothing_rqmc":
        return rqmc_points * units
    return units


def evaluate_baseline_units(
    problem: RBergomiBaselineProblem,
    proposal: FrozenBaselineProposal,
    *,
    unit_count: int,
    seed: int,
    rqmc_points_per_randomization: int = 1,
) -> BaselineUnitBatch:
    """Dispatch a baseline while preserving its declared inferential unit."""

    if isinstance(unit_count, bool) or not isinstance(unit_count, int) or unit_count < 1:
        raise ValueError("unit_count must be a positive integer")
    if proposal.method == "conditional_rbergomi":
        return evaluate_conditional_terminal_units(
            problem, proposal, sample_count=unit_count, seed=seed
        )
    if proposal.method == "smoothing_rqmc":
        return evaluate_smoothing_rqmc_units(
            problem,
            proposal,
            randomizations=unit_count,
            points_per_randomization=rqmc_points_per_randomization,
            seed=seed,
        )
    return evaluate_latent_is_units(
        problem,
        proposal,
        sample_count=_raw_count(proposal, unit_count, rqmc_points_per_randomization),
        seed=seed,
    )


def _evaluation_cost(
    problem: RBergomiBaselineProblem,
    proposal: FrozenBaselineProposal,
    batch: BaselineUnitBatch,
    *,
    role: str,
    wall_seconds: float,
    cpu_seconds: float,
    peak_memory_bytes: int,
) -> BaselineCostLedger:
    """Charge simulation, Gaussian generation, likelihood, and integration work."""

    if role not in {"planning", "final"}:
        raise ValueError("evaluation cost role must be planning or final")
    gaussian_dimension = (
        problem.local_dimension
        if proposal.method == "conditional_rbergomi"
        else problem.latent_dimension
    )
    work = batch.raw_sample_count * (gaussian_dimension + problem.steps)
    work += batch.likelihood_evaluations * proposal.dimension
    work += batch.cdf_calls + 32 * batch.quadrature_calls
    counts = {
        "planning_samples": batch.raw_sample_count if role == "planning" else 0,
        "final_samples": batch.raw_sample_count if role == "final" else 0,
    }
    return BaselineCostLedger(
        **counts,
        likelihood_evaluations=batch.likelihood_evaluations,
        cdf_calls=batch.cdf_calls,
        quadrature_calls=batch.quadrature_calls,
        algorithmic_work_units=float(work),
        wall_seconds=wall_seconds,
        cpu_seconds=cpu_seconds,
        peak_memory_bytes=peak_memory_bytes,
        measurement_mode="standardized_hardware_wall",
    )


def _measured_evaluation(
    problem: RBergomiBaselineProblem,
    proposal: FrozenBaselineProposal,
    *,
    unit_count: int,
    seed: int,
    rqmc_points_per_randomization: int,
    role: str,
) -> tuple[BaselineUnitBatch, BaselineCostLedger]:
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    batch = evaluate_baseline_units(
        problem,
        proposal,
        unit_count=unit_count,
        seed=seed,
        rqmc_points_per_randomization=rqmc_points_per_randomization,
    )
    cpu_seconds = time.process_time() - cpu_started
    wall_seconds = time.perf_counter() - wall_started
    cost = _evaluation_cost(
        problem,
        proposal,
        batch,
        role=role,
        wall_seconds=wall_seconds,
        cpu_seconds=cpu_seconds,
        peak_memory_bytes=process_peak_resident_memory_bytes(),
    )
    return batch, cost


def execute_baseline_lifecycle(
    problem: RBergomiBaselineProblem,
    proposal: FrozenBaselineProposal,
    request: BaselineExecutionRequest,
) -> BaselineExecutionArtifact:
    """Run an independent pilot, freeze integer allocation, then run final data."""

    if len({proposal.training_seed, request.pilot_seed, request.final_seed}) != 3:
        raise ValueError("training, pilot, and final seeds must be disjoint")
    rqmc_points = request.rqmc_points_per_randomization
    if proposal.method != "smoothing_rqmc" and rqmc_points != 1:
        raise ValueError("only smoothing RQMC may use multiple points per unit")
    pilot, pilot_cost = _measured_evaluation(
        problem,
        proposal,
        unit_count=request.pilot_units,
        seed=request.pilot_seed,
        rqmc_points_per_randomization=rqmc_points,
        role="planning",
    )
    pilot_variance = float(torch.var(pilot.unit_contributions, unbiased=True))
    pilot_nonzero_units = int(torch.count_nonzero(pilot.unit_contributions))
    if pilot_nonzero_units < request.minimum_nonzero_pilot_units:
        raise PilotSupportError(
            observed_nonzero_units=pilot_nonzero_units,
            required_nonzero_units=request.minimum_nonzero_pilot_units,
        )
    planning_variance = pilot_variance * request.pilot_variance_safety_factor
    plan = plan_baseline_allocation(
        proposal,
        pilot_variance=planning_variance,
        target_variance=request.target_estimator_variance,
        pilot_seed=request.pilot_seed,
        final_seed=request.final_seed,
        pilot_units=request.pilot_units,
        minimum_units=request.minimum_final_units,
        maximum_units=request.maximum_final_units,
        points_per_unit=rqmc_points if proposal.method == "smoothing_rqmc" else None,
        planning_cost=pilot_cost,
    )
    final, final_cost = _measured_evaluation(
        problem,
        proposal,
        unit_count=plan.planned_units,
        seed=plan.final_seed,
        rqmc_points_per_randomization=rqmc_points,
        role="final",
    )
    if final.unit_contributions.numel() != plan.planned_units:
        raise AssertionError("evaluator returned the wrong number of final units")
    estimate = finalize_baseline_estimate(
        proposal,
        plan,
        final.unit_contributions,
        final_cost=final_cost,
    )
    audit = audit_baseline_lifecycle(proposal, plan, estimate)
    if not audit.passed:
        raise RuntimeError(f"baseline lifecycle audit failed: {audit.failures}")
    return BaselineExecutionArtifact(
        proposal=proposal,
        plan=plan,
        estimate=estimate,
        audit=audit,
        pilot_unit_count=request.pilot_units,
        pilot_mean=float(torch.mean(pilot.unit_contributions)),
        pilot_variance=pilot_variance,
        planning_variance=planning_variance,
        pilot_nonzero_unit_count=pilot_nonzero_units,
        pilot_variance_safety_factor=request.pilot_variance_safety_factor,
        pilot_cost=pilot_cost,
    )
