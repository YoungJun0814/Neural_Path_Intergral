"""Exact conditional transport of the local Gaussian Volterra coordinates."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
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
from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.provenance import process_peak_resident_memory_bytes
from src.path_integral.residual_smc import (
    AdaptiveResidualSMCConfig,
    AdaptiveResidualSMCResult,
    run_adaptive_residual_smc,
)


def _derived_seed(root: int, role: str) -> int:
    if isinstance(root, bool) or not isinstance(root, int) or root < 0:
        raise ValueError("local Volterra root seed must be a nonnegative integer")
    digest = hashlib.sha256(f"NPI-V14-LOCAL-VOLTERRA\0{root}\0{role}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1) or 1


@dataclass(frozen=True)
class ConditionalTerminalLocalBatch:
    standardized_threshold: torch.Tensor
    log_conditional_value: torch.Tensor
    conditional_value: torch.Tensor


def evaluate_conditional_terminal_local(
    problem: RBergomiBaselineProblem, local: torch.Tensor
) -> ConditionalTerminalLocalBatch:
    """Integrate the complete independent price driver conditional on local paths."""

    if not isinstance(problem.task, TerminalThresholdTask):
        raise ValueError("local terminal conditioning requires a terminal task")
    if local.ndim != 2 or local.shape[1] != problem.local_dimension:
        raise ValueError("local Gaussian sample has the wrong shape")
    if local.device.type != "cpu" or local.dtype != torch.float64:
        raise ValueError("local Gaussian sample must be CPU float64")
    paths = problem.simulate_local(local)
    variance = (1.0 - problem.rho**2) * problem.step_dt * torch.sum(paths.variance[:, :-1], dim=1)
    if not torch.isfinite(variance).all() or bool((variance <= 0.0).any()):
        raise FloatingPointError("conditional terminal variance is invalid")
    threshold = (math.log(problem.task.level) - paths.log_spot[:, -1]) / torch.sqrt(variance)
    log_value = torch.special.log_ndtr(threshold)
    value = torch.exp(log_value)
    if bool(torch.isnan(log_value).any()) or not torch.isfinite(value).all():
        raise FloatingPointError("conditional terminal probability is invalid")
    return ConditionalTerminalLocalBatch(
        standardized_threshold=threshold,
        log_conditional_value=log_value,
        conditional_value=value,
    )


@dataclass(frozen=True)
class LocalVolterraTransportTrainingConfig:
    target_powers: tuple[float, ...] = (0.1, 0.25, 0.5)
    shifted_weights: tuple[float, ...] = (0.4, 0.35, 0.25)
    defensive_weight: float = 0.2
    replicates_per_power: int = 1
    replicate_aggregation: Literal["average", "mixture"] = "average"
    smc: AdaptiveResidualSMCConfig = AdaptiveResidualSMCConfig()

    def __post_init__(self) -> None:
        if not self.target_powers or len(self.target_powers) != len(self.shifted_weights):
            raise ValueError("one shifted weight is required per target power")
        if any(not math.isfinite(x) or not 0.0 < x <= 1.0 for x in self.target_powers):
            raise ValueError("local target powers must lie in (0, 1]")
        if any(a >= b for a, b in zip(self.target_powers, self.target_powers[1:], strict=False)):
            raise ValueError("local target powers must be strictly increasing")
        if any(not math.isfinite(x) or x <= 0.0 for x in self.shifted_weights):
            raise ValueError("shifted weights must be finite and positive")
        if not math.isclose(sum(self.shifted_weights), 1.0, rel_tol=1e-12):
            raise ValueError("shifted weights must sum to one")
        if not 0.0 < self.defensive_weight < 1.0:
            raise ValueError("defensive weight must lie in (0, 1)")
        if (
            isinstance(self.replicates_per_power, bool)
            or not isinstance(self.replicates_per_power, int)
            or self.replicates_per_power < 1
        ):
            raise ValueError("replicates per power must be a positive integer")
        if self.replicate_aggregation not in {"average", "mixture"}:
            raise ValueError("replicate aggregation must be average or mixture")


@dataclass(frozen=True)
class LocalVolterraTransportTrainingResult:
    proposal: FrozenBaselineProposal
    smc_results: tuple[AdaptiveResidualSMCResult, ...]
    target_powers: tuple[float, ...]
    mean_norms: tuple[float, ...]
    replicate_mean_norms: tuple[tuple[float, ...], ...]
    replicate_mean_minimum_cosines: tuple[float, ...]
    all_training_seeds: tuple[int, ...]


def train_local_volterra_transport(
    problem: RBergomiBaselineProblem,
    *,
    training_seed: int,
    config: LocalVolterraTransportTrainingConfig | None = None,
) -> LocalVolterraTransportTrainingResult:
    config = config or LocalVolterraTransportTrainingConfig()
    # Embed R^(2N) as the hyperplane orthogonal to the final coordinate. This
    # reuses the audited Gaussian-reference SMC without changing its law.
    direction = torch.zeros(problem.local_dimension + 1, dtype=torch.float64)
    direction[-1] = 1.0
    means: list[torch.Tensor] = [torch.zeros(problem.local_dimension, dtype=torch.float64)]
    smc_results: list[AdaptiveResidualSMCResult] = []
    replicate_norms: list[tuple[float, ...]] = []
    minimum_cosines: list[float] = []
    all_seeds = [training_seed]
    cost = BaselineCostLedger(measurement_mode="standardized_hardware_wall")
    for index, power in enumerate(config.target_powers):
        power_means: list[torch.Tensor] = []
        for replicate in range(config.replicates_per_power):
            smc_seed = _derived_seed(training_seed, f"power-{index}-replicate-{replicate}-smc")
            all_seeds.append(smc_seed)

            def log_potential(
                residual: torch.Tensor, *, target_power: float = power
            ) -> torch.Tensor:
                return (
                    target_power
                    * evaluate_conditional_terminal_local(
                        problem, residual[:, :-1]
                    ).log_conditional_value
                )

            smc = run_adaptive_residual_smc(
                direction=direction,
                log_potential_fn=log_potential,
                root_seed=smc_seed,
                config=config.smc,
                potential_work_per_particle=float(problem.local_dimension + problem.steps + 1),
            )
            power_means.append(smc.residual_particles[:, :-1].mean(dim=0))
            smc_results.append(smc)
            all_seeds.extend(smc.used_seeds)
            cost = cost.plus(
                smc.training_cost.plus(
                    BaselineCostLedger(
                        cdf_calls=smc.training_cost.training_samples,
                        measurement_mode="standardized_hardware_wall",
                    )
                )
            )
        if config.replicate_aggregation == "average":
            means.append(torch.stack(power_means).mean(dim=0))
        else:
            # Preserve independently discovered basins instead of placing one
            # Gaussian at their arithmetic midpoint.  The frozen proposal is
            # still a finite Gaussian mixture with an exact density.
            means.extend(power_means)
        replicate_norms.append(tuple(float(torch.linalg.vector_norm(item)) for item in power_means))
        if len(power_means) == 1:
            minimum_cosines.append(1.0)
        else:
            cosines = [
                float(
                    torch.dot(left, right)
                    / (torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right))
                )
                for left_index, left in enumerate(power_means)
                for right in power_means[left_index + 1 :]
            ]
            minimum_cosines.append(min(cosines))
    if len(set(all_seeds)) != len(all_seeds):
        raise AssertionError("local Volterra training streams collided")
    shifted_mass = 1.0 - config.defensive_weight
    if config.replicate_aggregation == "average":
        shifted_weights = config.shifted_weights
    else:
        shifted_weights = tuple(
            weight / config.replicates_per_power
            for weight in config.shifted_weights
            for _ in range(config.replicates_per_power)
        )
    weights = (config.defensive_weight,) + tuple(
        shifted_mass * weight for weight in shifted_weights
    )
    proposal = freeze_baseline_proposal(
        method="defensive_cem",
        task_id=problem.task_id,
        dimension=problem.local_dimension,
        training_seed=training_seed,
        training_cost=cost,
        training_budget_work_units=cost.algorithmic_work_units,
        component_means=tuple(tuple(float(value) for value in mean) for mean in means),
        component_weights=weights,
        conditional_integral="analytic_gaussian_cdf_after_local_transport",
    )
    return LocalVolterraTransportTrainingResult(
        proposal=proposal,
        smc_results=tuple(smc_results),
        target_powers=config.target_powers,
        mean_norms=tuple(float(torch.linalg.vector_norm(mean)) for mean in means[1:]),
        replicate_mean_norms=tuple(replicate_norms),
        replicate_mean_minimum_cosines=tuple(minimum_cosines),
        all_training_seeds=tuple(all_seeds),
    )


@dataclass(frozen=True)
class LocalVolterraTransportEvaluationBatch:
    contribution: torch.Tensor
    raw_contribution: torch.Tensor
    conditional_probability: torch.Tensor
    likelihood: torch.Tensor
    standardized_threshold: torch.Tensor
    evaluation_cost: BaselineCostLedger
    maximum_likelihood_bound_violation: float


def evaluate_local_volterra_transport(
    problem: RBergomiBaselineProblem,
    proposal: FrozenBaselineProposal,
    *,
    sample_count: int,
    proposal_seed: int,
    coordinate_seed: int,
) -> LocalVolterraTransportEvaluationBatch:
    if proposal.task_id != problem.task_id or proposal.dimension != problem.local_dimension:
        raise ValueError("local Volterra proposal does not match the problem")
    if proposal.method != "defensive_cem" or proposal.self_normalized:
        raise ValueError("local Volterra evaluation requires an exact defensive mixture")
    if proposal_seed == coordinate_seed:
        raise ValueError("proposal and coordinate seeds must be disjoint")
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    local = sample_baseline_proposal(proposal, sample_count=sample_count, seed=proposal_seed)
    conditional = evaluate_conditional_terminal_local(problem, local)
    log_q_over_p = evaluate_baseline_log_q_over_p(local, proposal)
    contribution = ordinary_is_contributions(conditional.conditional_value, log_q_over_p)
    likelihood = torch.exp(-log_q_over_p)
    coordinate = torch.randn(
        sample_count,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(coordinate_seed),
    )
    raw = (coordinate <= conditional.standardized_threshold).to(torch.float64) * likelihood
    bound = 1.0 / float(proposal.component_weights[0])
    violation = max(0.0, float(torch.amax(likelihood)) - bound)
    components = len(proposal.component_weights)
    work = sample_count * (
        problem.local_dimension + problem.steps + 1 + components * problem.local_dimension
    )
    cost = BaselineCostLedger(
        final_samples=sample_count,
        likelihood_evaluations=sample_count,
        cdf_calls=sample_count,
        algorithmic_work_units=float(work),
        wall_seconds=time.perf_counter() - started_wall,
        cpu_seconds=time.process_time() - started_cpu,
        peak_memory_bytes=process_peak_resident_memory_bytes(),
        measurement_mode="standardized_hardware_wall",
    )
    return LocalVolterraTransportEvaluationBatch(
        contribution=contribution,
        raw_contribution=raw,
        conditional_probability=conditional.conditional_value,
        likelihood=likelihood,
        standardized_threshold=conditional.standardized_threshold,
        evaluation_cost=cost,
        maximum_likelihood_bound_violation=violation,
    )
