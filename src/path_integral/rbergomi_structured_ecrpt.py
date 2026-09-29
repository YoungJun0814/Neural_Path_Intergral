"""V13 structured exact conditional residual transport for finite-grid rBergomi.

The adaptive SMC population is used only as training data.  Final estimators
sample independently from a frozen defensive low-rank flow and use the exact
ordinary likelihood ratio ``p_R / q_R``.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field

import torch

from src.path_integral.baseline_framework import BaselineCostLedger
from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.low_rank_residual_flow import (
    FrozenLowRankResidualFlow,
    LowRankResidualFlowTrainingConfig,
    fit_low_rank_residual_flow,
    freeze_low_rank_residual_flow,
    sample_low_rank_residual_flow,
)
from src.path_integral.provenance import process_peak_resident_memory_bytes
from src.path_integral.rbergomi_residual_transport import (
    DirectionSelectionResult,
    evaluate_rbergomi_conditional_residual,
    select_rbergomi_direction,
    validate_rbergomi_residual_direction,
)
from src.path_integral.residual_coupling_transport import ResidualHouseholderCoordinates
from src.path_integral.residual_smc import (
    AdaptiveResidualSMCConfig,
    AdaptiveResidualSMCResult,
    run_adaptive_residual_smc,
)


def _derived_seed(root: int, role: str) -> int:
    if isinstance(root, bool) or not isinstance(root, int) or root < 0:
        raise ValueError("V13 root seed must be a nonnegative integer")
    digest = hashlib.sha256(f"NPI-V13-STRUCTURED-ECRPT\0{root}\0{role}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1) or 1


@dataclass(frozen=True)
class StructuredECRPTTrainingConfig:
    direction_samples: int = 1024
    direction_rates: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)
    smc: AdaptiveResidualSMCConfig = field(default_factory=AdaptiveResidualSMCConfig)
    flow: LowRankResidualFlowTrainingConfig = field(
        default_factory=LowRankResidualFlowTrainingConfig
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.direction_samples, bool)
            or not isinstance(self.direction_samples, int)
            or self.direction_samples < 2
        ):
            raise ValueError("direction samples must be an integer of at least two")
        if not self.direction_rates or any(
            not math.isfinite(value) for value in self.direction_rates
        ):
            raise ValueError("direction rates must be a nonempty finite tuple")


@dataclass(frozen=True)
class StructuredECRPTTrainingResult:
    proposal: FrozenLowRankResidualFlow
    direction_selection: DirectionSelectionResult
    smc: AdaptiveResidualSMCResult
    training_seed: int
    selection_seed: int
    smc_seed: int
    optimizer_seed: int
    all_training_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.smc.particles_are_final_inferential_units:
            raise ValueError("SMC particles may not be declared final inferential units")
        if not self.smc.final_particles_equally_weighted or self.smc.final_beta != 1.0:
            raise ValueError("flow fitting requires resampled beta-one SMC particles")
        if len(set(self.all_training_seeds)) != len(self.all_training_seeds):
            raise ValueError("V13 training seed streams collided")


def train_structured_ecrpt(
    problem: RBergomiBaselineProblem,
    *,
    training_seed: int,
    config: StructuredECRPTTrainingConfig | None = None,
) -> StructuredECRPTTrainingResult:
    """Train and freeze the V13 proposal without creating inferential units."""

    config = config or StructuredECRPTTrainingConfig()
    selection_seed = _derived_seed(training_seed, "direction-selection")
    smc_seed = _derived_seed(training_seed, "adaptive-smc")
    optimizer_seed = _derived_seed(training_seed, "flow-optimizer")
    root_streams = (training_seed, selection_seed, smc_seed, optimizer_seed)
    if len(set(root_streams)) != len(root_streams):
        raise AssertionError("V13 root training streams collided")

    selection_wall = time.perf_counter()
    selection_cpu = time.process_time()
    selection = select_rbergomi_direction(
        problem,
        sample_count=config.direction_samples,
        seed=selection_seed,
        exponential_rates=config.direction_rates,
    )
    selection_elapsed_wall = time.perf_counter() - selection_wall
    selection_elapsed_cpu = time.process_time() - selection_cpu
    direction = selection.selected_direction

    def log_potential(residual: torch.Tensor) -> torch.Tensor:
        return evaluate_rbergomi_conditional_residual(
            problem, residual, direction
        ).log_conditional_value

    potential_work = float(problem.latent_dimension + problem.steps + 1)
    smc = run_adaptive_residual_smc(
        direction=direction,
        log_potential_fn=log_potential,
        root_seed=smc_seed,
        config=config.smc,
        potential_work_per_particle=potential_work,
    )
    flow = fit_low_rank_residual_flow(
        task_id=problem.task_id,
        residual=smc.residual_particles,
        direction=direction,
        training_seed=optimizer_seed,
        log_training_weights=None,
        config=config.flow,
    )

    selected_evaluations = config.direction_samples * len(config.direction_rates)
    selection_cost = BaselineCostLedger(
        screening_samples=selected_evaluations,
        cdf_calls=selected_evaluations,
        algorithmic_work_units=float(
            selected_evaluations * (problem.latent_dimension + problem.steps + 1)
        ),
        wall_seconds=selection_elapsed_wall,
        cpu_seconds=selection_elapsed_cpu,
        peak_memory_bytes=process_peak_resident_memory_bytes(),
        measurement_mode="standardized_hardware_wall",
    )
    # The generic SMC ledger counts potential calls as training samples.  In
    # this integration every such call is exactly one rBergomi CDF evaluation.
    smc_cost = smc.training_cost.plus(
        BaselineCostLedger(
            cdf_calls=smc.training_cost.training_samples,
            measurement_mode="standardized_hardware_wall",
        )
    )
    combined_cost = selection_cost.plus(smc_cost).plus(flow.training_cost)
    frozen = freeze_low_rank_residual_flow(
        task_id=problem.task_id,
        direction=direction,
        defensive_weight=flow.defensive_weight,
        maximum_log_scale=flow.maximum_log_scale,
        partition_style=flow.partition_style,
        layers=flow.layers,
        training_seed=training_seed,
        training_cost=combined_cost,
    )
    all_seeds = root_streams + smc.used_seeds
    return StructuredECRPTTrainingResult(
        proposal=frozen,
        direction_selection=selection,
        smc=smc,
        training_seed=training_seed,
        selection_seed=selection_seed,
        smc_seed=smc_seed,
        optimizer_seed=optimizer_seed,
        all_training_seeds=all_seeds,
    )


@dataclass(frozen=True)
class StructuredECRPTEvaluationBatch:
    raw_contribution: torch.Tensor
    ecrpt_contribution: torch.Tensor
    likelihood_normalization: torch.Tensor
    threshold: torch.Tensor
    labels: torch.Tensor
    raw_cost: BaselineCostLedger
    ecrpt_cost: BaselineCostLedger
    maximum_residual_projection_error: float
    maximum_path_reconstruction_error: float
    maximum_full_path_reconstruction_error: float
    maximum_likelihood_bound_violation: float
    hard_threshold_mismatch_count: int

    @property
    def inferential_unit_count(self) -> int:
        return int(self.ecrpt_contribution.numel())


def _evaluation_cost(
    problem: RBergomiBaselineProblem,
    proposal: FrozenLowRankResidualFlow,
    *,
    samples: int,
    cdf_calls: int,
    wall_seconds: float,
    cpu_seconds: float,
) -> BaselineCostLedger:
    conditioner_work = sum(
        4 * layer.rank * (len(layer.active_indices) + len(layer.transformed_indices))
        for layer in proposal.layers
    )
    work = samples * (
        problem.latent_dimension + problem.steps + conditioner_work + cdf_calls / samples
    )
    return BaselineCostLedger(
        final_samples=samples,
        likelihood_evaluations=samples,
        cdf_calls=cdf_calls,
        algorithmic_work_units=float(work),
        wall_seconds=wall_seconds,
        cpu_seconds=cpu_seconds,
        peak_memory_bytes=process_peak_resident_memory_bytes(),
        measurement_mode="standardized_hardware_wall",
    )


def evaluate_structured_ecrpt(
    problem: RBergomiBaselineProblem,
    proposal: FrozenLowRankResidualFlow,
    *,
    sample_count: int,
    gaussian_seed: int,
    label_seed: int,
    coordinate_seed: int,
    reconstruction_paths: int = 128,
) -> StructuredECRPTEvaluationBatch:
    """Return paired raw and conditional ordinary-IS IID contributions."""

    if proposal.task_id != problem.task_id:
        raise ValueError("structured proposal and problem task IDs differ")
    if not proposal.exact_likelihood or proposal.self_normalized or not proposal.frozen:
        raise ValueError("V13 evaluation requires a frozen exact ordinary proposal")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 2:
        raise ValueError("sample count must be an integer of at least two")
    seeds = (gaussian_seed, label_seed, coordinate_seed)
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("evaluation seeds must be nonnegative integers")
    if len(set(seeds)) != 3:
        raise ValueError("evaluation seed streams must be disjoint")
    if (
        isinstance(reconstruction_paths, bool)
        or not isinstance(reconstruction_paths, int)
        or reconstruction_paths < 1
    ):
        raise ValueError("reconstruction paths must be positive")
    direction = torch.tensor(proposal.direction, dtype=torch.float64)
    validate_rbergomi_residual_direction(problem, direction)

    common_wall = time.perf_counter()
    common_cpu = time.process_time()
    sample = sample_low_rank_residual_flow(
        proposal,
        sample_count,
        gaussian_seed=gaussian_seed,
        label_seed=label_seed,
    )
    conditional = evaluate_rbergomi_conditional_residual(problem, sample.residual, direction)
    common_elapsed_wall = time.perf_counter() - common_wall
    common_elapsed_cpu = time.process_time() - common_cpu

    raw_wall = time.perf_counter()
    raw_cpu = time.process_time()
    coordinate = torch.randn(
        sample_count,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(coordinate_seed),
    )
    raw = (coordinate <= conditional.threshold).to(torch.float64) * sample.likelihood
    raw_elapsed_wall = time.perf_counter() - raw_wall
    raw_elapsed_cpu = time.process_time() - raw_cpu

    conditional_wall = time.perf_counter()
    conditional_cpu = time.process_time()
    contribution = conditional.conditional_value * sample.likelihood
    conditional_elapsed_wall = time.perf_counter() - conditional_wall
    conditional_elapsed_cpu = time.process_time() - conditional_cpu
    if not torch.isfinite(raw).all() or not torch.isfinite(contribution).all():
        raise FloatingPointError("V13 inferential contribution became nonfinite")

    audit_count = min(sample_count, reconstruction_paths)
    full_latent = sample.residual[:audit_count] + coordinate[:audit_count].unsqueeze(
        1
    ) * direction.unsqueeze(0)
    full_paths = problem.simulate_latent(full_latent)
    threshold_hard = coordinate[:audit_count] <= conditional.threshold[:audit_count]
    if not torch.equal(problem.hard_event(full_paths), threshold_hard):
        raise AssertionError("V13 full-path reconstruction disagrees with scalar threshold")
    coordinate_map = ResidualHouseholderCoordinates.build(direction)
    reconstructed = coordinate_map.from_coordinates(
        coordinate_map.to_coordinates(
            full_latent - (full_latent @ direction).unsqueeze(1) * direction
        )
    )
    full_error = float(torch.amax(torch.abs(reconstructed - sample.residual[:audit_count])))
    return StructuredECRPTEvaluationBatch(
        raw_contribution=raw,
        ecrpt_contribution=contribution,
        likelihood_normalization=sample.likelihood,
        threshold=conditional.threshold,
        labels=sample.labels,
        raw_cost=_evaluation_cost(
            problem,
            proposal,
            samples=sample_count,
            cdf_calls=0,
            wall_seconds=common_elapsed_wall + raw_elapsed_wall,
            cpu_seconds=common_elapsed_cpu + raw_elapsed_cpu,
        ),
        ecrpt_cost=_evaluation_cost(
            problem,
            proposal,
            samples=sample_count,
            cdf_calls=sample_count,
            wall_seconds=common_elapsed_wall + conditional_elapsed_wall,
            cpu_seconds=common_elapsed_cpu + conditional_elapsed_cpu,
        ),
        maximum_residual_projection_error=max(
            sample.maximum_projection_error, conditional.maximum_coordinate_error
        ),
        maximum_path_reconstruction_error=conditional.maximum_path_reconstruction_error,
        maximum_full_path_reconstruction_error=full_error,
        maximum_likelihood_bound_violation=sample.maximum_likelihood_bound_violation,
        hard_threshold_mismatch_count=conditional.hard_threshold_mismatch_count,
    )
