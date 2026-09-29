"""V14 exact tempered residual flow mixtures for finite-grid rBergomi events."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field

import torch

from src.path_integral.baseline_framework import BaselineCostLedger
from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.low_rank_residual_flow import (
    LowRankResidualFlowTrainingConfig,
    fit_low_rank_residual_flow,
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
from src.path_integral.tempered_residual_flow_mixture import (
    FrozenTemperedResidualFlowMixture,
    TemperedFlowComponent,
    freeze_tempered_residual_flow_mixture,
    sample_tempered_residual_flow_mixture,
)


def _derived_seed(root: int, role: str) -> int:
    if isinstance(root, bool) or not isinstance(root, int) or root < 0:
        raise ValueError("V14 root seed must be a nonnegative integer")
    digest = hashlib.sha256(f"NPI-V14-RBERGOMI\0{root}\0{role}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1) or 1


@dataclass(frozen=True)
class TemperedECRPTTrainingConfig:
    direction_samples: int = 512
    direction_rates: tuple[float, ...] = (-1.0, 0.0, 1.0)
    target_powers: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    mixture_weights: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
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
        if not self.direction_rates or any(not math.isfinite(x) for x in self.direction_rates):
            raise ValueError("direction rates must be nonempty and finite")
        if len(self.target_powers) != len(self.mixture_weights) or not self.target_powers:
            raise ValueError("one mixture weight is required per target power")
        if any(not math.isfinite(x) or not 0.0 < x <= 1.0 for x in self.target_powers):
            raise ValueError("target powers must lie in (0, 1]")
        if any(a >= b for a, b in zip(self.target_powers, self.target_powers[1:], strict=False)):
            raise ValueError("target powers must be strictly increasing")
        if any(not math.isfinite(x) or x <= 0.0 for x in self.mixture_weights):
            raise ValueError("mixture weights must be finite and positive")
        if not math.isclose(sum(self.mixture_weights), 1.0, rel_tol=1e-12, abs_tol=1e-14):
            raise ValueError("mixture weights must sum to one")


@dataclass(frozen=True)
class TemperedECRPTTrainingResult:
    proposal: FrozenTemperedResidualFlowMixture
    direction_selection: DirectionSelectionResult
    smc_results: tuple[AdaptiveResidualSMCResult, ...]
    target_powers: tuple[float, ...]
    training_seed: int
    all_training_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if any(result.particles_are_final_inferential_units for result in self.smc_results):
            raise ValueError("SMC populations may not be final inferential units")
        if any(result.final_beta != 1.0 for result in self.smc_results):
            raise ValueError("every tempered SMC run must end at internal beta one")
        if len(set(self.all_training_seeds)) != len(self.all_training_seeds):
            raise ValueError("V14 training seed streams collided")


def train_tempered_ecrpt(
    problem: RBergomiBaselineProblem,
    *,
    training_seed: int,
    config: TemperedECRPTTrainingConfig | None = None,
) -> TemperedECRPTTrainingResult:
    config = config or TemperedECRPTTrainingConfig()
    selection_seed = _derived_seed(training_seed, "direction-selection")
    selection_wall = time.perf_counter()
    selection_cpu = time.process_time()
    selection = select_rbergomi_direction(
        problem,
        sample_count=config.direction_samples,
        seed=selection_seed,
        exponential_rates=config.direction_rates,
    )
    elapsed_wall = time.perf_counter() - selection_wall
    elapsed_cpu = time.process_time() - selection_cpu
    direction = selection.selected_direction
    selection_evaluations = config.direction_samples * len(config.direction_rates)
    combined_cost = BaselineCostLedger(
        screening_samples=selection_evaluations,
        cdf_calls=selection_evaluations,
        algorithmic_work_units=float(
            selection_evaluations * (problem.latent_dimension + problem.steps + 1)
        ),
        wall_seconds=elapsed_wall,
        cpu_seconds=elapsed_cpu,
        peak_memory_bytes=process_peak_resident_memory_bytes(),
        measurement_mode="standardized_hardware_wall",
    )
    all_seeds = [training_seed, selection_seed]
    smc_results: list[AdaptiveResidualSMCResult] = []
    components: list[TemperedFlowComponent] = []
    for index, (power, weight) in enumerate(
        zip(config.target_powers, config.mixture_weights, strict=True)
    ):
        smc_seed = _derived_seed(training_seed, f"power-{index}-smc")
        flow_seed = _derived_seed(training_seed, f"power-{index}-flow")
        all_seeds.extend((smc_seed, flow_seed))

        def log_potential(residual: torch.Tensor, *, target_power: float = power) -> torch.Tensor:
            return (
                target_power
                * evaluate_rbergomi_conditional_residual(
                    problem, residual, direction
                ).log_conditional_value
            )

        smc = run_adaptive_residual_smc(
            direction=direction,
            log_potential_fn=log_potential,
            root_seed=smc_seed,
            config=config.smc,
            potential_work_per_particle=float(problem.latent_dimension + problem.steps + 1),
        )
        flow = fit_low_rank_residual_flow(
            task_id=problem.task_id,
            residual=smc.residual_particles,
            direction=direction,
            training_seed=flow_seed,
            config=config.flow,
        )
        smc_cost = smc.training_cost.plus(
            BaselineCostLedger(
                cdf_calls=smc.training_cost.training_samples,
                measurement_mode="standardized_hardware_wall",
            )
        )
        combined_cost = combined_cost.plus(smc_cost).plus(flow.training_cost)
        all_seeds.extend(smc.used_seeds)
        smc_results.append(smc)
        components.append(
            TemperedFlowComponent(
                target_power=power,
                mixture_weight=weight,
                flow=flow,
            )
        )
    if len(set(all_seeds)) != len(all_seeds):
        raise AssertionError("V14 derived training streams collided")
    proposal = freeze_tempered_residual_flow_mixture(
        task_id=problem.task_id,
        components=tuple(components),
        training_seed=training_seed,
        training_cost=combined_cost,
    )
    return TemperedECRPTTrainingResult(
        proposal=proposal,
        direction_selection=selection,
        smc_results=tuple(smc_results),
        target_powers=config.target_powers,
        training_seed=training_seed,
        all_training_seeds=tuple(all_seeds),
    )


@dataclass(frozen=True)
class TemperedECRPTEvaluationBatch:
    raw_contribution: torch.Tensor
    ecrpt_contribution: torch.Tensor
    conditional_probability: torch.Tensor
    likelihood_normalization: torch.Tensor
    threshold: torch.Tensor
    component_labels: torch.Tensor
    inner_flow_labels: torch.Tensor
    raw_cost: BaselineCostLedger
    ecrpt_cost: BaselineCostLedger
    maximum_residual_projection_error: float
    maximum_path_reconstruction_error: float
    maximum_full_path_reconstruction_error: float
    maximum_likelihood_bound_violation: float
    hard_threshold_mismatch_count: int


def _cost(
    problem: RBergomiBaselineProblem,
    proposal: FrozenTemperedResidualFlowMixture,
    *,
    samples: int,
    cdf_calls: int,
    wall_seconds: float,
    cpu_seconds: float,
) -> BaselineCostLedger:
    density_work = sum(
        sum(
            4 * layer.rank * (len(layer.active_indices) + len(layer.transformed_indices))
            for layer in component.flow.layers
        )
        for component in proposal.components
    )
    work = samples * (problem.latent_dimension + problem.steps + density_work + cdf_calls / samples)
    return BaselineCostLedger(
        final_samples=samples,
        likelihood_evaluations=samples * len(proposal.components),
        cdf_calls=cdf_calls,
        algorithmic_work_units=float(work),
        wall_seconds=wall_seconds,
        cpu_seconds=cpu_seconds,
        peak_memory_bytes=process_peak_resident_memory_bytes(),
        measurement_mode="standardized_hardware_wall",
    )


def evaluate_tempered_ecrpt(
    problem: RBergomiBaselineProblem,
    proposal: FrozenTemperedResidualFlowMixture,
    *,
    sample_count: int,
    proposal_seed: int,
    coordinate_seed: int,
    reconstruction_paths: int = 128,
) -> TemperedECRPTEvaluationBatch:
    if proposal.task_id != problem.task_id:
        raise ValueError("tempered proposal and task IDs differ")
    if proposal_seed == coordinate_seed:
        raise ValueError("proposal and coordinate seeds must be disjoint")
    direction = torch.tensor(proposal.direction, dtype=torch.float64)
    validate_rbergomi_residual_direction(problem, direction)
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    sample = sample_tempered_residual_flow_mixture(proposal, sample_count, root_seed=proposal_seed)
    conditional = evaluate_rbergomi_conditional_residual(problem, sample.residual, direction)
    common_wall = time.perf_counter() - started_wall
    common_cpu = time.process_time() - started_cpu
    coordinate = torch.randn(
        sample_count,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(coordinate_seed),
    )
    raw = (coordinate <= conditional.threshold).to(torch.float64) * sample.likelihood
    ecrpt = conditional.conditional_value * sample.likelihood
    if not torch.isfinite(raw).all() or not torch.isfinite(ecrpt).all():
        raise FloatingPointError("V14 contribution became nonfinite")
    audit_count = min(sample_count, reconstruction_paths)
    full_latent = sample.residual[:audit_count] + coordinate[:audit_count].unsqueeze(
        1
    ) * direction.unsqueeze(0)
    full_paths = problem.simulate_latent(full_latent)
    threshold_hard = coordinate[:audit_count] <= conditional.threshold[:audit_count]
    if not torch.equal(problem.hard_event(full_paths), threshold_hard):
        raise AssertionError("V14 full path disagrees with scalar threshold")
    mapping = ResidualHouseholderCoordinates.build(direction)
    rebuilt = mapping.from_coordinates(
        mapping.to_coordinates(full_latent - (full_latent @ direction).unsqueeze(1) * direction)
    )
    full_error = float(torch.amax(torch.abs(rebuilt - sample.residual[:audit_count])))
    return TemperedECRPTEvaluationBatch(
        raw_contribution=raw,
        ecrpt_contribution=ecrpt,
        conditional_probability=conditional.conditional_value,
        likelihood_normalization=sample.likelihood,
        threshold=conditional.threshold,
        component_labels=sample.component_labels,
        inner_flow_labels=sample.inner_flow_labels,
        raw_cost=_cost(
            problem,
            proposal,
            samples=sample_count,
            cdf_calls=0,
            wall_seconds=common_wall,
            cpu_seconds=common_cpu,
        ),
        ecrpt_cost=_cost(
            problem,
            proposal,
            samples=sample_count,
            cdf_calls=sample_count,
            wall_seconds=common_wall,
            cpu_seconds=common_cpu,
        ),
        maximum_residual_projection_error=max(
            sample.maximum_projection_error, conditional.maximum_coordinate_error
        ),
        maximum_path_reconstruction_error=conditional.maximum_path_reconstruction_error,
        maximum_full_path_reconstruction_error=full_error,
        maximum_likelihood_bound_violation=sample.maximum_likelihood_bound_violation,
        hard_threshold_mismatch_count=conditional.hard_threshold_mismatch_count,
    )
