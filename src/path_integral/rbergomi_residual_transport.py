"""Exact conditional residual transports for finite-grid rBergomi path events."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field

import torch

from src.path_integral.baseline_framework import BaselineCostLedger
from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.provenance import process_peak_resident_memory_bytes
from src.path_integral.rbergomi_dcs_mlmc import scalar_task_threshold
from src.path_integral.rbergomi_smoothing import affine_rbergomi_log_spot
from src.path_integral.residual_transport import (
    FrozenResidualTransport,
    ResidualTransportTrainingConfig,
    evaluate_residual_contribution,
    evaluate_residual_likelihood,
    fit_weighted_residual_transport,
    fit_weighted_residual_transport_from_importance_samples,
    freeze_residual_transport,
    project_orthogonal,
    sample_residual_mixture,
)


def _derived_seed(root: int, role: str) -> int:
    if isinstance(root, bool) or not isinstance(root, int) or root < 0:
        raise ValueError("root seed must be a nonnegative integer")
    digest = hashlib.sha256(f"NPI-V12-ECRPT\0{root}\0{role}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1) or 1


def embed_positive_price_direction(
    problem: RBergomiBaselineProblem,
    price_weights: torch.Tensor,
    *,
    tolerance: float = 1e-12,
) -> torch.Tensor:
    """Normalize positive price weights and embed them in the exact ``3N`` law."""

    if price_weights.shape != (problem.steps,):
        raise ValueError("price weights must have one entry per time step")
    if price_weights.device.type != "cpu" or price_weights.dtype != torch.float64:
        raise ValueError("price weights must be CPU float64")
    if not torch.isfinite(price_weights).all() or bool((price_weights <= 0.0).any()):
        raise ValueError("every price weight must be finite and strictly positive")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    price = price_weights / torch.linalg.vector_norm(price_weights)
    direction = torch.zeros(problem.latent_dimension, dtype=torch.float64)
    direction[problem.local_dimension :] = price
    validate_rbergomi_residual_direction(problem, direction, tolerance=tolerance)
    return direction


def validate_rbergomi_residual_direction(
    problem: RBergomiBaselineProblem,
    direction: torch.Tensor,
    *,
    tolerance: float = 1e-10,
) -> None:
    """Enforce the exact scalar-threshold support and positivity contract."""

    if direction.shape != (problem.latent_dimension,):
        raise ValueError("rBergomi direction has the wrong latent dimension")
    if direction.device.type != "cpu" or direction.dtype != torch.float64:
        raise ValueError("rBergomi direction must be CPU float64")
    if not torch.isfinite(direction).all():
        raise ValueError("rBergomi direction must be finite")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if abs(float(torch.linalg.vector_norm(direction)) - 1.0) > tolerance:
        raise ValueError("rBergomi direction must have unit norm")
    if float(torch.amax(torch.abs(direction[: problem.local_dimension]))) > tolerance:
        raise ValueError("integrated direction may not enter the volatility/local block")
    if bool((direction[problem.local_dimension :] <= 0.0).any()):
        raise ValueError("every integrated price coordinate must be strictly positive")


def positive_price_direction_candidates(
    problem: RBergomiBaselineProblem,
    *,
    exponential_rates: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0),
) -> tuple[torch.Tensor, ...]:
    """Return a fixed family of positive temporal directions for training selection."""

    if not exponential_rates:
        raise ValueError("at least one direction rate is required")
    if any(not math.isfinite(rate) for rate in exponential_rates):
        raise ValueError("direction rates must be finite")
    time_grid = (torch.arange(problem.steps, dtype=torch.float64) + 0.5) / problem.steps
    directions: list[torch.Tensor] = []
    for rate in exponential_rates:
        weights = torch.exp(rate * (time_grid - 0.5))
        directions.append(embed_positive_price_direction(problem, weights))
    return tuple(directions)


@dataclass(frozen=True)
class RBergomiConditionalResidualBatch:
    threshold: torch.Tensor
    log_conditional_value: torch.Tensor
    conditional_value: torch.Tensor
    maximum_coordinate_error: float
    maximum_path_reconstruction_error: float
    hard_threshold_mismatch_count: int


def evaluate_rbergomi_conditional_residual(
    problem: RBergomiBaselineProblem,
    residual: torch.Tensor,
    direction: torch.Tensor,
    *,
    tolerance: float = 1e-10,
) -> RBergomiConditionalResidualBatch:
    """Evaluate the exact target conditional event probability ``g(R)``."""

    validate_rbergomi_residual_direction(problem, direction, tolerance=tolerance)
    if residual.ndim != 2 or residual.shape[1] != problem.latent_dimension:
        raise ValueError("residual has the wrong shape")
    if residual.device.type != "cpu" or residual.dtype != torch.float64:
        raise ValueError("rBergomi residual must be CPU float64")
    if not torch.isfinite(residual).all():
        raise ValueError("rBergomi residual must be finite")
    projection = float(torch.amax(torch.abs(residual @ direction)))
    scale = max(1.0, float(torch.amax(torch.abs(residual))))
    if projection > tolerance * scale:
        raise ValueError("rBergomi residual is not orthogonal to its direction")
    paths = problem.simulate_latent(residual)
    increments = paths.target_brownian_increments
    if increments is None:
        raise ValueError("BLP simulator did not retain target Brownian increments")
    price_direction = direction[problem.local_dimension :]
    affine = affine_rbergomi_log_spot(
        spot=paths.spot,
        log_spot=paths.log_spot,
        variance=paths.variance,
        proposal_fine_brownian_increments=increments,
        fine_step_dt=problem.step_dt,
        rho=problem.rho,
        direction=price_direction,
    )
    threshold = scalar_task_threshold(
        affine.intercept,
        affine.slope,
        step_dt=paths.step_dt,
        task=problem.task,
    )
    if bool(torch.isnan(threshold).any()):
        raise FloatingPointError("conditional threshold contains NaN")
    hard = problem.hard_event(paths)
    threshold_event = affine.coordinate <= threshold
    mismatches = int(torch.count_nonzero(hard != threshold_event))
    if mismatches:
        raise AssertionError("residual scalar threshold is not pathwise exact")
    log_value = torch.special.log_ndtr(threshold)
    value = torch.exp(log_value)
    if not torch.isfinite(value).all() or bool((value < 0.0).any()) or bool((value > 1.0).any()):
        raise FloatingPointError("conditional event value is invalid")
    coordinate_error = max(
        projection,
        float(torch.amax(torch.abs(affine.coordinate))),
    )
    return RBergomiConditionalResidualBatch(
        threshold=threshold,
        log_conditional_value=log_value,
        conditional_value=value,
        maximum_coordinate_error=coordinate_error,
        maximum_path_reconstruction_error=affine.maximum_path_reconstruction_error,
        hard_threshold_mismatch_count=mismatches,
    )


@dataclass(frozen=True)
class DirectionSelectionResult:
    directions: tuple[tuple[float, ...], ...]
    conditional_second_moments: tuple[float, ...]
    conditional_means: tuple[float, ...]
    selected_index: int
    selection_seed: int
    sample_count: int

    @property
    def selected_direction(self) -> torch.Tensor:
        return torch.tensor(self.directions[self.selected_index], dtype=torch.float64)


def select_rbergomi_direction(
    problem: RBergomiBaselineProblem,
    *,
    sample_count: int,
    seed: int,
    exponential_rates: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0),
) -> DirectionSelectionResult:
    """Select a direction on training data by held-out conditional second moment."""

    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 2:
        raise ValueError("direction selection requires at least two samples")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("selection seed must be a nonnegative integer")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    target = torch.randn(
        (sample_count, problem.latent_dimension),
        dtype=torch.float64,
        generator=generator,
    )
    directions = positive_price_direction_candidates(
        problem, exponential_rates=exponential_rates
    )
    second_moments: list[float] = []
    means: list[float] = []
    for direction in directions:
        residual = project_orthogonal(target, direction)
        conditional = evaluate_rbergomi_conditional_residual(problem, residual, direction)
        second_moments.append(float(torch.mean(conditional.conditional_value.square())))
        means.append(float(torch.mean(conditional.conditional_value)))
    selected = min(range(len(second_moments)), key=lambda index: (second_moments[index], index))
    return DirectionSelectionResult(
        directions=tuple(tuple(float(value) for value in direction) for direction in directions),
        conditional_second_moments=tuple(second_moments),
        conditional_means=tuple(means),
        selected_index=selected,
        selection_seed=seed,
        sample_count=sample_count,
    )


@dataclass(frozen=True)
class RBergomiResidualTrainingConfig:
    direction_samples: int = 2048
    transport_samples: int = 4096
    direction_rates: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)
    transport: ResidualTransportTrainingConfig = field(
        default_factory=ResidualTransportTrainingConfig
    )
    adaptive_rounds: int = 0
    adaptive_samples_per_round: int = 2048
    tempering_powers: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.direction_samples, bool)
            or not isinstance(self.direction_samples, int)
            or self.direction_samples < 2
        ):
            raise ValueError("direction samples must be an integer of at least two")
        if (
            isinstance(self.transport_samples, bool)
            or not isinstance(self.transport_samples, int)
            or self.transport_samples < 2
        ):
            raise ValueError("transport samples must be an integer of at least two")
        if not self.direction_rates or any(not math.isfinite(x) for x in self.direction_rates):
            raise ValueError("direction rates must be a nonempty finite tuple")
        if (
            isinstance(self.adaptive_rounds, bool)
            or not isinstance(self.adaptive_rounds, int)
            or self.adaptive_rounds < 0
        ):
            raise ValueError("adaptive rounds must be a nonnegative integer")
        if (
            isinstance(self.adaptive_samples_per_round, bool)
            or not isinstance(self.adaptive_samples_per_round, int)
            or self.adaptive_samples_per_round < 2
        ):
            raise ValueError("adaptive samples per round must be at least two")
        if self.tempering_powers is not None:
            expected = self.adaptive_rounds + 1
            if len(self.tempering_powers) != expected:
                raise ValueError("tempering powers require one value per training round")
            if (
                any(not math.isfinite(value) or not 0.0 < value <= 1.0 for value in self.tempering_powers)
                or any(
                    a >= b
                    for a, b in zip(
                        self.tempering_powers,
                        self.tempering_powers[1:],
                        strict=False,
                    )
                )
                or self.tempering_powers[-1] != 1.0
            ):
                raise ValueError("tempering powers must increase strictly to one")

    @property
    def effective_tempering_powers(self) -> tuple[float, ...]:
        if self.tempering_powers is not None:
            return self.tempering_powers
        count = self.adaptive_rounds + 1
        return tuple((index + 1) / count for index in range(count))


@dataclass(frozen=True)
class RBergomiResidualTrainingResult:
    proposal: FrozenResidualTransport
    direction_selection: DirectionSelectionResult
    loss_history: tuple[float, ...]
    normalized_weight_ess: float
    finite_target_count: int
    selection_seed: int
    target_seed: int
    optimizer_seed: int
    round_normalized_weight_ess: tuple[float, ...]
    round_finite_target_counts: tuple[int, ...]
    tempering_powers: tuple[float, ...]


def train_rbergomi_residual_transport(
    problem: RBergomiBaselineProblem,
    *,
    training_seed: int,
    config: RBergomiResidualTrainingConfig | None = None,
) -> RBergomiResidualTrainingResult:
    """Select a positive direction and fit the exact residual proposal."""

    config = config or RBergomiResidualTrainingConfig()
    selection_seed = _derived_seed(training_seed, "direction-selection")
    target_seed = _derived_seed(training_seed, "residual-target")
    optimizer_seed = _derived_seed(training_seed, "optimizer")
    if len({training_seed, selection_seed, target_seed, optimizer_seed}) != 4:
        raise AssertionError("derived training streams collided")
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    selection = select_rbergomi_direction(
        problem,
        sample_count=config.direction_samples,
        seed=selection_seed,
        exponential_rates=config.direction_rates,
    )
    direction = selection.selected_direction
    tempering_powers = config.effective_tempering_powers
    target = torch.randn(
        (config.transport_samples, problem.latent_dimension),
        dtype=torch.float64,
        generator=torch.Generator(device="cpu").manual_seed(target_seed),
    )
    residual = project_orthogonal(target, direction)
    conditional = evaluate_rbergomi_conditional_residual(problem, residual, direction)
    fitted = fit_weighted_residual_transport(
        task_id=problem.task_id,
        target_residuals=residual,
        log_target_weights=tempering_powers[0] * conditional.log_conditional_value,
        direction=direction,
        training_seed=optimizer_seed,
        config=config.transport,
    )
    current = fitted
    round_ess = [fitted.normalized_weight_ess]
    round_finite = [fitted.finite_target_count]
    adaptive_sampling_cost = BaselineCostLedger()
    for round_index in range(config.adaptive_rounds):
        gaussian_seed = _derived_seed(training_seed, f"adaptive-{round_index}-gaussian")
        label_seed = _derived_seed(training_seed, f"adaptive-{round_index}-label")
        round_optimizer_seed = _derived_seed(
            training_seed, f"adaptive-{round_index}-optimizer"
        )
        if len({gaussian_seed, label_seed, round_optimizer_seed}) != 3:
            raise AssertionError("adaptive residual streams collided")
        spec = current.proposal.spec()
        sample = sample_residual_mixture(
            spec,
            config.adaptive_samples_per_round,
            gaussian_generator=torch.Generator(device="cpu").manual_seed(gaussian_seed),
            label_generator=torch.Generator(device="cpu").manual_seed(label_seed),
        )
        adaptive_conditional = evaluate_rbergomi_conditional_residual(
            problem, sample.residual, direction
        )
        adaptive_likelihood = evaluate_residual_likelihood(sample.residual, spec)
        log_target_over_sampling = (
            tempering_powers[round_index + 1]
            * adaptive_conditional.log_conditional_value
            + adaptive_likelihood.log_likelihood
        )
        next_fit = fit_weighted_residual_transport_from_importance_samples(
            task_id=problem.task_id,
            sampled_residuals=sample.residual,
            log_unnormalized_target_over_sampling=log_target_over_sampling,
            direction=direction,
            training_seed=round_optimizer_seed,
            config=config.transport,
            training_objective="adaptive_exact_importance_weighted_conditional_cross_entropy",
        )
        density_work = (
            config.adaptive_samples_per_round * spec.components * problem.latent_dimension
        )
        simulation_work = config.adaptive_samples_per_round * (
            problem.latent_dimension + problem.steps + 1
        )
        adaptive_sampling_cost = adaptive_sampling_cost.plus(
            BaselineCostLedger(
                likelihood_evaluations=config.adaptive_samples_per_round,
                cdf_calls=config.adaptive_samples_per_round,
                algorithmic_work_units=float(density_work + simulation_work),
            )
        )
        current = next_fit
        round_ess.append(next_fit.normalized_weight_ess)
        round_finite.append(next_fit.finite_target_count)
    fit_cost = fitted.proposal.training_cost.plus(adaptive_sampling_cost)
    for _ in range(config.adaptive_rounds):
        # Each adaptive optimizer uses the same declared architecture and budget.
        # Its measured ledger is represented by the corresponding fitted proposal;
        # reconstruct the deterministic algorithmic count for earlier overwritten
        # rounds so no hyperparameter/training work disappears.
        fit_cost = fit_cost.plus(
            BaselineCostLedger(
                training_samples=config.adaptive_samples_per_round,
                optimizer_steps=config.transport.epochs,
                hyperparameter_trials=1,
                algorithmic_work_units=float(
                    config.adaptive_samples_per_round
                    * config.transport.epochs
                    * (config.transport.components - 1)
                    * problem.latent_dimension
                ),
            )
        )
    selection_paths = config.direction_samples * len(config.direction_rates)
    simulated_paths = selection_paths + config.transport_samples
    simulation_work = simulated_paths * (problem.latent_dimension + problem.steps)
    combined_cost = BaselineCostLedger(
        training_samples=simulated_paths,
        optimizer_steps=fit_cost.optimizer_steps,
        hyperparameter_trials=1,
        algorithmic_work_units=fit_cost.algorithmic_work_units + simulation_work,
        wall_seconds=time.perf_counter() - wall_started,
        cpu_seconds=time.process_time() - cpu_started,
        peak_memory_bytes=max(
            fit_cost.peak_memory_bytes, process_peak_resident_memory_bytes()
        ),
        measurement_mode="standardized_hardware_wall",
    )
    proposal = freeze_residual_transport(
        task_id=problem.task_id,
        spec=current.proposal.spec(),
        training_seed=training_seed,
        training_objective="direction_selected_annealed_conditional_target_cross_entropy",
        training_cost=combined_cost,
    )
    return RBergomiResidualTrainingResult(
        proposal=proposal,
        direction_selection=selection,
        loss_history=current.loss_history,
        normalized_weight_ess=fitted.normalized_weight_ess,
        finite_target_count=fitted.finite_target_count,
        selection_seed=selection_seed,
        target_seed=target_seed,
        optimizer_seed=optimizer_seed,
        round_normalized_weight_ess=tuple(round_ess),
        round_finite_target_counts=tuple(round_finite),
        tempering_powers=tempering_powers,
    )


@dataclass(frozen=True)
class RBergomiResidualEvaluationBatch:
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


def _evaluation_cost(
    problem: RBergomiBaselineProblem,
    *,
    samples: int,
    components: int,
    cdf_calls: int,
    wall_seconds: float,
    cpu_seconds: float,
) -> BaselineCostLedger:
    simulation_work = samples * (problem.latent_dimension + problem.steps)
    density_work = samples * components * problem.latent_dimension
    return BaselineCostLedger(
        final_samples=samples,
        likelihood_evaluations=samples,
        cdf_calls=cdf_calls,
        algorithmic_work_units=float(simulation_work + density_work + cdf_calls),
        wall_seconds=wall_seconds,
        cpu_seconds=cpu_seconds,
        peak_memory_bytes=process_peak_resident_memory_bytes(),
        measurement_mode="standardized_hardware_wall",
    )


def evaluate_rbergomi_residual_transport(
    problem: RBergomiBaselineProblem,
    proposal: FrozenResidualTransport,
    *,
    sample_count: int,
    gaussian_seed: int,
    label_seed: int,
    coordinate_seed: int,
    reconstruction_paths: int = 128,
) -> RBergomiResidualEvaluationBatch:
    """Evaluate paired raw and exact-conditional contributions on one proposal."""

    if proposal.task_id != problem.task_id:
        raise ValueError("residual proposal and problem task IDs differ")
    if not proposal.exact_likelihood or proposal.self_normalized or not proposal.frozen:
        raise ValueError("ECRPT requires a frozen exact ordinary proposal")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 2:
        raise ValueError("sample count must be an integer of at least two")
    seeds = (gaussian_seed, label_seed, coordinate_seed)
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("evaluation seeds must be nonnegative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Gaussian, label, and coordinate seeds must be disjoint")
    if (
        isinstance(reconstruction_paths, bool)
        or not isinstance(reconstruction_paths, int)
        or reconstruction_paths < 1
    ):
        raise ValueError("reconstruction paths must be positive")
    spec = proposal.spec()
    validate_rbergomi_residual_direction(problem, spec.direction)
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    sample = sample_residual_mixture(
        spec,
        sample_count,
        gaussian_generator=torch.Generator(device="cpu").manual_seed(gaussian_seed),
        label_generator=torch.Generator(device="cpu").manual_seed(label_seed),
    )
    conditional = evaluate_rbergomi_conditional_residual(
        problem, sample.residual, spec.direction
    )
    likelihood = evaluate_residual_likelihood(sample.residual, spec)
    common_cpu = time.process_time() - cpu_started
    common_wall = time.perf_counter() - wall_started

    raw_started_wall = time.perf_counter()
    raw_started_cpu = time.process_time()
    coordinate = torch.randn(
        sample_count,
        dtype=torch.float64,
        generator=torch.Generator(device="cpu").manual_seed(coordinate_seed),
    )
    raw = (coordinate <= conditional.threshold).to(torch.float64) * likelihood.likelihood
    raw_cpu = time.process_time() - raw_started_cpu
    raw_wall = time.perf_counter() - raw_started_wall

    dcs_started_wall = time.perf_counter()
    dcs_started_cpu = time.process_time()
    evaluated = evaluate_residual_contribution(
        sample.residual,
        spec,
        log_conditional_value=conditional.log_conditional_value,
    )
    dcs_cpu = time.process_time() - dcs_started_cpu
    dcs_wall = time.perf_counter() - dcs_started_wall

    audit_count = min(sample_count, reconstruction_paths)
    full_latent = (
        sample.residual[:audit_count]
        + coordinate[:audit_count].unsqueeze(1) * spec.direction.unsqueeze(0)
    )
    full_paths = problem.simulate_latent(full_latent)
    full_hard = problem.hard_event(full_paths)
    threshold_hard = coordinate[:audit_count] <= conditional.threshold[:audit_count]
    if not torch.equal(full_hard, threshold_hard):
        raise AssertionError("reconstructed full path disagrees with scalar threshold")
    reconstructed_residual = project_orthogonal(full_latent, spec.direction)
    full_reconstruction_error = float(
        torch.amax(torch.abs(reconstructed_residual - sample.residual[:audit_count]))
    )
    return RBergomiResidualEvaluationBatch(
        raw_contribution=raw,
        ecrpt_contribution=evaluated.contribution,
        likelihood_normalization=likelihood.likelihood,
        threshold=conditional.threshold,
        labels=sample.labels,
        raw_cost=_evaluation_cost(
            problem,
            samples=sample_count,
            components=spec.components,
            cdf_calls=0,
            wall_seconds=common_wall + raw_wall,
            cpu_seconds=common_cpu + raw_cpu,
        ),
        ecrpt_cost=_evaluation_cost(
            problem,
            samples=sample_count,
            components=spec.components,
            cdf_calls=sample_count,
            wall_seconds=common_wall + dcs_wall,
            cpu_seconds=common_cpu + dcs_cpu,
        ),
        maximum_residual_projection_error=max(
            sample.maximum_projection_error,
            evaluated.likelihood.maximum_projection_error,
            conditional.maximum_coordinate_error,
        ),
        maximum_path_reconstruction_error=conditional.maximum_path_reconstruction_error,
        maximum_full_path_reconstruction_error=full_reconstruction_error,
        maximum_likelihood_bound_violation=evaluated.likelihood.maximum_bound_violation,
        hard_threshold_mismatch_count=conditional.hard_threshold_mismatch_count,
    )
