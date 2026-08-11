"""Conditional cross-entropy adaptation of an exact Gaussian transport component."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from src.path_integral.baseline_framework import BaselineCostLedger
from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.cameron_martin_basis import CameronMartinBasis
from src.path_integral.finite_rank_gaussian_transport import (
    DefensiveFiniteRankGaussianMixture,
    FiniteRankGaussianComponent,
)
from src.path_integral.provenance import process_peak_resident_memory_bytes
from src.path_integral.volterra_conditional_payoffs import (
    evaluate_rbergomi_conditional_terminal,
)


@dataclass(frozen=True)
class ConditionalTransportAdaptationConfig:
    iterations: int = 3
    samples_per_iteration: int = 2048
    smoothing: float = 0.7
    minimum_ess_fraction: float = 0.1
    minimum_variance: float = 0.05
    maximum_variance: float = 20.0
    covariance_ridge: float = 1e-6

    def __post_init__(self) -> None:
        if isinstance(self.iterations, bool) or not isinstance(self.iterations, int):
            raise ValueError("iterations must be an integer")
        if self.iterations < 1:
            raise ValueError("iterations must be positive")
        if (
            isinstance(self.samples_per_iteration, bool)
            or not isinstance(self.samples_per_iteration, int)
            or self.samples_per_iteration < 2
        ):
            raise ValueError("samples_per_iteration must be an integer of at least two")
        if not 0.0 < self.smoothing <= 1.0:
            raise ValueError("smoothing must lie in (0,1]")
        if not 0.0 < self.minimum_ess_fraction <= 1.0:
            raise ValueError("minimum_ess_fraction must lie in (0,1]")
        if not 0.0 < self.minimum_variance <= self.maximum_variance:
            raise ValueError("adaptation variance bounds are invalid")
        if not math.isfinite(self.maximum_variance):
            raise ValueError("maximum_variance must be finite")
        if not math.isfinite(self.covariance_ridge) or self.covariance_ridge <= 0.0:
            raise ValueError("covariance_ridge must be finite and positive")


@dataclass(frozen=True)
class ConditionalTransportAdaptationResult:
    proposal: DefensiveFiniteRankGaussianMixture
    training_cost: BaselineCostLedger
    effective_sample_sizes: tuple[float, ...]
    tempering_powers: tuple[float, ...]


def _normalized_tempered_weights(
    log_weights: torch.Tensor,
    *,
    minimum_ess: float,
) -> tuple[torch.Tensor, float, float]:
    centered = log_weights - torch.max(log_weights)

    def at(power: float) -> tuple[torch.Tensor, float]:
        weights = torch.softmax(power * centered, dim=0)
        ess = 1.0 / float(torch.sum(weights.square()))
        return weights, ess

    full, full_ess = at(1.0)
    if full_ess >= minimum_ess:
        return full, full_ess, 1.0
    lower = 0.0
    upper = 1.0
    for _ in range(50):
        middle = 0.5 * (lower + upper)
        _, ess = at(middle)
        if ess >= minimum_ess:
            lower = middle
        else:
            upper = middle
    weights, ess = at(lower)
    return weights, ess, lower


def _basis_covariance(
    component: FiniteRankGaussianComponent,
    basis: CameronMartinBasis,
) -> torch.Tensor:
    projected = basis.matrix.T @ component.directions
    correction = projected @ (
        torch.diag(component.variance_eigenvalues - 1.0) @ projected.T
    )
    return torch.eye(basis.rank, dtype=torch.float64) + correction


def adapt_conditional_transport(
    problem: RBergomiBaselineProblem,
    basis: CameronMartinBasis,
    initial: DefensiveFiniteRankGaussianMixture,
    *,
    epsilon: float,
    seed: int,
    config: ConditionalTransportAdaptationConfig | None = None,
) -> ConditionalTransportAdaptationResult:
    """Fit one conditional-target component; final sampling remains ordinary IS."""

    config = config or ConditionalTransportAdaptationConfig()
    if initial.dimension != problem.local_dimension or basis.dimension != initial.dimension:
        raise ValueError("adaptation dimensions do not match the problem")
    if not math.isfinite(epsilon) or not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must lie in (0,1]")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    fixed_count = 1
    if len(initial.components) >= 3 and initial.components[1].rank == initial.dimension:
        fixed_count = 2
    fixed_components = initial.components[:fixed_count]
    fixed_weights = initial.weights[:fixed_count]
    adaptive_mass = 1.0 - float(torch.sum(fixed_weights))
    if adaptive_mass <= 0.0:
        raise ValueError("initial proposal leaves no mass for adaptation")
    shifted_index = fixed_count + int(torch.argmax(initial.weights[fixed_count:]))
    component = initial.components[shifted_index]
    coefficient_mean = basis.project(component.mean)
    coefficient_covariance = _basis_covariance(component, basis)
    proposal = initial
    generator = torch.Generator().manual_seed(seed)
    effective_sample_sizes: list[float] = []
    powers: list[float] = []
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    for _iteration in range(config.iterations):
        draw = proposal.sample(
            config.samples_per_iteration,
            path_seed=int(torch.randint(1, 2**62, (), generator=generator)),
            label_seed=int(torch.randint(1, 2**62, (), generator=generator)),
        )
        conditional = evaluate_rbergomi_conditional_terminal(
            problem,
            draw.samples,
            epsilon=epsilon,
        )
        log_target_over_q = conditional.payoffs.log_left_probability + draw.log_p_over_q
        weights, ess, power = _normalized_tempered_weights(
            log_target_over_q,
            minimum_ess=config.minimum_ess_fraction * config.samples_per_iteration,
        )
        coefficients = basis.project(draw.samples)
        fitted_mean = torch.sum(weights[:, None] * coefficients, dim=0)
        centered = coefficients - fitted_mean
        fitted_covariance = centered.T @ (weights[:, None] * centered)
        fitted_covariance = fitted_covariance + config.covariance_ridge * torch.eye(
            basis.rank,
            dtype=torch.float64,
        )
        coefficient_mean = (
            (1.0 - config.smoothing) * coefficient_mean
            + config.smoothing * fitted_mean
        )
        coefficient_covariance = (
            (1.0 - config.smoothing) * coefficient_covariance
            + config.smoothing * fitted_covariance
        )
        eigenvalues, eigenvectors = torch.linalg.eigh(coefficient_covariance)
        eigenvalues = torch.clamp(
            eigenvalues,
            min=config.minimum_variance,
            max=config.maximum_variance,
        )
        coefficient_covariance = eigenvectors @ (
            torch.diag(eigenvalues) @ eigenvectors.T
        )
        component = FiniteRankGaussianComponent(
            mean=basis.expand(coefficient_mean),
            directions=basis.matrix @ eigenvectors,
            variance_eigenvalues=eigenvalues,
        )
        proposal = DefensiveFiniteRankGaussianMixture(
            components=(*fixed_components, component),
            weights=torch.cat(
                (fixed_weights, torch.tensor([adaptive_mass], dtype=torch.float64))
            ),
        )
        effective_sample_sizes.append(ess)
        powers.append(power)
    training_samples = config.iterations * config.samples_per_iteration
    work = training_samples * (
        problem.local_dimension + problem.steps + 4 * basis.rank**2
    )
    cost = BaselineCostLedger(
        training_samples=training_samples,
        optimizer_steps=config.iterations,
        hyperparameter_trials=1,
        algorithmic_work_units=float(work),
        wall_seconds=time.perf_counter() - started_wall,
        cpu_seconds=time.process_time() - started_cpu,
        peak_memory_bytes=process_peak_resident_memory_bytes(),
        measurement_mode="standardized_hardware_wall",
    )
    return ConditionalTransportAdaptationResult(
        proposal=proposal,
        training_cost=cost,
        effective_sample_sizes=tuple(effective_sample_sizes),
        tempering_powers=tuple(powers),
    )
