"""Fit an exact defensive Gaussian mixture to conditional tempered-SMC particles."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from src.path_integral.baseline_framework import BaselineCostLedger
from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.blp_cameron_martin_embedding import (
    build_mesh_compatible_blp_trace_safety_geometry,
)
from src.path_integral.cameron_martin_basis import CameronMartinBasis
from src.path_integral.finite_rank_gaussian_transport import (
    DefensiveFiniteRankGaussianMixture,
    FiniteRankGaussianComponent,
    build_trace_class_small_noise_safety_component,
)
from src.path_integral.provenance import process_peak_resident_memory_bytes
from src.path_integral.tempered_conditional_smc import (
    TemperedSMCConfig,
    estimate_tempered_normalizer,
)
from src.path_integral.volterra_conditional_payoffs import (
    evaluate_rbergomi_conditional_terminal,
)


@dataclass(frozen=True)
class TemperedTargetTransportConfig:
    smc: TemperedSMCConfig
    defensive_mass: float = 0.15
    safety_mass: float = 0.02
    components: int = 3
    minimum_variance: float = 0.05
    maximum_variance: float = 20.0
    covariance_ridge: float = 1e-6

    def __post_init__(self) -> None:
        if not self.smc.retain_final_particles:
            raise ValueError("tempered target transport must retain final SMC particles")
        if not 0.0 < self.defensive_mass < 1.0:
            raise ValueError("defensive mass must lie in (0,1)")
        if not 0.0 < self.safety_mass < 1.0 - self.defensive_mass:
            raise ValueError("safety mass must be positive and leave adaptive mass")
        if isinstance(self.components, bool) or not isinstance(self.components, int):
            raise ValueError("components must be an integer")
        if self.components < 1:
            raise ValueError("components must be positive")
        if not 0.0 < self.minimum_variance <= self.maximum_variance:
            raise ValueError("variance bounds are invalid")
        if not math.isfinite(self.maximum_variance):
            raise ValueError("maximum variance must be finite")
        if not math.isfinite(self.covariance_ridge) or self.covariance_ridge <= 0.0:
            raise ValueError("covariance ridge must be finite and positive")


@dataclass(frozen=True)
class TemperedTargetTransportResult:
    proposal: DefensiveFiniteRankGaussianMixture
    training_cost: BaselineCostLedger
    normalizer_estimate: float
    normalizer_standard_error: float
    minimum_incremental_ess_fraction: float
    mutation_acceptance_rate: float
    fitted_particle_count: int


def _fit_component(
    coefficients: torch.Tensor,
    basis: CameronMartinBasis,
    config: TemperedTargetTransportConfig,
) -> FiniteRankGaussianComponent:
    mean = torch.mean(coefficients, dim=0)
    centered = coefficients - mean
    covariance = centered.T @ centered / coefficients.shape[0]
    covariance = covariance + config.covariance_ridge * torch.eye(
        basis.rank,
        dtype=torch.float64,
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = torch.clamp(
        eigenvalues,
        min=config.minimum_variance,
        max=config.maximum_variance,
    )
    return FiniteRankGaussianComponent(
        mean=basis.expand(mean),
        directions=basis.matrix @ eigenvectors,
        variance_eigenvalues=eigenvalues,
    )


def fit_tempered_target_transport(
    problem: RBergomiBaselineProblem,
    basis: CameronMartinBasis,
    *,
    config: TemperedTargetTransportConfig,
) -> TemperedTargetTransportResult:
    """Approximate ``Pi*(dx) proportional to g(x)P(dx)``; keep final IS exact."""

    if basis.dimension != problem.local_dimension:
        raise ValueError("tempered target basis and problem dimensions differ")

    def log_potential(points: torch.Tensor) -> torch.Tensor:
        return evaluate_rbergomi_conditional_terminal(
            problem,
            points,
        ).payoffs.log_left_probability

    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    smc = estimate_tempered_normalizer(
        log_potential,
        dimension=problem.local_dimension,
        config=config.smc,
    )
    if smc.final_particles is None:
        raise RuntimeError("tempered SMC did not return its final target particles")
    coefficients = basis.project(smc.final_particles)
    if coefficients.shape[0] < 2 * config.components:
        raise RuntimeError("too few final particles for the requested mixture")
    if config.components == 1:
        labels = torch.zeros(coefficients.shape[0], dtype=torch.int64)
    else:
        centered = coefficients - torch.mean(coefficients, dim=0)
        covariance = centered.T @ centered / coefficients.shape[0]
        _, eigenvectors = torch.linalg.eigh(covariance)
        score = centered @ eigenvectors[:, -1]
        order = torch.argsort(score)
        ordered_labels = torch.div(
            torch.arange(coefficients.shape[0]) * config.components,
            coefficients.shape[0],
            rounding_mode="floor",
        )
        labels = torch.empty_like(ordered_labels)
        labels[order] = ordered_labels
    adaptive_components = []
    adaptive_weights = []
    for label in range(config.components):
        selected = labels == label
        count = int(torch.sum(selected))
        if count < 2:
            raise RuntimeError("tempered target split produced an empty component")
        adaptive_components.append(_fit_component(coefficients[selected], basis, config))
        adaptive_weights.append(count / coefficients.shape[0])
    directions, spectrum = build_mesh_compatible_blp_trace_safety_geometry(
        steps=problem.steps,
        maturity=problem.maturity,
        hurst=problem.hurst,
        spectrum_decay=2.0,
        spectrum_scale=4.0,
        complement_decay=2.0,
    )
    safety = build_trace_class_small_noise_safety_component(
        directions,
        spectrum,
        epsilon=1.0,
    )
    adaptive_mass = 1.0 - config.defensive_mass - config.safety_mass
    proposal = DefensiveFiniteRankGaussianMixture(
        components=(
            FiniteRankGaussianComponent.natural(problem.local_dimension),
            safety,
            *adaptive_components,
        ),
        weights=torch.tensor(
            [
                config.defensive_mass,
                config.safety_mass,
                *(adaptive_mass * value for value in adaptive_weights),
            ],
            dtype=torch.float64,
        ),
    )
    work = smc.potential_evaluations * (problem.local_dimension + problem.steps)
    work += coefficients.shape[0] * basis.rank**2
    cost = BaselineCostLedger(
        training_samples=smc.potential_evaluations,
        optimizer_steps=len(config.smc.temperatures) - 1,
        hyperparameter_trials=1,
        cdf_calls=smc.potential_evaluations,
        algorithmic_work_units=float(work),
        wall_seconds=time.perf_counter() - started_wall,
        cpu_seconds=time.process_time() - started_cpu,
        peak_memory_bytes=process_peak_resident_memory_bytes(),
        measurement_mode="standardized_hardware_wall",
    )
    return TemperedTargetTransportResult(
        proposal=proposal,
        training_cost=cost,
        normalizer_estimate=smc.mean,
        normalizer_standard_error=smc.standard_error,
        minimum_incremental_ess_fraction=smc.minimum_incremental_ess_fraction,
        mutation_acceptance_rate=smc.mutation_acceptance_rate,
        fitted_particle_count=coefficients.shape[0],
    )
