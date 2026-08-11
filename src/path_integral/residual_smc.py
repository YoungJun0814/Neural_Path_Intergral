"""ESS-adaptive annealed SMC on a Gaussian residual hyperplane."""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from src.path_integral.baseline_framework import BaselineCostLedger
from src.path_integral.provenance import process_peak_resident_memory_bytes
from src.path_integral.residual_coupling_transport import ResidualHouseholderCoordinates

LogPotential = Callable[[torch.Tensor], torch.Tensor]


def _derived_seed(root: int, role: str) -> int:
    if isinstance(root, bool) or not isinstance(root, int) or root < 0:
        raise ValueError("SMC root seed must be a nonnegative integer")
    digest = hashlib.sha256(f"NPI-V13-SMC\0{root}\0{role}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1) or 1


@dataclass(frozen=True)
class AdaptiveResidualSMCConfig:
    particles: int = 1024
    target_ess_fraction: float = 0.7
    pcn_scale: float = 0.25
    pcn_sweeps_per_stage: int = 2
    beta_bisection_steps: int = 50
    beta_tolerance: float = 1e-10
    maximum_stages: int = 128

    def __post_init__(self) -> None:
        integers = (
            self.particles,
            self.pcn_sweeps_per_stage,
            self.beta_bisection_steps,
            self.maximum_stages,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in integers
        ):
            raise ValueError("SMC counts must be positive integers")
        if self.particles < 2:
            raise ValueError("SMC requires at least two particles")
        if not 0.0 < self.target_ess_fraction < 1.0:
            raise ValueError("SMC target ESS fraction must lie in (0, 1)")
        if not 0.0 < self.pcn_scale <= 1.0:
            raise ValueError("pCN scale must lie in (0, 1]")
        if not math.isfinite(self.beta_tolerance) or self.beta_tolerance <= 0.0:
            raise ValueError("beta tolerance must be finite and positive")


@dataclass(frozen=True)
class SMCStageDiagnostics:
    stage: int
    beta_previous: float
    beta_next: float
    incremental_ess: float
    target_ess: float
    ess_target_met: bool
    log_normalizer_increment: float
    pcn_attempts: int
    pcn_accepts: int
    pcn_acceptance_rate: float


@dataclass(frozen=True)
class AdaptiveResidualSMCResult:
    residual_particles: torch.Tensor
    log_potential: torch.Tensor
    stages: tuple[SMCStageDiagnostics, ...]
    root_seed: int
    used_seeds: tuple[int, ...]
    training_cost: BaselineCostLedger
    final_beta: float
    final_particles_equally_weighted: bool
    particles_are_final_inferential_units: bool


def _normalized_weights_and_ess(log_weights: torch.Tensor) -> tuple[torch.Tensor, float]:
    if log_weights.ndim != 1 or bool(torch.isnan(log_weights).any()):
        raise ValueError("SMC log weights are invalid")
    finite = torch.isfinite(log_weights)
    if not bool(finite.any()):
        raise FloatingPointError("every SMC incremental weight is zero")
    normalized = torch.softmax(log_weights, dim=0)
    ess = 1.0 / float(torch.sum(normalized.square()))
    if not math.isfinite(ess) or not 1.0 <= ess <= log_weights.numel() * (1.0 + 1e-12):
        raise FloatingPointError("SMC ESS became invalid")
    return normalized, min(float(log_weights.numel()), ess)


def _incremental_ess(log_potential: torch.Tensor, delta_beta: float) -> float:
    return _normalized_weights_and_ess(delta_beta * log_potential)[1]


def _next_beta(
    beta: float,
    log_potential: torch.Tensor,
    *,
    target_ess: float,
    bisection_steps: int,
    tolerance: float,
) -> float:
    if not 0.0 <= beta < 1.0:
        raise ValueError("adaptive beta must lie in [0, 1)")
    if _incremental_ess(log_potential, 1.0 - beta) >= target_ess:
        return 1.0
    low, high = beta, 1.0
    for _ in range(bisection_steps):
        middle = 0.5 * (low + high)
        ess = _incremental_ess(log_potential, middle - beta)
        if ess >= target_ess:
            low = middle
        else:
            high = middle
        if high - low <= tolerance:
            break
    if low <= beta or low - beta <= math.ulp(max(1.0, beta)):
        raise FloatingPointError("adaptive beta schedule stagnated")
    return low


def _systematic_resample(weights: torch.Tensor, *, seed: int) -> torch.Tensor:
    if weights.ndim != 1 or bool((weights < 0.0).any()) or not torch.isfinite(weights).all():
        raise ValueError("systematic resampling weights are invalid")
    total = float(torch.sum(weights))
    if not math.isclose(total, 1.0, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("systematic resampling weights must sum to one")
    count = weights.numel()
    generator = torch.Generator().manual_seed(seed)
    offset = float(torch.rand((), dtype=torch.float64, generator=generator)) / count
    positions = offset + torch.arange(count, dtype=torch.float64) / count
    cumulative = torch.cumsum(weights, dim=0)
    cumulative[-1] = 1.0
    indices = torch.searchsorted(cumulative, positions, right=False)
    if indices.shape != (count,) or bool((indices < 0).any()) or bool((indices >= count).any()):
        raise AssertionError("systematic resampling returned invalid ancestors")
    return indices


def run_adaptive_residual_smc(
    *,
    direction: torch.Tensor,
    log_potential_fn: LogPotential,
    root_seed: int,
    config: AdaptiveResidualSMCConfig | None = None,
    potential_work_per_particle: float = 1.0,
) -> AdaptiveResidualSMCResult:
    """Approximate ``g(r) p_R(r)`` for training; never returns final IID units."""

    config = config or AdaptiveResidualSMCConfig()
    if not math.isfinite(potential_work_per_particle) or potential_work_per_particle <= 0.0:
        raise ValueError("potential work per particle must be finite and positive")
    coordinate_map = ResidualHouseholderCoordinates.build(direction)
    used_seeds: list[int] = []

    def seed(role: str) -> int:
        value = _derived_seed(root_seed, role)
        if value in used_seeds:
            raise AssertionError("SMC derived seeds collided")
        used_seeds.append(value)
        return value

    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    initial_seed = seed("initial-particles")
    coordinates = torch.randn(
        (config.particles, coordinate_map.coordinate_dimension),
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(initial_seed),
    )
    residual = coordinate_map.from_coordinates(coordinates)
    log_potential = log_potential_fn(residual)
    if log_potential.shape != (config.particles,) or bool(torch.isnan(log_potential).any()):
        raise ValueError("SMC log potential must return one non-NaN value per particle")
    if bool((log_potential > 1e-12).any()):
        raise ValueError("conditional log potential must not exceed zero")
    potential_evaluations = config.particles
    beta = 0.0
    stages: list[SMCStageDiagnostics] = []
    log_normalizer = 0.0
    target_ess = config.target_ess_fraction * config.particles
    pcn_reference_scale = math.sqrt(1.0 - config.pcn_scale**2)
    while beta < 1.0:
        if len(stages) >= config.maximum_stages:
            raise RuntimeError("adaptive SMC exceeded its maximum stage count")
        next_beta = _next_beta(
            beta,
            log_potential,
            target_ess=target_ess,
            bisection_steps=config.beta_bisection_steps,
            tolerance=config.beta_tolerance,
        )
        delta = next_beta - beta
        log_incremental = delta * log_potential
        weights, ess = _normalized_weights_and_ess(log_incremental)
        log_increment = float(torch.logsumexp(log_incremental, dim=0)) - math.log(config.particles)
        log_normalizer += log_increment
        resample_seed = seed(f"stage-{len(stages)}-resample")
        ancestors = _systematic_resample(weights, seed=resample_seed)
        coordinates = coordinates[ancestors].clone()
        log_potential = log_potential[ancestors].clone()
        accepts = 0
        attempts = config.particles * config.pcn_sweeps_per_stage
        for sweep in range(config.pcn_sweeps_per_stage):
            noise_seed = seed(f"stage-{len(stages)}-sweep-{sweep}-noise")
            uniform_seed = seed(f"stage-{len(stages)}-sweep-{sweep}-uniform")
            noise = torch.randn(
                coordinates.shape,
                dtype=torch.float64,
                generator=torch.Generator().manual_seed(noise_seed),
            )
            proposed_coordinates = pcn_reference_scale * coordinates + config.pcn_scale * noise
            proposed_residual = coordinate_map.from_coordinates(proposed_coordinates)
            proposed_log_potential = log_potential_fn(proposed_residual)
            potential_evaluations += config.particles
            if proposed_log_potential.shape != log_potential.shape or bool(
                torch.isnan(proposed_log_potential).any()
            ):
                raise ValueError("pCN potential evaluation is invalid")
            if bool((proposed_log_potential > 1e-12).any()):
                raise ValueError("conditional pCN log potential must not exceed zero")
            log_acceptance = next_beta * (proposed_log_potential - log_potential)
            uniforms = torch.rand(
                config.particles,
                dtype=torch.float64,
                generator=torch.Generator().manual_seed(uniform_seed),
            )
            accepted = torch.log(uniforms) <= torch.minimum(
                log_acceptance, torch.zeros_like(log_acceptance)
            )
            coordinates = torch.where(accepted.unsqueeze(1), proposed_coordinates, coordinates)
            log_potential = torch.where(accepted, proposed_log_potential, log_potential)
            accepts += int(torch.count_nonzero(accepted))
        stages.append(
            SMCStageDiagnostics(
                stage=len(stages),
                beta_previous=beta,
                beta_next=next_beta,
                incremental_ess=ess,
                target_ess=target_ess,
                ess_target_met=ess + 1e-7 >= target_ess or next_beta == 1.0,
                log_normalizer_increment=log_increment,
                pcn_attempts=attempts,
                pcn_accepts=accepts,
                pcn_acceptance_rate=accepts / attempts,
            )
        )
        beta = next_beta
    final_residual = coordinate_map.from_coordinates(coordinates)
    basic_work = potential_evaluations * coordinate_map.ambient_dimension
    work = potential_evaluations * potential_work_per_particle + basic_work
    cost = BaselineCostLedger(
        training_samples=potential_evaluations,
        # pCN transitions are charged in work units and potential calls; they
        # are not gradient-optimizer steps.
        optimizer_steps=0,
        algorithmic_work_units=float(work),
        wall_seconds=time.perf_counter() - started_wall,
        cpu_seconds=time.process_time() - started_cpu,
        peak_memory_bytes=process_peak_resident_memory_bytes(),
        measurement_mode="standardized_hardware_wall",
    )
    if not stages or stages[-1].beta_next != 1.0:
        raise AssertionError("adaptive SMC did not terminate at beta one")
    return AdaptiveResidualSMCResult(
        residual_particles=final_residual,
        log_potential=log_potential,
        stages=tuple(stages),
        root_seed=root_seed,
        used_seeds=tuple(used_seeds),
        training_cost=cost,
        final_beta=beta,
        final_particles_equally_weighted=True,
        particles_are_final_inferential_units=False,
    )
