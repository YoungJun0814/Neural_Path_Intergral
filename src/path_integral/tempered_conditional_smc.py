"""Independent tempered-SMC normalizing-constant reference.

The target sequence is ``pi_beta(z) proportional to phi(z) g(z)^beta``.  It uses
multinomial resampling and pCN mutation kernels, so it neither reuses nor fits the
Gaussian transport proposal.  With a deterministic temperature schedule, the
standard Feynman--Kac product estimator is unbiased for ``E_phi[g]``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

LogPotential = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class TemperedSMCConfig:
    particles: int
    temperatures: tuple[float, ...]
    mutation_steps: int
    pcn_scale: float
    replicates: int
    seed: int

    def __post_init__(self) -> None:
        integers = (self.particles, self.mutation_steps, self.replicates)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in integers):
            raise ValueError("SMC counts must be positive integers")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("SMC seed must be an integer")
        if len(self.temperatures) < 2:
            raise ValueError("SMC requires at least two temperatures")
        if self.temperatures[0] != 0.0 or self.temperatures[-1] != 1.0:
            raise ValueError("SMC temperatures must start at zero and end at one")
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in self.temperatures
        ):
            raise ValueError("SMC temperatures must be finite and lie in [0,1]")
        if any(
            right <= left
            for left, right in zip(self.temperatures[:-1], self.temperatures[1:], strict=True)
        ):
            raise ValueError("SMC temperatures must be strictly increasing")
        if not math.isfinite(self.pcn_scale) or not 0.0 < self.pcn_scale < 1.0:
            raise ValueError("pCN scale must lie in (0,1)")


@dataclass(frozen=True)
class TemperedSMCResult:
    replicate_estimates: torch.Tensor
    mean: float
    standard_error: float
    log_replicate_estimates: torch.Tensor
    mutation_acceptance_rate: float
    minimum_incremental_ess_fraction: float
    potential_evaluations: int


def _validate_log_potential(values: torch.Tensor, particles: int) -> None:
    if values.shape != (particles,):
        raise ValueError("log potential must return one value per particle")
    if values.device.type != "cpu" or values.dtype != torch.float64:
        raise ValueError("log potential must return CPU float64")
    if torch.isnan(values).any() or torch.isposinf(values).any():
        raise FloatingPointError("log potential returned NaN or positive infinity")
    if bool((values > 64.0 * torch.finfo(torch.float64).eps).any()):
        raise ValueError("tempered SMC requires a potential bounded by one")


def estimate_tempered_normalizer(
    log_potential: LogPotential,
    *,
    dimension: int,
    config: TemperedSMCConfig,
) -> TemperedSMCResult:
    """Estimate ``E[g(Z)]`` for ``Z~N(0,I)`` using independent SMC replicates."""

    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise ValueError("SMC dimension must be a positive integer")
    log_estimates = []
    accepted = 0
    proposed = 0
    evaluations = 0
    minimum_ess_fraction = 1.0
    retained = math.sqrt(1.0 - config.pcn_scale**2)
    for replicate in range(config.replicates):
        generator = torch.Generator().manual_seed(config.seed + 104_729 * replicate)
        particles = torch.randn(
            (config.particles, dimension),
            dtype=torch.float64,
            generator=generator,
        )
        log_value = log_potential(particles)
        evaluations += config.particles
        _validate_log_potential(log_value, config.particles)
        log_normalizer = torch.zeros((), dtype=torch.float64)
        for stage, (beta_left, beta_right) in enumerate(
            zip(config.temperatures[:-1], config.temperatures[1:], strict=True)
        ):
            increment = (beta_right - beta_left) * log_value
            log_ratio = torch.logsumexp(increment, dim=0) - math.log(config.particles)
            if not torch.isfinite(log_ratio):
                raise FloatingPointError("all SMC incremental weights vanished")
            log_normalizer = log_normalizer + log_ratio
            probabilities = torch.softmax(increment, dim=0)
            ess_fraction = 1.0 / (
                config.particles * float(torch.sum(probabilities.square()))
            )
            minimum_ess_fraction = min(minimum_ess_fraction, ess_fraction)
            if stage == len(config.temperatures) - 2:
                continue
            ancestors = torch.multinomial(
                probabilities,
                config.particles,
                replacement=True,
                generator=generator,
            )
            particles = particles[ancestors]
            log_value = log_value[ancestors]
            for _ in range(config.mutation_steps):
                innovation = torch.randn(
                    particles.shape,
                    dtype=torch.float64,
                    generator=generator,
                )
                candidate = retained * particles + config.pcn_scale * innovation
                candidate_log_value = log_potential(candidate)
                evaluations += config.particles
                _validate_log_potential(candidate_log_value, config.particles)
                log_acceptance = beta_right * (candidate_log_value - log_value)
                uniforms = torch.rand(
                    config.particles,
                    dtype=torch.float64,
                    generator=generator,
                )
                accept = torch.log(uniforms) < torch.minimum(
                    log_acceptance,
                    torch.zeros_like(log_acceptance),
                )
                particles[accept] = candidate[accept]
                log_value[accept] = candidate_log_value[accept]
                accepted += int(torch.sum(accept))
                proposed += config.particles
        log_estimates.append(log_normalizer)
    log_replicates = torch.stack(log_estimates)
    estimates = torch.exp(log_replicates)
    if not torch.isfinite(estimates).all():
        raise FloatingPointError("SMC normalizing-constant estimate became nonfinite")
    mean = float(torch.mean(estimates))
    standard_error = math.sqrt(
        float(torch.var(estimates, unbiased=True)) / config.replicates
    ) if config.replicates > 1 else math.nan
    return TemperedSMCResult(
        replicate_estimates=estimates,
        mean=mean,
        standard_error=standard_error,
        log_replicate_estimates=log_replicates,
        mutation_acceptance_rate=accepted / proposed if proposed else math.nan,
        minimum_incremental_ess_fraction=minimum_ess_fraction,
        potential_evaluations=evaluations,
    )
