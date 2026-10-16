"""Fixed-schedule weighted SMC for conditional rare-event normalizers.

The target at inverse temperature beta is ``phi(z) g(z)**beta``. Unlike a
resample-at-every-stage implementation, normalized particle weights are carried
through skipped resampling stages. A fixed resampling calendar avoids a
data-dependent temperature or stopping-time claim in the primary reference.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch

from src.path_integral.elliptical_slice_kernel import elliptical_slice_transition
from src.path_integral.finite_rank_gaussian_transport import DefensiveFiniteRankGaussianMixture

LogPotential = Callable[[torch.Tensor], torch.Tensor]
ResamplingScheme = Literal["multinomial", "stratified"]


@dataclass(frozen=True)
class WeightedSMCConfig:
    particles: int
    temperatures: tuple[float, ...]
    mutation_steps: int
    pcn_scale: float
    replicates: int
    seed: int
    resample_every: int = 1
    resampling_scheme: ResamplingScheme = "multinomial"
    retain_final_particles: bool = False
    independence_every: int = 0
    mutation_kernel: Literal["pcn", "elliptical_slice"] = "pcn"
    slice_maximum_attempts: int = 1024

    def __post_init__(self) -> None:
        for name in ("particles", "replicates", "resample_every"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.mutation_steps, bool) or not isinstance(self.mutation_steps, int) or self.mutation_steps < 0:
            raise ValueError("mutation_steps must be a nonnegative integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if len(self.temperatures) < 2 or self.temperatures[0] != 0.0 or self.temperatures[-1] != 1.0:
            raise ValueError("temperatures must start at zero and end at one")
        if any(not math.isfinite(x) or not 0.0 <= x <= 1.0 for x in self.temperatures):
            raise ValueError("temperatures must be finite in [0,1]")
        if any(b <= a for a, b in zip(self.temperatures[:-1], self.temperatures[1:], strict=True)):
            raise ValueError("temperatures must be strictly increasing")
        if not math.isfinite(self.pcn_scale) or not 0.0 < self.pcn_scale < 1.0:
            raise ValueError("pCN scale must lie in (0,1)")
        if self.resampling_scheme not in ("multinomial", "stratified"):
            raise ValueError("unsupported resampling scheme")
        if not isinstance(self.retain_final_particles, bool):
            raise ValueError("retain_final_particles must be boolean")
        if isinstance(self.independence_every, bool) or not isinstance(self.independence_every, int) or self.independence_every < 0:
            raise ValueError("independence_every must be a nonnegative integer")
        if self.mutation_kernel not in ("pcn", "elliptical_slice"):
            raise ValueError("unsupported mutation kernel")
        if self.mutation_kernel == "elliptical_slice" and self.independence_every:
            raise ValueError("ellipse reference must not use static-global mutation")
        if (isinstance(self.slice_maximum_attempts, bool) or not isinstance(self.slice_maximum_attempts, int)
                or self.slice_maximum_attempts < 1):
            raise ValueError("invalid ellipse attempt cap")


@dataclass(frozen=True)
class WeightedSMCResult:
    log_replicate_estimates: torch.Tensor
    replicate_estimates: torch.Tensor
    mean: float
    standard_error: float
    potential_evaluations: int
    mutation_acceptance_rate: float | None
    replicate_diagnostics: tuple[dict[str, object], ...]
    final_particles: torch.Tensor | None
    final_weights: torch.Tensor | None


def _validate_log_potential(values: torch.Tensor, count: int) -> None:
    if (values.shape != (count,) or values.device.type != "cpu"
            or values.dtype != torch.float64 or torch.isnan(values).any()
            or torch.isposinf(values).any() or bool(torch.isneginf(values).all())
            or bool((values > 64.0 * torch.finfo(torch.float64).eps).any())):
        raise ValueError("log potential must be a nonpositive CPU float64 vector")


def _resample(
    weights: torch.Tensor, *, generator: torch.Generator, scheme: ResamplingScheme,
) -> torch.Tensor:
    count = weights.numel()
    if scheme == "multinomial":
        return torch.multinomial(weights, count, replacement=True, generator=generator)
    positions = (torch.arange(count, dtype=torch.float64) + torch.rand(
        count, dtype=torch.float64, generator=generator,
    )) / count
    cumulative = torch.cumsum(weights, dim=0)
    cumulative[-1] = 1.0
    return torch.searchsorted(cumulative, positions).clamp(max=count - 1)


def estimate_weighted_tempered_normalizer(
    log_potential: LogPotential, *, dimension: int, config: WeightedSMCConfig,
    independence_proposal: DefensiveFiniteRankGaussianMixture | None = None,
    observer: Callable[[str, int, float, torch.Tensor, torch.Tensor], None] | None = None,
) -> WeightedSMCResult:
    """Estimate E_phi[g] with deterministic bridge and exact pCN MH mutation."""

    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise ValueError("dimension must be a positive integer")
    if config.independence_every and (independence_proposal is None or independence_proposal.dimension != dimension):
        raise ValueError("declared global mutation requires a dimension-matched normalized proposal")
    if independence_proposal is not None and not config.independence_every:
        raise ValueError("unused independence proposal")
    retained = math.sqrt(1.0 - config.pcn_scale**2)
    log_estimates: list[torch.Tensor] = []
    diagnostics: list[dict[str, object]] = []
    final_particles: list[torch.Tensor] = []
    final_weights: list[torch.Tensor] = []
    accepted_total = 0
    proposed_total = 0
    potential_evaluations = 0
    count = config.particles
    for replicate in range(config.replicates):
        generator = torch.Generator().manual_seed(config.seed + 104_729 * replicate)
        particles = torch.randn((count, dimension), dtype=torch.float64, generator=generator)
        log_g = log_potential(particles)
        _validate_log_potential(log_g, count)
        potential_evaluations += count
        log_weights = torch.full((count,), -math.log(count), dtype=torch.float64)
        if observer is not None:
            observer("initial", replicate, 0., particles.clone(), torch.exp(log_weights).clone())
        log_normalizer = torch.zeros((), dtype=torch.float64)
        lineage = torch.arange(count)
        stages: list[dict[str, object]] = []
        for stage, (beta_left, beta_right) in enumerate(
            zip(config.temperatures[:-1], config.temperatures[1:], strict=True)
        ):
            log_unnormalized = log_weights + (beta_right - beta_left) * log_g
            log_increment = torch.logsumexp(log_unnormalized, dim=0)
            if not torch.isfinite(log_increment):
                raise FloatingPointError("SMC incremental normalizer is invalid")
            log_normalizer += log_increment
            log_weights = log_unnormalized - log_increment
            weights = torch.exp(log_weights)
            ess = 1.0 / float(torch.sum(weights.square()))
            max_weight = float(torch.max(weights))
            is_final = stage == len(config.temperatures) - 2
            should_resample = (stage + 1) % config.resample_every == 0 and not is_final
            if observer is not None:
                observer("pre_resample", replicate, beta_right, particles.clone(), weights.clone())
            if should_resample:
                ancestors = _resample(
                    weights, generator=generator, scheme=config.resampling_scheme,
                )
                particles = particles[ancestors]
                log_g = log_g[ancestors]
                lineage = lineage[ancestors]
                log_weights.fill_(-math.log(count))
            if observer is not None:
                observer("post_resample", replicate, beta_right, particles.clone(), torch.exp(log_weights).clone())
            stage_accepted = 0
            global_accepted, global_proposed = 0, 0
            if not is_final:
                for mutation in range(config.mutation_steps):
                    if config.mutation_kernel == "elliptical_slice":
                        moved = elliptical_slice_transition(particles, log_g, beta=beta_right,
                            log_potential=log_potential, generator=generator,
                            maximum_attempts=config.slice_maximum_attempts)
                        particles, log_g = moved.particles, moved.log_potential
                        _validate_log_potential(log_g, count)
                        potential_evaluations += moved.potential_evaluations
                        stage_accepted += count
                        accepted_total += count
                        proposed_total += count
                        continue
                    global_move = config.independence_every > 0 and (mutation+1) % config.independence_every == 0
                    if global_move:
                        assert independence_proposal is not None
                        path_seed = int(torch.randint(0, 2**63-1, (), generator=generator))
                        label_seed = int(torch.randint(0, 2**63-1, (), generator=generator))
                        if label_seed == path_seed:
                            label_seed = (label_seed+1) % (2**63-1)
                        draw = independence_proposal.sample(count, path_seed=path_seed, label_seed=label_seed)
                        candidate = draw.samples
                        correction = independence_proposal.log_q_over_p(particles)-draw.log_q_over_p
                        global_proposed += count
                    else:
                        noise = torch.randn(particles.shape, dtype=torch.float64, generator=generator)
                        candidate = retained * particles + config.pcn_scale * noise
                        correction = torch.zeros(count, dtype=torch.float64)
                    candidate_log_g = log_potential(candidate)
                    _validate_log_potential(candidate_log_g, count)
                    potential_evaluations += count
                    log_acceptance = beta_right * (candidate_log_g - log_g) + correction
                    uniform = torch.rand(count, dtype=torch.float64, generator=generator)
                    accept = torch.log(uniform) < torch.minimum(
                        log_acceptance, torch.zeros_like(log_acceptance),
                    )
                    particles[accept] = candidate[accept]
                    log_g[accept] = candidate_log_g[accept]
                    stage_accepted += int(torch.sum(accept))
                    if global_move:
                        global_accepted += int(torch.sum(accept))
                    accepted_total += int(torch.sum(accept))
                    proposed_total += count
            if observer is not None:
                observer("post_mutation", replicate, beta_right, particles.clone(), torch.exp(log_weights).clone())
            stages.append({
                "stage": stage,
                "beta": beta_right,
                "incremental_ess_fraction": ess / count,
                "maximum_incremental_weight_fraction": max_weight,
                "resampled": should_resample,
                "unique_initial_ancestors": int(torch.unique(lineage).numel()),
                "mutation_acceptance": (
                    stage_accepted / (config.mutation_steps * count)
                    if config.mutation_steps and not is_final else None
                ),
                "log_normalizer": float(log_normalizer),
                "global_mutation_proposals": global_proposed,
                "global_mutation_accepts": global_accepted,
                "mutation_kernel": config.mutation_kernel,
            })
        log_estimates.append(log_normalizer)
        diagnostics.append({
            "replicate": replicate,
            "stages": stages,
            "final_unique_initial_ancestors": int(torch.unique(lineage).numel()),
            "resampling_stages": sum(bool(stage["resampled"]) for stage in stages),
            "final_weight_ess_fraction": 1.0 / (
                count * float(torch.sum(torch.exp(log_weights).square()))
            ),
            "final_maximum_weight_fraction": float(torch.max(torch.exp(log_weights))),
        })
        if config.retain_final_particles:
            final_particles.append(particles.detach().clone())
            final_weights.append(torch.exp(log_weights).detach().clone())
    log_replicates = torch.stack(log_estimates)
    estimates = torch.exp(log_replicates)
    if not torch.isfinite(estimates).all():
        raise FloatingPointError("SMC estimates are not finite")
    standard_error = (
        math.sqrt(float(torch.var(estimates, unbiased=True)) / config.replicates)
        if config.replicates >= 2 else math.nan
    )
    return WeightedSMCResult(
        log_replicate_estimates=log_replicates,
        replicate_estimates=estimates,
        mean=float(torch.mean(estimates)),
        standard_error=standard_error,
        potential_evaluations=potential_evaluations,
        mutation_acceptance_rate=(accepted_total / proposed_total if proposed_total else None),
        replicate_diagnostics=tuple(diagnostics),
        final_particles=torch.cat(final_particles, dim=0) if final_particles else None,
        final_weights=torch.cat(final_weights, dim=0) if final_weights else None,
    )
