"""Guide-free fixed-work probability SMC with exact Gaussian-split MH moves."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from .gaussian_split_hmc import gaussian_split_trajectory, probability_evaluator
from .gradient_umbrella_smc import pcnl_log_acceptance
from .weighted_tempered_smc import WeightedSMCConfig, _resample


@dataclass(frozen=True)
class HamiltonianSMCResult:
    log_normalizer: float
    callback_rows: int
    gradient_requests: int
    stages: tuple[dict[str, float | int], ...]


def estimate_hamiltonian_probability(callback: Callable[[torch.Tensor], torch.Tensor], *,
                                      dimension: int, config: WeightedSMCConfig,
                                      use_hamiltonian: bool, trajectory_steps: int = 4,
                                      step_size: float = .35,
                                      maximum_drift_norm: float = 20.) -> HamiltonianSMCResult:
    if (type(dimension) is not int or dimension < 1 or type(use_hamiltonian) is not bool
            or type(trajectory_steps) is not int or trajectory_steps < 1
            or config.replicates != 1 or config.retain_final_particles or config.independence_every
            or config.mutation_kernel != "pcn" or config.mutation_steps != trajectory_steps
            or not math.isfinite(step_size) or not 0 < step_size < 1):
        raise ValueError("one frozen equal-callback-work probability SMC required")
    evaluate = probability_evaluator(callback, maximum_drift_norm=maximum_drift_norm)
    generator = torch.Generator().manual_seed(config.seed)
    n = config.particles
    z = torch.randn((n, dimension), dtype=torch.float64, generator=generator)
    comp, h, drift = evaluate(z)
    weights = torch.full((n,), -math.log(n), dtype=torch.float64)
    log_z = torch.zeros((), dtype=torch.float64)
    lineage = torch.arange(n)
    calls, stages = n, []
    a = math.sqrt(1-config.pcn_scale**2)
    for stage, (left, right) in enumerate(zip(config.temperatures[:-1], config.temperatures[1:], strict=True)):
        u = weights+(right-left)*h
        increment = torch.logsumexp(u, 0)
        if not torch.isfinite(increment):
            raise FloatingPointError("nonfinite fixed-bridge normalizer increment")
        log_z += increment
        weights = u-increment
        ess = float(torch.exp(-torch.logsumexp(2*weights, 0)))
        final = stage == len(config.temperatures)-2
        if (stage+1) % config.resample_every == 0 and not final:
            indices = _resample(weights.exp(), generator=generator, scheme=config.resampling_scheme)
            z, comp, h, drift, lineage = z[indices], comp[indices], h[indices], drift[indices], lineage[indices]
            weights.fill_(-math.log(n))
        accepted, distance = [], 0.
        if not final:
            before = z.clone()
            if use_hamiltonian:
                p = torch.randn(z.shape, dtype=torch.float64, generator=generator)
                proposal = gaussian_split_trajectory(z, p, comp, h, drift, evaluate=evaluate,
                    beta=right, step_size=step_size, steps=trajectory_steps)
                take = torch.rand(n, dtype=torch.float64, generator=generator).log() < proposal.log_acceptance
                z = torch.where(take[:, None], proposal.position, z)
                comp = torch.where(take[:, None], proposal.components, comp)
                h = torch.where(take, proposal.potential, h)
                drift = torch.where(take[:, None], proposal.drift, drift)
                accepted.append(float(take.double().mean()))
                calls += proposal.callback_rows
            else:
                for _ in range(trajectory_steps):
                    candidate = a*z+(1-a)*right*drift+config.pcn_scale*torch.randn(z.shape, dtype=torch.float64, generator=generator)
                    new_comp, new_h, new_drift = evaluate(candidate)
                    ratio = pcnl_log_acceptance(z, candidate, h, new_h, drift, new_drift,
                                                beta=right, scale=config.pcn_scale)
                    take = torch.rand(n, dtype=torch.float64, generator=generator).log() < ratio
                    z = torch.where(take[:, None], candidate, z)
                    comp = torch.where(take[:, None], new_comp, comp)
                    h = torch.where(take, new_h, h)
                    drift = torch.where(take[:, None], new_drift, drift)
                    accepted.append(float(take.double().mean()))
                    calls += n
            distance = float((z-before).square().sum(1).mean())/dimension
        stages.append({"beta": right, "ess_before_resampling": ess,
            "initial_ancestors": int(lineage.unique().numel()),
            "mean_acceptance": sum(accepted)/len(accepted) if accepted else 0.,
            "mean_squared_displacement_per_coordinate": distance,
            "log_normalizer": float(log_z)})
    return HamiltonianSMCResult(float(log_z), calls, calls, tuple(stages))
