"""SMC unnormalized-measure estimates for a frozen target family.

For positive bounded H_j and frozen positive weights w_j, H=sum w_j H_j.
The endpoint estimator Z_hat * sum W_i H_j(X_i)/H(X_i) estimates E_p H_j.
This is an unnormalized SMC measure, not a self-normalized final IS estimator.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from .collective_variance_reference import CollectiveVarianceReference
from .guide_free_block_smc import multiscale_block_pcn
from .proposal_bridge_smc import NormalizedProposal, bridge_mutation_log_ratio, directional_pcn
from .weighted_tempered_smc import WeightedSMCConfig, _resample


def umbrella_log_potential(log_components: torch.Tensor, log_weights: torch.Tensor) -> torch.Tensor:
    if (log_components.ndim != 2 or log_weights.ndim != 1 or len(log_weights) < 1
            or log_components.shape[1] != len(log_weights)
            or any(x.dtype != torch.float64 or x.device.type != "cpu" or not torch.isfinite(x).all()
                   for x in (log_components, log_weights))
            or bool((log_components > 1e-12).any())
            or abs(float(torch.logsumexp(log_weights, 0))) > 1e-12):
        raise ValueError("positive bounded component potentials and normalized frozen log weights required")
    return torch.logsumexp(log_components+log_weights, 1)


def terminal_component_log_normalizers(log_normalizer: float, log_particle_weights: torch.Tensor,
                                       log_components: torch.Tensor, log_mixture_weights: torch.Tensor) -> torch.Tensor:
    log_h = umbrella_log_potential(log_components, log_mixture_weights)
    if (not math.isfinite(log_normalizer) or log_particle_weights.shape != (len(log_components),)
            or log_particle_weights.dtype != torch.float64 or log_particle_weights.device.type != "cpu"
            or not torch.isfinite(log_particle_weights).all()
            or abs(float(torch.logsumexp(log_particle_weights, 0))) > 1e-12):
        raise ValueError("normalized terminal weights and finite SMC normalizer required")
    result = log_normalizer+torch.logsumexp(log_particle_weights[:, None]+log_components-log_h[:, None], 0)
    if abs(float(torch.logsumexp(result+log_mixture_weights, 0))-log_normalizer) > 1e-10:
        raise FloatingPointError("unnormalized family measure identity failed")
    if bool((log_components-log_h[:, None] > -log_mixture_weights+1e-10).any()):
        raise FloatingPointError("bounded component-to-umbrella ratio failed")
    return result


@dataclass(frozen=True)
class UmbrellaNormalizerResult:
    log_normalizer: float
    log_component_normalizers: torch.Tensor
    potential_evaluations: int
    stages: tuple[dict[str, float | int], ...]


def estimate_umbrella_normalizers(log_components: Callable[[torch.Tensor], torch.Tensor], *,
                                  dimension: int, log_mixture_weights: torch.Tensor,
                                  config: WeightedSMCConfig, use_blocks: bool,
                                  block_scale: float = .7,
                                  proposal: NormalizedProposal | None = None,
                                  directional_units: torch.Tensor | None = None,
                                  directional_price_indices: torch.Tensor | None = None) -> UmbrellaNormalizerResult:
    if (isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 2 or dimension & (dimension-1)
            or config.replicates != 1 or config.retain_final_particles
            or config.mutation_kernel != "pcn" or not isinstance(use_blocks, bool)
            or isinstance(block_scale, bool) or not math.isfinite(block_scale) or not 0 < block_scale <= 1):
        raise ValueError("one fixed power-of-two run required")
    if proposal is None and (config.independence_every or directional_units is not None or directional_price_indices is not None):
        raise ValueError("guide-free base cannot use guided or independence mutation")
    if proposal is not None and proposal.dimension != dimension:
        raise ValueError("normalized proposal dimension mismatch")
    if (directional_units is None) != (directional_price_indices is None):
        raise ValueError("both directional tensors required")
    if directional_units is not None:
        assert directional_price_indices is not None
        CollectiveVarianceReference(directional_units, directional_price_indices)
        if directional_units.shape[1] != dimension:
            raise ValueError("directional geometry dimension mismatch")
    if log_mixture_weights.ndim != 1:
        raise ValueError("one-dimensional mixture log weights required")
    log_mixture_weights = log_mixture_weights.detach().clone()
    umbrella_log_potential(torch.zeros((1, len(log_mixture_weights)), dtype=torch.float64), log_mixture_weights)
    count = config.particles
    generator = torch.Generator().manual_seed(config.seed)
    def proposal_sample() -> torch.Tensor:
        assert proposal is not None
        seeds = torch.randint(1, 2**62, (2,), generator=generator).tolist()
        if seeds[0] == seeds[1]:
            raise ValueError("proposal child seed collision")
        return proposal.sample(count, path_seed=seeds[0], label_seed=seeds[1])
    def evaluate(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        components = log_components(z)
        if components.shape != (count, len(log_mixture_weights)):
            raise ValueError("component family shape mismatch")
        return components, umbrella_log_potential(components, log_mixture_weights)
    z = (proposal_sample() if proposal is not None else
         torch.randn((count, dimension), dtype=torch.float64, generator=generator))
    components, h = evaluate(z)
    density = proposal.log_q_over_p(z) if proposal is not None else torch.zeros_like(h)
    calls = count
    weights = torch.full((count,), -math.log(count), dtype=torch.float64)
    normalizer = torch.zeros((), dtype=torch.float64)
    lineage = torch.arange(count)
    stages = []
    retained = math.sqrt(1-config.pcn_scale**2)
    for stage, (left, right) in enumerate(zip(config.temperatures[:-1], config.temperatures[1:], strict=True)):
        unnormalized = weights+(right-left)*(h-density if proposal is not None else h)
        increment = torch.logsumexp(unnormalized, 0)
        if not torch.isfinite(increment):
            raise FloatingPointError("nonfinite umbrella normalizer increment")
        normalizer += increment
        weights = unnormalized-increment
        ess = float(torch.exp(-torch.logsumexp(2*weights, 0)))
        final = stage == len(config.temperatures)-2
        if (stage+1) % config.resample_every == 0 and not final:
            indices = _resample(weights.exp(), generator=generator, scheme=config.resampling_scheme)
            z, h, components, density, lineage = z[indices], h[indices], components[indices], density[indices], lineage[indices]
            weights.fill_(-math.log(count))
        accepted = 0
        if not final:
            for mutation in range(config.mutation_steps):
                global_move = proposal is not None and config.independence_every > 0 and (mutation+1) % config.independence_every == 0
                if global_move:
                    candidate = proposal_sample()
                elif directional_units is not None and mutation % 4 == 2:
                    assert directional_price_indices is not None
                    candidate = directional_pcn(z, directional_units, directional_price_indices,
                                                scale=config.pcn_scale, generator=generator)
                elif use_blocks and mutation % 2 == 1:
                    candidate = multiscale_block_pcn(z, scale=block_scale, generator=generator)
                else:
                    candidate = retained*z+config.pcn_scale*torch.randn(z.shape, dtype=torch.float64, generator=generator)
                new_components, new_h = evaluate(candidate)
                new_density = proposal.log_q_over_p(candidate) if proposal is not None else density
                ratio = bridge_mutation_log_ratio(right, h, new_h, density, new_density,
                    base="proposal" if proposal is not None else "natural", global_move=global_move)
                take = torch.rand(count, dtype=torch.float64, generator=generator).log() < ratio
                z = torch.where(take[:, None], candidate, z)
                h = torch.where(take, new_h, h)
                density = torch.where(take, new_density, density)
                components = torch.where(take[:, None], new_components, components)
                calls += count
                accepted += int(take.sum())
        stages.append({"beta": right, "ess_before_resampling": ess,
                       "surviving_initial_ancestors": int(lineage.unique().numel()),
                       "mutation_acceptance": accepted/(config.mutation_steps*count) if config.mutation_steps and not final else 0.})
    vector = terminal_component_log_normalizers(float(normalizer), weights, components, log_mixture_weights)
    return UmbrellaNormalizerResult(float(normalizer), vector, calls, tuple(stages))
