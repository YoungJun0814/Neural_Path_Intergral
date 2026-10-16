"""Guide-independent multiscale block-pCN normalizer SMC.

Blocks are selected independently of state, target, banks and Volterra geometry.
Each orthogonal-coordinate pCN proposal is Gaussian-prior reversible, so the
MH ratio is beta*(log H(candidate)-log H(current)).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from .weighted_tempered_smc import WeightedSMCConfig, _resample


def multiscale_block_pcn(z: torch.Tensor, *, scale: float,
                        generator: torch.Generator) -> torch.Tensor:
    if (z.ndim != 2 or z.shape[1] < 2 or z.shape[1] & (z.shape[1]-1)
            or z.dtype != torch.float64 or z.device.type != "cpu" or not torch.isfinite(z).all()
            or isinstance(scale, bool) or not math.isfinite(scale) or not 0 < scale <= 1):
        raise ValueError("finite CPU float64 power-of-two Gaussian coordinates required")
    count, dimension = z.shape
    sizes = torch.tensor([s for s in (2, 8, 32, dimension) if s <= dimension], dtype=torch.int64).unique()
    size = sizes[torch.randint(len(sizes), (count,), generator=generator)]
    # State-independent partition and uniform block selection. Uniform start
    # followed by division induces uniform aligned blocks because size divides d.
    starts = (torch.randint(dimension, (count,), generator=generator)//size)*size
    coordinates = torch.arange(dimension)[None, :]
    mask = (coordinates >= starts[:, None]) & (coordinates < (starts+size)[:, None])
    noise = torch.randn(z.shape, dtype=torch.float64, generator=generator)
    candidate = math.sqrt(1-scale**2)*z+scale*noise
    return torch.where(mask, candidate, z)


@dataclass(frozen=True)
class GuideFreeBlockResult:
    log_normalizer: float
    potential_evaluations: int
    stages: tuple[dict[str, float | int], ...]


def estimate_guide_free_block_normalizer(log_h: Callable[[torch.Tensor], torch.Tensor], *,
                                         dimension: int, config: WeightedSMCConfig,
                                         use_blocks: bool, block_scale: float = .7) -> GuideFreeBlockResult:
    if (isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 2
            or dimension & (dimension-1) or not isinstance(use_blocks, bool)
            or config.replicates != 1 or config.independence_every or config.retain_final_particles
            or config.mutation_kernel != "pcn" or isinstance(block_scale, bool)
            or not math.isfinite(block_scale) or not 0 < block_scale <= 1):
        raise ValueError("one guide-free fixed-schedule run, no retained bank or independence proposal")
    generator = torch.Generator().manual_seed(config.seed)
    count = config.particles
    def evaluate(z: torch.Tensor) -> torch.Tensor:
        value = log_h(z)
        if (value.shape != (count,) or value.dtype != torch.float64 or value.device.type != "cpu"
                or not torch.isfinite(value).all() or bool((value > 1e-12).any())):
            raise ValueError("strictly positive bounded H and finite nonpositive log H required")
        return value
    z = torch.randn((count, dimension), dtype=torch.float64, generator=generator)
    h = evaluate(z)
    calls = count
    weights = torch.full((count,), -math.log(count), dtype=torch.float64)
    normalizer = torch.zeros((), dtype=torch.float64)
    lineage = torch.arange(count)
    stages = []
    retained = math.sqrt(1-config.pcn_scale**2)
    for stage, (left, right) in enumerate(zip(config.temperatures[:-1], config.temperatures[1:], strict=True)):
        unnormalized = weights+(right-left)*h
        increment = torch.logsumexp(unnormalized, 0)
        if not torch.isfinite(increment):
            raise FloatingPointError("nonfinite normalizer increment")
        normalizer += increment
        weights = unnormalized-increment
        ess = float(torch.exp(-torch.logsumexp(2*weights, 0)))
        final = stage == len(config.temperatures)-2
        if (stage+1) % config.resample_every == 0 and not final:
            indices = _resample(weights.exp(), generator=generator, scheme=config.resampling_scheme)
            z, h, lineage = z[indices], h[indices], lineage[indices]
            weights.fill_(-math.log(count))
        accepted, block_accepted, full_accepted, block_proposed, full_proposed = 0, 0, 0, 0, 0
        if not final:
            for mutation in range(config.mutation_steps):
                block = use_blocks and mutation % 2 == 1
                candidate = (multiscale_block_pcn(z, scale=block_scale, generator=generator) if block else
                             retained*z+config.pcn_scale*torch.randn(z.shape, dtype=torch.float64, generator=generator))
                new_h = evaluate(candidate)
                take = torch.rand(count, dtype=torch.float64, generator=generator).log() < right*(new_h-h)
                z = torch.where(take[:, None], candidate, z)
                h = torch.where(take, new_h, h)
                calls += count
                n = int(take.sum())
                accepted += n
                if block:
                    block_accepted += n
                    block_proposed += count
                else:
                    full_accepted += n
                    full_proposed += count
        total_proposed = block_proposed+full_proposed
        stages.append({"beta": right, "ess_before_resampling": ess,
                       "surviving_initial_ancestors": int(lineage.unique().numel()),
                       "mutation_acceptance": accepted/total_proposed if total_proposed else 0.,
                       "block_acceptance": block_accepted/block_proposed if block_proposed else 0.,
                       "full_acceptance": full_accepted/full_proposed if full_proposed else 0.})
    return GuideFreeBlockResult(float(normalizer), calls, tuple(stages))
