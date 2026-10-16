"""Fixed-schedule normalizer SMC with natural or normalized-proposal base.

Proposal-base target: r^(1-beta) (p H)^beta. Natural-base: p H^beta.
The reported log normalizer always estimates E_p H, not E_r H.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

import torch

from .collective_variance_reference import CollectiveVarianceReference
from .weighted_tempered_smc import WeightedSMCConfig, _resample


class NormalizedProposal(Protocol):
    @property
    def dimension(self) -> int: ...
    def sample(self, count: int, *, path_seed: int, label_seed: int) -> torch.Tensor: ...
    def log_q_over_p(self, z: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class ProposalBridgeResult:
    log_normalizer: float
    potential_evaluations: int
    stages: tuple[dict[str, float | int], ...]


def bridge_mutation_log_ratio(beta: float, old_h: torch.Tensor, new_h: torch.Tensor,
                              old_d: torch.Tensor, new_d: torch.Tensor, *,
                              base: Literal["natural", "proposal"], global_move: bool) -> torch.Tensor:
    if global_move:
        return beta*(new_h-old_h)+(beta if base == "proposal" else 1.)*(old_d-new_d)
    return beta*(new_h-old_h)+((1-beta) if base == "proposal" else 0.)*(new_d-old_d)


def arithmetic_bridge_log_density(beta: float, log_h: torch.Tensor, log_d: torch.Tensor, *,
                                  base: Literal["natural", "proposal"], normalizer_scale: float) -> torch.Tensor:
    """log(F_beta/p), F_beta=(1-beta)b+beta*p*H/c; fixed c>0."""
    if not 0 <= beta <= 1 or not math.isfinite(normalizer_scale) or normalizer_scale <= 0:
        raise ValueError("invalid arithmetic bridge parameter")
    if base not in ("natural", "proposal"):
        raise ValueError("invalid arithmetic bridge base")
    initial = log_d if base == "proposal" else torch.zeros_like(log_h)
    if beta == 0:
        return initial
    endpoint = log_h-math.log(normalizer_scale)
    if beta == 1:
        return endpoint
    return torch.logaddexp(math.log1p(-beta)+initial, math.log(beta)+endpoint)


def estimate_proposal_bridge(log_h: Callable[[torch.Tensor], torch.Tensor], *,
                             proposal: NormalizedProposal, config: WeightedSMCConfig,
                             base: Literal["natural", "proposal"],
                             directional_units: torch.Tensor | None = None,
                             directional_price_indices: torch.Tensor | None = None,
                             bridge: Literal["geometric", "arithmetic"] = "geometric",
                             normalizer_scale: float = 1.) -> ProposalBridgeResult:
    if base not in ("natural", "proposal") or config.replicates != 1 or config.mutation_kernel != "pcn":
        raise ValueError("one fixed pCN whole-run and declared base required")
    if (bridge not in ("geometric", "arithmetic") or isinstance(normalizer_scale, bool)
            or not math.isfinite(normalizer_scale) or normalizer_scale <= 0):
        raise ValueError("invalid fixed bridge/normalizer scale")
    if bridge == "geometric" and normalizer_scale != 1:
        raise ValueError("geometric bridge does not use a scale")
    generator = torch.Generator().manual_seed(config.seed)
    count, dimension = config.particles, proposal.dimension
    if (directional_units is None) != (directional_price_indices is None):
        raise ValueError("both directional geometry tensors required")
    if directional_units is not None:
        assert directional_price_indices is not None
        CollectiveVarianceReference(directional_units, directional_price_indices)
        if directional_units.shape[1] != dimension:
            raise ValueError("directional dimension mismatch")
    def sample() -> torch.Tensor:
        seeds = torch.randint(1, 2**62, (2,), generator=generator).tolist()
        if seeds[0] == seeds[1]:
            raise ValueError("seed collision")
        return proposal.sample(count, path_seed=seeds[0], label_seed=seeds[1])
    def evaluate(z: torch.Tensor) -> torch.Tensor:
        values = log_h(z)
        if (values.shape != (count,) or values.dtype != torch.float64 or values.device.type != "cpu"
                or not torch.isfinite(values).all() or bool((values > 1e-12).any())):
            raise ValueError("bounded H requires finite nonpositive log potential")
        return values
    z = sample() if base == "proposal" else torch.randn((count, dimension), dtype=torch.float64, generator=generator)
    h, d = evaluate(z), proposal.log_q_over_p(z)
    calls = count
    weights = torch.full((count,), -math.log(count), dtype=torch.float64)
    normalizer = torch.zeros((), dtype=torch.float64)
    lineage = torch.arange(count)
    stages = []
    retained = math.sqrt(1-config.pcn_scale**2)
    for stage, (left, right) in enumerate(zip(config.temperatures[:-1], config.temperatures[1:], strict=True)):
        if bridge == "arithmetic":
            unnormalized = weights+arithmetic_bridge_log_density(right, h, d, base=base, normalizer_scale=normalizer_scale)
            unnormalized -= arithmetic_bridge_log_density(left, h, d, base=base, normalizer_scale=normalizer_scale)
        else:
            potential = h-d if base == "proposal" else h
            unnormalized = weights+(right-left)*potential
        increment = torch.logsumexp(unnormalized, 0)
        if not torch.isfinite(increment):
            raise FloatingPointError("nonfinite normalizer increment")
        normalizer += increment
        weights = unnormalized-increment
        ess = float(torch.exp(-torch.logsumexp(2*weights, 0)))
        final = stage == len(config.temperatures)-2
        if (stage+1) % config.resample_every == 0 and not final:
            indices = _resample(weights.exp(), generator=generator, scheme=config.resampling_scheme)
            z, h, d, lineage = z[indices], h[indices], d[indices], lineage[indices]
            weights.fill_(-math.log(count))
        accepted, proposed = 0, 0
        if not final:
            for mutation in range(config.mutation_steps):
                global_move = config.independence_every > 0 and (mutation+1) % config.independence_every == 0
                if global_move:
                    candidate = sample()
                elif directional_units is not None and mutation % 4 == 2:
                    assert directional_price_indices is not None
                    candidate = directional_pcn(z, directional_units, directional_price_indices,
                                                scale=config.pcn_scale, generator=generator)
                else:
                    candidate = retained*z+config.pcn_scale*torch.randn(z.shape, dtype=torch.float64, generator=generator)
                new_h, new_d = evaluate(candidate), proposal.log_q_over_p(candidate)
                if bridge == "arithmetic":
                    ratio = (arithmetic_bridge_log_density(right, new_h, new_d, base=base, normalizer_scale=normalizer_scale)
                             - arithmetic_bridge_log_density(right, h, d, base=base, normalizer_scale=normalizer_scale))
                    if global_move:
                        ratio += d-new_d
                else:
                    ratio = bridge_mutation_log_ratio(right, h, new_h, d, new_d, base=base, global_move=global_move)
                take = torch.rand(count, dtype=torch.float64, generator=generator).log() < ratio
                z = torch.where(take[:, None], candidate, z)
                h, d = torch.where(take, new_h, h), torch.where(take, new_d, d)
                calls += count
                accepted += int(take.sum())
                proposed += count
        stages.append({"beta": right, "ess_before_resampling": ess,
                       "surviving_initial_ancestors": int(lineage.unique().numel()),
                       "mutation_acceptance": accepted/proposed if proposed else 0.})
    return ProposalBridgeResult(float(normalizer)+(math.log(normalizer_scale) if bridge == "arithmetic" else 0.), calls, tuple(stages))


def directional_pcn(z: torch.Tensor, units: torch.Tensor, price_indices: torch.Tensor, *,
                    scale: float, generator: torch.Generator) -> torch.Tensor:
    """State-independent mixture of prior-reversible orthogonal rank-two pCN.

    Geometry must be validated by the caller before invoking this hot path.
    No target-gradient selection or state-dependent direction probabilities.
    """
    if not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError("invalid directional pCN scale")
    labels = torch.randint(len(units), (len(z),), generator=generator)
    u, j = units[labels], price_indices[labels]
    rows = torch.arange(len(z))
    noise = torch.randn((len(z), 2), dtype=torch.float64, generator=generator)
    retained = math.sqrt(1-scale**2)
    projection, price = (z*u).sum(1), z[rows, j]
    candidate = z+((retained-1)*projection+scale*noise[:, 0])[:, None]*u
    candidate[rows, j] += (retained-1)*price+scale*noise[:, 1]
    return candidate
