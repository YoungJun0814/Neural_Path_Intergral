"""Guide-free Gaussian-prior pair permutations composed with exact gradient MH."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from .gradient_umbrella_smc import evaluate_family_gradient, pcnl_log_acceptance
from .umbrella_normalizer_smc import terminal_component_log_normalizers
from .weighted_tempered_smc import WeightedSMCConfig, _resample


def cyclic_pair_permutation(z: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    if (z.ndim != 2 or z.shape[1] < 2 or z.shape[1] % 2 or len(z) < 1
            or z.dtype != torch.float64 or z.device.type != "cpu" or not torch.isfinite(z).all()
            or shifts.shape != (len(z),) or shifts.dtype != torch.int64 or shifts.device.type != "cpu"
            or (shifts < 0).any() or (shifts >= z.shape[1]//2).any()):
        raise ValueError("finite CPUfloat64 pairs and in-range row-wise shifts required")
    pairs = z.reshape(len(z), -1, 2)
    index = (torch.arange(pairs.shape[1])[None, :]-shifts[:, None]) % pairs.shape[1]
    return pairs.gather(1, index[:, :, None].expand(-1, -1, 2)).reshape_as(z)


@dataclass(frozen=True)
class CyclicUmbrellaResult:
    log_normalizer: float
    log_component_normalizers: torch.Tensor
    joint_evaluations: int
    joint_gradient_requests: int
    stages: tuple[dict[str, float | int], ...]


def estimate_cyclic_umbrella(log_components: Callable[[torch.Tensor], torch.Tensor], *,
                             dimension: int, log_family_weights: torch.Tensor,
                             config: WeightedSMCConfig, use_cyclic: bool,
                             maximum_drift_norm: float = 20.) -> CyclicUmbrellaResult:
    if (type(dimension) is not int or dimension < 2 or dimension % 2 or type(use_cyclic) is not bool
            or config.replicates != 1 or config.mutation_steps != 2 or config.independence_every
            or config.retain_final_particles or config.mutation_kernel != "pcn"):
        raise ValueError("one fixed two-mutation Gaussian pair family required")
    generator = torch.Generator().manual_seed(config.seed)
    count = config.particles
    z = torch.randn((count, dimension), dtype=torch.float64, generator=generator)

    def evaluate(x):
        return evaluate_family_gradient(log_components, x, log_family_weights,
                                        maximum_drift_norm=maximum_drift_norm)

    components, h, drift = evaluate(z)
    if (h > 1e-12).any():
        raise ValueError("bounded probability-family potential required")
    calls = count
    weights = torch.full((count,), -math.log(count), dtype=torch.float64)
    normalizer = torch.zeros((), dtype=torch.float64)
    lineage, stages = torch.arange(count), []
    a = math.sqrt(1-config.pcn_scale**2)
    for stage, (left, right) in enumerate(zip(config.temperatures[:-1], config.temperatures[1:], strict=True)):
        unnormalized = weights+(right-left)*h
        increment = torch.logsumexp(unnormalized, 0)
        if not torch.isfinite(increment):
            raise FloatingPointError("nonfinite family normalizer increment")
        normalizer += increment
        weights = unnormalized-increment
        ess = float(torch.exp(-torch.logsumexp(2*weights, 0)))
        final = stage == len(config.temperatures)-2
        if (stage+1) % config.resample_every == 0 and not final:
            indices = _resample(weights.exp(), generator=generator, scheme=config.resampling_scheme)
            z, h, drift, components, lineage = z[indices], h[indices], drift[indices], components[indices], lineage[indices]
            weights.fill_(-math.log(count))
        acceptance = [0., 0.]
        if not final:
            for move in range(2):
                cyclic = use_cyclic and move == 1
                candidate = (cyclic_pair_permutation(z, torch.randint(dimension//2, (count,), generator=generator))
                    if cyclic else a*z+(1-a)*right*drift+config.pcn_scale*torch.randn(z.shape, dtype=torch.float64, generator=generator))
                new_components, new_h, new_drift = evaluate(candidate)
                if (new_h > 1e-12).any():
                    raise ValueError("bounded probability-family potential required")
                ratio = (right*(new_h-h) if cyclic else pcnl_log_acceptance(
                    z, candidate, h, new_h, drift, new_drift, beta=right, scale=config.pcn_scale))
                take = torch.rand(count, dtype=torch.float64, generator=generator).log() < ratio
                z = torch.where(take[:, None], candidate, z)
                h = torch.where(take, new_h, h)
                drift = torch.where(take[:, None], new_drift, drift)
                components = torch.where(take[:, None], new_components, components)
                calls += count
                acceptance[move] = float(take.double().mean())
        stages.append({"beta": right, "ess_before_resampling": ess,
            "surviving_initial_ancestors": int(lineage.unique().numel()),
            "first_gradient_acceptance": acceptance[0], "second_mutation_acceptance": acceptance[1],
            "second_mutation_is_cyclic": int(use_cyclic), "log_normalizer": float(normalizer)})
    vector = terminal_component_log_normalizers(float(normalizer), weights, components, log_family_weights)
    # Gradient callback requests are counted conservatively even for a constant oracle.
    return CyclicUmbrellaResult(float(normalizer), vector, calls, calls, tuple(stages))
