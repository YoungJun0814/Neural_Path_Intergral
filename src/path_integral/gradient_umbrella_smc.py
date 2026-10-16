"""Guide-free gradient MH and fixed-schedule unnormalized family SMC.

The callback must be row-separable. Only the proposal drift can be norm-capped;
potentials and endpoint contributions are never clipped. Finite-grid exact MH
does not imply dimension-independent mixing or a continuum error certificate.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from .umbrella_normalizer_smc import terminal_component_log_normalizers, umbrella_log_potential
from .weighted_tempered_smc import WeightedSMCConfig, _resample


def _coordinates(z: torch.Tensor) -> None:
    if (z.ndim != 2 or z.dtype != torch.float64 or z.device.type != "cpu"
            or not torch.isfinite(z).all() or z.shape[0] < 1 or z.shape[1] < 1):
        raise ValueError("finite CPU float64 coordinate matrix required")


def capped_proposal_drift(gradient: torch.Tensor, *, maximum_norm: float) -> torch.Tensor:
    _coordinates(gradient)
    if isinstance(maximum_norm, bool) or not math.isfinite(maximum_norm) or maximum_norm <= 0:
        raise ValueError("positive finite drift norm cap required")
    norms = torch.linalg.vector_norm(gradient, dim=1, keepdim=True)
    if not torch.isfinite(norms).all():
        raise FloatingPointError("nonfinite gradient norm")
    # clamp applies to the proposal drift only, never to the target or weight.
    return gradient*(maximum_norm/norms.clamp_min(maximum_norm))


def pcnl_log_acceptance(z: torch.Tensor, candidate: torch.Tensor,
                        h: torch.Tensor, new_h: torch.Tensor,
                        drift: torch.Tensor, new_drift: torch.Tensor,
                        *, beta: float, scale: float) -> torch.Tensor:
    """Stable expansion of target plus reverse/forward Gaussian log densities."""
    for v in (z, candidate, drift, new_drift):
        _coordinates(v)
        if v.shape != z.shape:
            raise ValueError("coordinate/drift shape mismatch")
    if (isinstance(beta, bool) or not math.isfinite(beta) or not 0 <= beta <= 1
            or isinstance(scale, bool) or not math.isfinite(scale) or not 0 < scale < 1
            or any(v.shape != (len(z),) or v.dtype != torch.float64 or v.device.type != "cpu"
                   or not torch.isfinite(v).all() for v in (h, new_h))):
        raise ValueError("invalid temperature, scale or log potential")
    a = math.sqrt(1-scale**2)
    c = 1-a
    result = (beta*(new_h-h)
              + beta/(1+a)*((z-a*candidate)*new_drift-(candidate-a*z)*drift).sum(1)
              + c*beta**2/(2*(1+a))*(drift.square().sum(1)-new_drift.square().sum(1)))
    if not torch.isfinite(result).all():
        raise FloatingPointError("nonfinite gradient MH ratio")
    return result


def evaluate_family_gradient(log_components: Callable[[torch.Tensor], torch.Tensor],
                             z: torch.Tensor, log_weights: torch.Tensor,
                             *, maximum_drift_norm: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A row-separable derivative; constant callbacks need no backward pass."""
    components, h, drift, _ = _evaluate_family_gradient(log_components, z, log_weights,
                                                      maximum_drift_norm=maximum_drift_norm)
    return components, h, drift


def _evaluate_family_gradient(log_components: Callable[[torch.Tensor], torch.Tensor],
                              z: torch.Tensor, log_weights: torch.Tensor,
                              *, maximum_drift_norm: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    _coordinates(z)
    with torch.enable_grad():
        leaf = z.detach().clone().requires_grad_(True)
        components = log_components(leaf)
        if components.shape != (len(z), len(log_weights)):
            raise ValueError("component family shape mismatch")
        h = umbrella_log_potential(components, log_weights)
        backward = h.requires_grad
        if backward:
            gradient = torch.autograd.grad(h.sum(), leaf, create_graph=False)[0]
        else:
            gradient = torch.zeros_like(leaf)
    drift = capped_proposal_drift(gradient.detach(), maximum_norm=maximum_drift_norm)
    return components.detach(), h.detach(), drift, backward


@dataclass(frozen=True)
class GradientUmbrellaResult:
    log_normalizer: float
    log_component_normalizers: torch.Tensor
    potential_evaluations: int
    gradient_evaluations: int
    stages: tuple[dict[str, float | int], ...]


def estimate_gradient_umbrella_normalizers(log_components: Callable[[torch.Tensor], torch.Tensor], *,
                                          dimension: int, log_mixture_weights: torch.Tensor,
                                          config: WeightedSMCConfig,
                                          maximum_drift_norm: float = 20.) -> GradientUmbrellaResult:
    if (isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1
            or config.replicates != 1 or config.retain_final_particles
            or config.independence_every or config.mutation_kernel != "pcn"
            or log_mixture_weights.ndim != 1):
        raise ValueError("one frozen guide-free run without retained bank required")
    weights_family = log_mixture_weights.detach().clone()
    umbrella_log_potential(torch.zeros((1, len(weights_family)), dtype=torch.float64), weights_family)
    # Validate cap even if a constant potential has a zero derivative.
    capped_proposal_drift(torch.zeros((1, dimension), dtype=torch.float64), maximum_norm=maximum_drift_norm)
    count = config.particles
    generator = torch.Generator().manual_seed(config.seed)
    z = torch.randn((count, dimension), dtype=torch.float64, generator=generator)
    def evaluate(x):
        return _evaluate_family_gradient(log_components, x, weights_family,
                                         maximum_drift_norm=maximum_drift_norm)
    components, h, drift, backward = evaluate(z)
    calls = count
    gradient_calls = count*int(backward)
    weights = torch.full((count,), -math.log(count), dtype=torch.float64)
    normalizer = torch.zeros((), dtype=torch.float64)
    lineage = torch.arange(count)
    stages = []
    a = math.sqrt(1-config.pcn_scale**2)
    c = 1-a
    for stage, (left, right) in enumerate(zip(config.temperatures[:-1], config.temperatures[1:], strict=True)):
        unnormalized = weights+(right-left)*h
        increment = torch.logsumexp(unnormalized, 0)
        if not torch.isfinite(increment):
            raise FloatingPointError("nonfinite gradient-family normalizer increment")
        normalizer += increment
        weights = unnormalized-increment
        ess = float(torch.exp(-torch.logsumexp(2*weights, 0)))
        final = stage == len(config.temperatures)-2
        if (stage+1) % config.resample_every == 0 and not final:
            indices = _resample(weights.exp(), generator=generator, scheme=config.resampling_scheme)
            z, h, drift, components, lineage = z[indices], h[indices], drift[indices], components[indices], lineage[indices]
            weights.fill_(-math.log(count))
        accepted = 0
        if not final:
            for _ in range(config.mutation_steps):
                candidate = (a*z+c*right*drift
                             +config.pcn_scale*torch.randn(z.shape, dtype=torch.float64, generator=generator))
                new_components, new_h, new_drift, backward = evaluate(candidate)
                ratio = pcnl_log_acceptance(z, candidate, h, new_h, drift, new_drift,
                                            beta=right, scale=config.pcn_scale)
                take = torch.rand(count, dtype=torch.float64, generator=generator).log() < ratio
                z = torch.where(take[:, None], candidate, z)
                h = torch.where(take, new_h, h)
                drift = torch.where(take[:, None], new_drift, drift)
                components = torch.where(take[:, None], new_components, components)
                calls += count
                gradient_calls += count*int(backward)
                accepted += int(take.sum())
        stages.append({"beta": right, "ess_before_resampling": ess,
                       "surviving_initial_ancestors": int(lineage.unique().numel()),
                       "mutation_acceptance": accepted/(config.mutation_steps*count) if config.mutation_steps and not final else 0.,
                       "maximum_proposal_drift_norm": float(torch.linalg.vector_norm(drift, dim=1).max())})
    vector = terminal_component_log_normalizers(float(normalizer), weights, components, weights_family)
    return GradientUmbrellaResult(float(normalizer), vector, calls, gradient_calls, tuple(stages))
