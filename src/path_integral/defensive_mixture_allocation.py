"""Convex fixed-dictionary allocation fitted only on an independent IS bank.

Training objectives/gaps are empirical diagnostics, not population guarantees.
Natural mass and a uniform shifted floor remain fixed. Final probability
estimates must use fresh draws and the full learned mixture density.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def _validate(log_components: torch.Tensor, log_payoff: torch.Tensor,
              log_sampling_density: torch.Tensor) -> None:
    if (log_components.ndim != 2 or min(log_components.shape) < 2
            or log_payoff.shape != (len(log_components),)
            or log_sampling_density.shape != log_payoff.shape
            or any(t.dtype != torch.float64 or t.device.type != "cpu" or t.requires_grad
                   or not torch.isfinite(t).all()
                   for t in (log_components, log_payoff, log_sampling_density))
            or not torch.equal(log_components[:, 0], torch.zeros_like(log_payoff))):
        raise ValueError("finite non-autograd CPU float64 bank and natural first component required")


def allocation_weights(v: torch.Tensor, *, natural_mass: float = .1,
                       shifted_floor_mass: float = .1) -> torch.Tensor:
    if (v.ndim != 1 or len(v) < 1 or v.dtype != torch.float64 or v.device.type != "cpu"
            or v.requires_grad or not torch.isfinite(v).all() or (v < 0).any()
            or abs(float(v.sum())-1) > 1e-12
            or not 0 < natural_mass < 1 or not 0 < shifted_floor_mass < 1-natural_mass):
        raise ValueError("valid simplex and strictly defensive allocation required")
    shifted = shifted_floor_mass/len(v)+(1-natural_mass-shifted_floor_mass)*v
    return torch.cat((torch.tensor([natural_mass], dtype=torch.float64), shifted))


def log_second_moment_objective(log_components: torch.Tensor, log_payoff: torch.Tensor,
                                log_sampling_density: torch.Tensor,
                                weights: torch.Tensor) -> float:
    """Log ordinary bank mean g^2/(D_sampling D_weights); not self-normalized IS."""
    _validate(log_components, log_payoff, log_sampling_density)
    if (weights.shape != (log_components.shape[1],) or weights.dtype != torch.float64
            or weights.device.type != "cpu" or weights.requires_grad
            or not torch.isfinite(weights).all() or (weights <= 0).any()
            or abs(float(weights.sum())-1) > 1e-12):
        raise ValueError("positive normalized mixture weights required")
    log_density = torch.logsumexp(log_components+weights.log(), 1)
    return float(torch.logsumexp(2*log_payoff-log_sampling_density-log_density, 0))-math.log(len(log_payoff))


@dataclass(frozen=True)
class MixtureAllocationFit:
    weights: torch.Tensor
    initial_log_objective: float
    final_log_objective: float
    relative_empirical_fw_gap: float
    iterations: int
    history: tuple[float, ...]


def fit_mixture_allocation(log_components: torch.Tensor, log_payoff: torch.Tensor,
                           log_sampling_density: torch.Tensor, *, maximum_iterations: int = 256,
                           relative_gap_tolerance: float = 1e-6,
                           line_search_iterations: int = 40) -> MixtureAllocationFit:
    """Frank--Wolfe on a fixed convex domain, stable relative-gradient line search.

    Softmax here scales the optimization gradient only. No estimated normalizer
    is inserted into the training objective or any final probability estimate.
    """
    _validate(log_components, log_payoff, log_sampling_density)
    if (any(isinstance(n, bool) or not isinstance(n, int) or n < 1
            for n in (maximum_iterations, line_search_iterations))
            or not math.isfinite(relative_gap_tolerance) or relative_gap_tolerance <= 0):
        raise ValueError("fixed positive optimization limits required")
    k = log_components.shape[1]-1
    v = torch.full((k,), 1/k, dtype=torch.float64)
    row_max = log_components.max(1).values
    a = (log_components-row_max[:, None]).exp()
    base_logs = 2*log_payoff-log_sampling_density-row_max
    history: list[float] = []
    gap, completed = math.inf, 0
    for iteration in range(maximum_iterations+1):
        weights = allocation_weights(v)
        denominator = a @ weights
        if not torch.isfinite(denominator).all() or (denominator <= 0).any():
            raise FloatingPointError("unresolved mixture density range; no bank row discarded")
        terms = base_logs-denominator.log()
        objective = float(torch.logsumexp(terms, 0))-math.log(len(terms))
        if history and objective > history[-1]+1e-11:
            raise FloatingPointError("convex line search increased empirical objective")
        history.append(objective)
        relative_terms = terms.softmax(0)
        gradient = -.8*(a[:, 1:].T @ (relative_terms/denominator))
        vertex = int(gradient.argmin())
        gap = float(gradient @ v-gradient[vertex])
        if not math.isfinite(gap) or gap < -1e-10:
            raise FloatingPointError("invalid empirical convex gap")
        completed = iteration
        if gap <= relative_gap_tolerance or iteration == maximum_iterations:
            break
        vertex_v = torch.zeros_like(v)
        vertex_v[vertex] = 1.
        vertex_denominator = a @ allocation_weights(vertex_v)
        delta = (vertex_denominator-denominator)/denominator

        def derivative(step: float, terms_at_zero: torch.Tensor = relative_terms,
                       relative_delta: torch.Tensor = delta) -> float:
            return -float((terms_at_zero*relative_delta/(1+step*relative_delta).square()).sum())

        if derivative(1.) <= 0:
            step = 1.
        else:
            low, high = 0., 1.
            for _ in range(line_search_iterations):
                mid = (low+high)/2
                if derivative(mid) < 0:
                    low = mid
                else:
                    high = mid
            step = (low+high)/2
        v = (1-step)*v+step*vertex_v
    final = allocation_weights(v)
    direct = log_second_moment_objective(log_components, log_payoff, log_sampling_density, final)
    if abs(direct-history[-1]) > 1e-10:
        raise FloatingPointError("scaled/direct objective mismatch")
    return MixtureAllocationFit(final, history[0], direct, max(0., gap), completed, tuple(history))
