"""Weight-only learning on a fixed physical-guide dictionary and training bank.

Both objectives use the identical defensive simplex: natural mass 0.1, total
uniform shifted floor 0.1, and adaptive shifted mass 0.8. No component is pruned,
changed, or added. The event-KL loss is normalized *only for fitting*; ordinary
importance sampling with fresh draws and the full learned density is still
required for final probability estimates. All gaps describe the finite training
bank, not population generalization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

from .defensive_mixture_allocation import allocation_weights, fit_mixture_allocation

DictionaryObjective = Literal["event-kl", "risk-m2"]


def _validate_bank(log_components: torch.Tensor, log_payoff: torch.Tensor,
                   log_sampling_density: torch.Tensor) -> None:
    tensors = (log_components, log_payoff, log_sampling_density)
    if (not all(isinstance(t, torch.Tensor) for t in tensors)
            or log_components.ndim != 2 or min(log_components.shape) < 2
            or log_payoff.shape != (len(log_components),)
            or log_sampling_density.shape != log_payoff.shape
            or any(t.dtype != torch.float64 or t.device.type != "cpu" or t.requires_grad
                   or not torch.isfinite(t).all() for t in tensors)
            or not torch.equal(log_components[:, 0], torch.zeros_like(log_payoff))):
        raise ValueError("finite detached CPU float64 bank and natural first component required")


def _validate_limits(maximum_iterations: int, relative_gap_tolerance: float,
                     line_search_iterations: int) -> None:
    if (any(isinstance(n, bool) or not isinstance(n, int) or n < 1
            for n in (maximum_iterations, line_search_iterations))
            or isinstance(relative_gap_tolerance, bool)
            or not math.isfinite(relative_gap_tolerance) or relative_gap_tolerance <= 0):
        raise ValueError("fixed positive optimization limits required")


def _validate_weights(weights: torch.Tensor, components: int) -> None:
    if (not isinstance(weights, torch.Tensor) or weights.shape != (components,)
            or weights.dtype != torch.float64 or weights.device.type != "cpu"
            or weights.requires_grad or not torch.isfinite(weights).all()
            or (weights <= 0).any() or abs(float(weights.sum())-1) > 1e-12):
        raise ValueError("positive normalized mixture weights required")


def event_training_diagnostics(log_payoff: torch.Tensor,
                               log_sampling_density: torch.Tensor) -> dict[str, float | int]:
    """Ordinary event-mass estimate and concentration of fitting weights.

    ``ordinary_log_mean_event_mass`` is log(mean(g / D_sampling)), without
    self-normalization. ESS and maximum fraction concern the normalized weights
    used by event-KL fitting; they do not change that ordinary estimate.
    """
    tensors = (log_payoff, log_sampling_density)
    if (not all(isinstance(t, torch.Tensor) for t in tensors)
            or log_payoff.ndim != 1 or len(log_payoff) < 2
            or log_sampling_density.shape != log_payoff.shape
            or any(t.dtype != torch.float64 or t.device.type != "cpu" or t.requires_grad
                   or not torch.isfinite(t).all() for t in tensors)):
        raise ValueError("finite detached CPU float64 event bank required")
    log_terms = log_payoff-log_sampling_density
    target = log_terms.softmax(0)
    return {
        "ordinary_log_mean_event_mass": float(torch.logsumexp(log_terms, 0))-math.log(len(log_terms)),
        "effective_sample_size": float(1/target.square().sum()),
        "maximum_weight_fraction": float(target.max()),
        "samples": len(log_terms),
    }


def event_kl_objective(log_components: torch.Tensor, log_payoff: torch.Tensor,
                       log_sampling_density: torch.Tensor, weights: torch.Tensor) -> float:
    """Normalized event-weighted cross-entropy, in nats, not an estimated KL.

    This is ``-sum(softmax(log(g)-log(D_sampling)) * log(D_weights))``.
    The entropy/normalizer terms independent of the mixture are omitted, so the
    returned value can be negative. Normalization is confined to this fitting
    loss and must never be used as a final probability estimator.
    """
    _validate_bank(log_components, log_payoff, log_sampling_density)
    _validate_weights(weights, log_components.shape[1])
    target = (log_payoff-log_sampling_density).softmax(0)
    log_density = torch.logsumexp(log_components+weights.log(), 1)
    return -float(target @ log_density)


@dataclass(frozen=True)
class DictionaryWeightFit:
    """Empirical fit, with objective-specific loss and gap units.

    ``relative_empirical_gap`` preserves a common output key. For risk-M2 it is
    the FW directional gap divided by the ordinary second moment. For event-KL
    it is the unscaled additive FW directional gap under normalized event
    weights (nats), *not* divided by its possibly negative loss. See
    ``gap_units``. Neither is a population certificate.
    """

    weights: torch.Tensor
    objective: DictionaryObjective
    initial_objective: float
    final_objective: float
    relative_empirical_gap: float
    iterations: int
    history: tuple[float, ...]

    @property
    def objective_units(self) -> str:
        return "log-ordinary-second-moment" if self.objective == "risk-m2" else "nats"

    @property
    def gap_units(self) -> str:
        return ("relative-ordinary-second-moment" if self.objective == "risk-m2"
                else "normalized-event-directional-gap")


def fit_dictionary_weights(log_components: torch.Tensor, log_payoff: torch.Tensor,
                           log_sampling_density: torch.Tensor, *,
                           objective: DictionaryObjective,
                           maximum_iterations: int = 256,
                           relative_gap_tolerance: float = 1e-6,
                           line_search_iterations: int = 40) -> DictionaryWeightFit:
    """Matched Frank--Wolfe weight fitting on an immutable finite dictionary.

    Both fits start at the static uniform-shift mixture and use the same domain,
    iteration cap, tolerance, and bisection limit. Risk-M2 delegates directly to
    the frozen allocator and reports log of its *ordinary* bank second moment.
    Event-KL minimizes normalized event cross-entropy; its tolerance applies to
    the normalized-target directional gap, not relative cross-entropy.
    """
    _validate_bank(log_components, log_payoff, log_sampling_density)
    _validate_limits(maximum_iterations, relative_gap_tolerance, line_search_iterations)
    if objective not in ("event-kl", "risk-m2"):
        raise ValueError("objective must be 'event-kl' or 'risk-m2'")
    if objective == "risk-m2":
        fit = fit_mixture_allocation(
            log_components, log_payoff, log_sampling_density,
            maximum_iterations=maximum_iterations,
            relative_gap_tolerance=relative_gap_tolerance,
            line_search_iterations=line_search_iterations,
        )
        return DictionaryWeightFit(
            fit.weights, objective, fit.initial_log_objective, fit.final_log_objective,
            fit.relative_empirical_fw_gap, fit.iterations, fit.history,
        )

    k = log_components.shape[1]-1
    v = torch.full((k,), 1/k, dtype=torch.float64)
    row_max = log_components.max(1).values
    a = (log_components-row_max[:, None]).exp()
    target = (log_payoff-log_sampling_density).softmax(0)
    history: list[float] = []
    gap, completed = math.inf, 0
    for iteration in range(maximum_iterations+1):
        weights = allocation_weights(v)
        denominator = a @ weights
        if not torch.isfinite(denominator).all() or (denominator <= 0).any():
            raise FloatingPointError("unresolved mixture density range; no bank row discarded")
        loss = -float(target @ (row_max+denominator.log()))
        if not math.isfinite(loss):
            raise FloatingPointError("nonfinite empirical event-KL objective")
        if history and loss > history[-1]+1e-11:
            raise FloatingPointError("convex line search increased empirical objective")
        history.append(loss)
        gradient = -.8*(a[:, 1:].T @ (target/denominator))
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

        def derivative(step: float, relative_delta: torch.Tensor = delta) -> float:
            return -float((target*relative_delta/(1+step*relative_delta)).sum())

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
    direct = event_kl_objective(log_components, log_payoff, log_sampling_density, final)
    if abs(direct-history[-1]) > 1e-10:
        raise FloatingPointError("scaled/direct objective mismatch")
    return DictionaryWeightFit(final, objective, history[0], direct, max(0., gap),
                               completed, tuple(history))
