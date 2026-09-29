"""Finite-dimensional variance identities for exact residual importance sampling."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def log_density_stability_relative_variance_bound(epsilon: float) -> float:
    """Return ``exp(epsilon)-1`` for a uniform log-density approximation error."""

    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("log-density error must be finite and nonnegative")
    return math.expm1(epsilon)


def strict_improvement_certified(*, epsilon: float, natural_relative_variance: float) -> bool:
    """Check the sufficient condition for beating natural residual sampling."""

    if not math.isfinite(natural_relative_variance) or natural_relative_variance < 0.0:
        raise ValueError("natural relative variance must be finite and nonnegative")
    return log_density_stability_relative_variance_bound(epsilon) < natural_relative_variance


def rao_blackwell_variance_gap_integrand(
    likelihood: torch.Tensor, conditional_probability: torch.Tensor
) -> torch.Tensor:
    """Return ``L(R)^2 g(R)(1-g(R))``, the conditional variance gap.

    Its expectation under the residual proposal is exactly
    ``Var(L*1{A<=a(R)}) - Var(L*g(R))``. The returned sample average is only a
    diagnostic estimate of that expectation.
    """

    if likelihood.ndim != 1 or conditional_probability.shape != likelihood.shape:
        raise ValueError("likelihood and conditional probability shapes differ")
    if not torch.isfinite(likelihood).all() or bool((likelihood < 0.0).any()):
        raise ValueError("likelihood must be finite and nonnegative")
    if not torch.isfinite(conditional_probability).all() or bool(
        ((conditional_probability < 0.0) | (conditional_probability > 1.0)).any()
    ):
        raise ValueError("conditional probability must lie in [0, 1]")
    return likelihood.square() * conditional_probability * (1.0 - conditional_probability)


@dataclass(frozen=True)
class DiscreteLogDensityStabilityCheck:
    epsilon: float
    chi_square_divergence: float
    relative_variance_bound: float
    bound_holds: bool


def check_discrete_log_density_stability(
    zero_variance_density: torch.Tensor,
    proposal_density: torch.Tensor,
    *,
    tolerance: float = 1e-12,
) -> DiscreteLogDensityStabilityCheck:
    """Numerically verify the stability theorem on a finite probability space."""

    if zero_variance_density.ndim != 1 or proposal_density.shape != zero_variance_density.shape:
        raise ValueError("discrete densities must be vectors of the same shape")
    if zero_variance_density.numel() < 2:
        raise ValueError("discrete stability check requires at least two states")
    if (
        not torch.isfinite(zero_variance_density).all()
        or not torch.isfinite(proposal_density).all()
    ):
        raise ValueError("discrete densities must be finite")
    if bool((zero_variance_density <= 0.0).any()) or bool((proposal_density <= 0.0).any()):
        raise ValueError("uniform log stability requires strictly positive densities")
    if not math.isclose(float(torch.sum(zero_variance_density)), 1.0, abs_tol=tolerance):
        raise ValueError("zero-variance density must sum to one")
    if not math.isclose(float(torch.sum(proposal_density)), 1.0, abs_tol=tolerance):
        raise ValueError("proposal density must sum to one")
    log_ratio = torch.log(proposal_density) - torch.log(zero_variance_density)
    epsilon = float(torch.amax(torch.abs(log_ratio)))
    chi_square = float(torch.sum(zero_variance_density.square() / proposal_density) - 1.0)
    bound = log_density_stability_relative_variance_bound(epsilon)
    return DiscreteLogDensityStabilityCheck(
        epsilon=epsilon,
        chi_square_divergence=max(0.0, chi_square),
        relative_variance_bound=bound,
        bound_holds=chi_square <= bound + tolerance * max(1.0, bound),
    )
