"""Exact finite-grid Volterra Cameron-Martin tilts, not hard conditioning."""

from __future__ import annotations

import math

import torch

from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.finite_rank_gaussian_transport import (
    DefensiveFiniteRankGaussianMixture,
    FiniteRankGaussianComponent,
)


def volterra_monitoring_operator(problem: RBergomiBaselineProblem) -> torch.Tensor:
    """B z = Y on stochastic LEFT monitoring points used by the payoff.

    Probe the simulator's linear Gaussian driver, not its nonlinear volatility.
    t=0 has zero variance and t=T is unused by left-point integrated variance.
    """
    if problem.steps < 2:
        raise ValueError("Volterra excursion templates require at least two steps")
    basis_paths = problem.simulate_local(torch.eye(problem.local_dimension, dtype=torch.float64))
    operator = basis_paths.volterra[:, 1:-1].T.contiguous()
    if operator.shape != (problem.steps-1, problem.local_dimension) or not torch.isfinite(operator).all():
        raise ValueError("invalid finite-grid Volterra driver operator")
    if bool((operator.square().sum(dim=1) <= 0).any()):
        raise ValueError("zero variance stochastic monitoring point")
    return operator


def build_volterra_excursion_guide(
    problem: RBergomiBaselineProblem, *, amplitudes: list[float], defensive_mass: float = .1,
    price_amplitudes: list[float] | None = None,
) -> DefensiveFiniteRankGaussianMixture:
    """Uniform time/amplitude mixture of N(kappa*b_i/||b_i||, I).

    This mean is the minimum-energy forcing for b_i.mean=kappa*||b_i||.
    Covariance is UNCHANGED; this is not conditioning Y_i to a fixed value.
    No guarantee that these one-point templates cover all payoff modes is made.
    """
    if not amplitudes or any(not math.isfinite(a) or a <= 0 for a in amplitudes) or not 0 < defensive_mass < 1:
        raise ValueError("invalid excursion guide amplitudes/defensive mass")
    price_amplitudes = [0.] if price_amplitudes is None else price_amplitudes
    if not price_amplitudes or any(not math.isfinite(a) or a < 0 for a in price_amplitudes):
        raise ValueError("invalid local price amplitudes")
    operator = volterra_monitoring_operator(problem)
    units = operator/torch.linalg.vector_norm(operator, dim=1, keepdim=True)
    d = problem.local_dimension
    components = [FiniteRankGaussianComponent.natural(d)]
    price_sign = -1. if problem.rho > 0 else 1.
    for i, unit in enumerate(units):
        # Y_i is at the LEFT endpoint; the next Brownian price innovation is
        # independent of its past. Its future effect on volatility is retained.
        price_direction = torch.zeros(d, dtype=torch.float64)
        price_direction[2*(i+1)] = price_sign
        if abs(float(unit@price_direction)) > 1e-12:
            raise ValueError("Volterra/next-price causality contract violated")
        for amplitude in amplitudes:
            for price_amplitude in price_amplitudes:
                components.append(FiniteRankGaussianComponent(amplitude*unit+price_amplitude*price_direction,
                    torch.empty((d, 0), dtype=torch.float64), torch.empty(0, dtype=torch.float64)))
    weights = torch.full((len(components),), (1-defensive_mass)/(len(components)-1), dtype=torch.float64)
    weights[0] = defensive_mass
    return DefensiveFiniteRankGaussianMixture(tuple(components), weights)
