from __future__ import annotations

import math

import torch

from src.path_integral.residual_stability import (
    check_discrete_log_density_stability,
    log_density_stability_relative_variance_bound,
    rao_blackwell_variance_gap_integrand,
    strict_improvement_certified,
)


def test_discrete_log_density_stability_bound_is_verified() -> None:
    target = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    proposal = torch.tensor([0.12, 0.18, 0.27, 0.43], dtype=torch.float64)
    result = check_discrete_log_density_stability(target, proposal)
    assert result.bound_holds
    assert result.chi_square_divergence <= result.relative_variance_bound
    assert math.isclose(result.relative_variance_bound, math.expm1(result.epsilon), rel_tol=1e-15)


def test_strict_improvement_condition_is_sufficient_not_automatic() -> None:
    bound = log_density_stability_relative_variance_bound(0.1)
    assert strict_improvement_certified(epsilon=0.1, natural_relative_variance=bound + 1e-8)
    assert not strict_improvement_certified(epsilon=0.1, natural_relative_variance=bound)


def test_rao_blackwell_gap_is_nonnegative_and_strict_inside_unit_interval() -> None:
    likelihood = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64)
    conditional = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    gap = rao_blackwell_variance_gap_integrand(likelihood, conditional)
    assert torch.equal(gap, torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64))
    assert bool((gap >= 0.0).all())
