"""Design checks only: these do not qualify a rare-event reference."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from experiments.post_audit_r1_diagnostics import _problem
from src.path_integral.volterra_conditional_payoffs import evaluate_rbergomi_conditional_terminal


@pytest.mark.parametrize("steps", [1, 4, 32])
@pytest.mark.parametrize("eta", [1.5, 3.0])
def test_last_pair_identity_matches_actual_finite_grid(steps: int, eta: float) -> None:
    problem = _problem({"spot": 100., "maturity": 1., "hurst": .1,
                        "eta": eta, "xi": .04, "rho": -.7},
                       {"id": "design-only", "threshold": 80.}, steps)
    base = torch.randn((12, 2 * steps), dtype=torch.float64,
                       generator=torch.Generator().manual_seed(731)) * .3
    base[:, -2:] = 0.
    paths = problem.simulate_local(base)
    integrated = problem.step_dt * paths.variance[:, :-1].sum(1)
    a = math.log(problem.task.level) - paths.log_spot[:, -1]
    b = problem.rho * torch.sqrt(paths.variance[:, -2] * problem.step_dt)
    c = torch.sqrt((1 - problem.rho**2) * integrated)
    for x, y in [(-2., 3.), (0., -4.), (1.5, 2.)]:
        varied = base.clone()
        varied[:, -2] = x
        varied[:, -1] = y
        actual = evaluate_rbergomi_conditional_terminal(problem, varied)
        torch.testing.assert_close(actual.integrated_variance, integrated,
                                   rtol=2e-12, atol=2e-14)
        torch.testing.assert_close(actual.payoffs.log_left_probability,
                                   torch.special.log_ndtr((a - b * x) / c),
                                   rtol=2e-12, atol=2e-12)


def test_gaussian_cdf_mean_identity_and_square_noncommutation() -> None:
    # Quadrature here cross-checks an analytic identity, not a certified
    # production integration or tail-coverage theorem.
    nodes, weights = np.polynomial.hermite.hermgauss(96)
    x = torch.tensor(nodes * math.sqrt(2), dtype=torch.float64)
    w = torch.tensor(weights / math.sqrt(math.pi), dtype=torch.float64)
    a, b, c = .3, -.7, 1.1
    g = torch.special.ndtr((a - b * x) / c)
    analytic = torch.special.ndtr(torch.tensor(a / math.sqrt(c*c + b*b), dtype=torch.float64))
    torch.testing.assert_close(w @ g, analytic, rtol=1e-13, atol=1e-14)
    assert float(w @ g.square()) > float(analytic.square()) + .01
