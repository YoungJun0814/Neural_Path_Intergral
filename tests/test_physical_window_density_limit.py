"""Independent Gaussian toy oracles; not a Volterra performance certificate."""

import numpy as np
import pytest
from scipy.special import ndtr, roots_hermitenorm


def second_moment(duration: float, order: int = 120) -> float:
    z, weights = roots_hermitenorm(order)
    weights = weights/np.sqrt(2*np.pi)
    terminal = np.sqrt(duration)*z[:, None]+np.sqrt(1-duration)*z[None, :]
    payoff = ndtr((-1-terminal)/.5)
    density = .1+.9*np.exp(-z-.5)
    return float(np.einsum("i,j,ij->", weights, weights, payoff**2/density[:, None]))


def test_vanishing_cell_shift_retains_inverse_density_noise_in_gaussian_toy():
    z, weights = roots_hermitenorm(120)
    weights = weights/np.sqrt(2*np.pi)
    prior = float(weights@ndtr((-1-z)/.5)**2)
    inverse_density = float(weights@(1/(.1+.9*np.exp(-z-.5))))
    independent_limit = prior*inverse_density
    assert inverse_density > 1
    assert second_moment(0) == pytest.approx(independent_limit, rel=2e-13)
    assert second_moment(2**-24) == pytest.approx(independent_limit, rel=.005)
    assert second_moment(.25) < independent_limit


def test_fixed_window_gaussian_quadrature_is_numerically_resolved():
    assert second_moment(.25, 120) == pytest.approx(second_moment(.25, 160), rel=2e-7)
