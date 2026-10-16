import math

import pytest
import torch
from scipy.integrate import dblquad
from scipy.stats import norm

from src.path_integral.conditional_excursion_reference import ConditionalExcursionReference


def reference():
    return ConditionalExcursionReference(torch.tensor([[1., 0.]], dtype=torch.float64), torch.tensor([1]),
                                         thresholds=(1.,), price_threshold=.5)


def test_excursion_density_independent_overlap_formula():
    r = reference()
    points = torch.tensor([[0., 0.], [2., 0.], [2., 2.]], dtype=torch.float64)
    t, s = norm.sf(1), norm.sf(.5)
    expected = torch.tensor([.1, .1+.45/t, .1+.45/t+.45/(t*s)], dtype=torch.float64)
    torch.testing.assert_close(r.log_q_over_p(points).exp(), expected)
    def integrand(y, x):
        density = .1+(.45/t if x >= 1 else 0)+(.45/(t*s) if x >= 1 and y >= .5 else 0)
        return norm.pdf(x)*norm.pdf(y)*density
    mass = 0
    # Split the discontinuity exactly instead of trusting unsplit quadrature.
    for lower_x, upper_x in ((-10, 1), (1, 10)):
        for lower_y, upper_y in ((-10, .5), (.5, 10)):
            mass += dblquad(integrand, lower_x, upper_x, lambda x, lo=lower_y: lo,
                            lambda x, hi=upper_y: hi, epsabs=1e-10)[0]
    assert abs(mass-1) < 1e-9


def test_excursion_sampling_and_corrected_expectation():
    r = reference()
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        z = r.sample(150000, path_seed=507, label_seed=508)
        mean_x = .9*norm.pdf(1)/norm.sf(1)
        mean_y = .45*norm.pdf(.5)/norm.sf(.5)
        assert abs(float(z[:, 0].mean())-mean_x) < .014
        assert abs(float(z[:, 1].mean())-mean_y) < .014
        weights = torch.exp(-r.log_q_over_p(z))
        assert abs(float(weights.mean())-1) < .02
        value = torch.exp(-.5*z.square().sum(1))*weights
        assert abs(float(value.mean())-.5) < .01
        assert bool((r.log_q_over_p(z) >= math.log(.1)-1e-12).all())
        with pytest.raises(ValueError):
            r.sample(10, path_seed=1, label_seed=1)
    finally:
        torch.set_num_threads(old)


def test_excursion_orthogonal_rotation_and_price_sign():
    u = torch.tensor([[1/math.sqrt(2), 1/math.sqrt(2), 0.]], dtype=torch.float64)
    r = ConditionalExcursionReference(u, torch.tensor([2]), price_sign=-1., thresholds=(2.,))
    z = r.sample(40000, path_seed=811, label_seed=812)
    assert float(z[:, 2].mean()) < -.6
    assert float((z @ u.T).mean()) > 2.
    with pytest.raises(ValueError):
        ConditionalExcursionReference(u, torch.tensor([0]))
