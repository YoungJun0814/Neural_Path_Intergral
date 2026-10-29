import math

import pytest
import torch

from src.path_integral.driver_conditioned_terminal import DriverConditionedTerminal
from src.path_integral.sharp_jensen_continuum_bound import (
    average_driver_moments,
    inverse_moment_log_upper,
    sharp_jensen_bounds,
)
from src.path_integral.tail_weighted_continuum_bound import probability_root_budget


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("n,h", [(2, .05), (8, .1), (32, .3)])
def test_discrete_average_driver_variance_matches_independent_dense_rows(n, h):
    from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
    from src.path_integral.path_functionals import TerminalThresholdTask
    p = RBergomiBaselineProblem("test", TerminalThresholdTask(1.), 100., 1., n, h, 1.5, .04, -.7)
    dense = DriverConditionedTerminal.create(p)
    expected = float((dense.driver_rows.sum(0)/n).square().sum())
    actual = average_driver_moments(hurst=h, maturity=1., steps=n)
    assert actual.discrete_time_average_variance == pytest.approx(expected, rel=1e-12)
    assert actual.discrete_mean_point_variance == pytest.approx(float(dense.driver_rows.square().sum(1).sum()/n))


def test_continuous_average_variance_exact_kernel_integral_and_n1():
    h, t = .05, 2.
    m = average_driver_moments(hurst=h, maturity=t, steps=1)
    # The averaged continuous kernel is sqrt(2H)*(T-r)**(H+.5)/(T*(H+.5)).
    expected = 2*h*t**(2*h+2)/(t*t*(h+.5)**2*(2*h+2))
    assert m.continuous_time_average_variance == pytest.approx(expected, rel=1e-14)
    assert m.discrete_time_average_variance == m.discrete_mean_point_variance == 0.


def test_sharp_inverse_moment_improves_old_envelope_without_claiming_relative_certificate():
    kw = dict(eta=1.5, xi=.04, rho=-.7, maturity=1., hurst=.05, steps=4096, holder_r=1.25)
    m = average_driver_moments(hurst=.05, maturity=1., steps=4096)
    for order in (1., 2., 4., 16.):
        old = -order*math.log(.04)+.5*1.5**2*(order+order**2)
        assert inverse_moment_log_upper(m, order=order, eta=1.5, xi=.04, maturity=1.) < old
    a = probability_root_budget(**kw)
    b = sharp_jensen_bounds(**kw, moments=m)
    assert b["log_root_error_upper"] < a["log_root_error_upper"]
    assert not b["relative_bias_certificate_obtained"]
    assert not b["directed_rounding_certificate"]
