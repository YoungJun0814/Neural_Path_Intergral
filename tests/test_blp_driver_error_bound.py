import math

import pytest
import torch
from scipy.integrate import quad

from src.path_integral.blp_driver_error_bound import original_blp_left_driver_mse_bound
from src.path_integral.rbergomi_fft import blp_fft_kernel
from tests.test_p1_recovery_p2_parallel import problem


@pytest.mark.parametrize("hurst", [.05, .2, .45])
def test_increment_integral_projection_error_and_n1_are_below_analytic_bounds(hurst):
    bound = original_blp_left_driver_mse_bound(hurst=hurst, maturity=1., steps=32)
    alpha = hurst-.5
    near = quad(lambda x: ((x+1)**alpha-x**alpha)**2, 0, 1, epsabs=1e-10)[0]
    far = quad(lambda x: (x**alpha*math.expm1(alpha*math.log1p(1/x)))**2, 1, math.inf, epsabs=1e-10)[0]
    assert 1+2*hurst*(near+far) <= bound.increment_constant_upper*(1+1e-10)
    p = problem(32)
    kernel = blp_fft_kernel(p.simulator(), n_steps=32, step_dt=1/32, H=hurst)
    times = torch.arange(33, dtype=torch.float64)/32
    errors = times**(2*hurst)-kernel.volterra_variance
    assert float(errors.min()) >= -1e-12
    assert float(errors.max()) <= bound.projection_constant_upper*(1/32)**(2*hurst)+1e-12
    one = original_blp_left_driver_mse_bound(hurst=hurst, maturity=1., steps=1)
    assert 1/(2*hurst+1) <= one.integrated_mse_upper
    fine = original_blp_left_driver_mse_bound(hurst=hurst, maturity=1., steps=64)
    assert fine.integrated_mse_upper/bound.integrated_mse_upper == pytest.approx(2**(-2*hurst))


@pytest.mark.parametrize("kwargs", [{"hurst": 0., "maturity": 1., "steps": 32},
                                    {"hurst": .05, "maturity": 0., "steps": 32},
                                    {"hurst": .05, "maturity": 1., "steps": True}])
def test_driver_bound_rejects_invalid_model_domain(kwargs):
    with pytest.raises(ValueError):
        original_blp_left_driver_mse_bound(**kwargs)


def test_driver_bound_has_brownian_maturity_scaling_and_rejects_float_range_loss():
    base = original_blp_left_driver_mse_bound(hurst=.2, maturity=1., steps=32)
    longer = original_blp_left_driver_mse_bound(hurst=.2, maturity=2., steps=32)
    assert longer.integrated_mse_upper/base.integrated_mse_upper == pytest.approx(2**1.4)
    with pytest.raises(FloatingPointError):
        original_blp_left_driver_mse_bound(hurst=.2, maturity=5e-324, steps=32)
    with pytest.raises(FloatingPointError):
        original_blp_left_driver_mse_bound(hurst=.2, maturity=1e308, steps=1)
