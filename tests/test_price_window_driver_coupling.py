import math

import pytest
import torch
from scipy.integrate import quad

from src.path_integral.physical_window_guide import PhysicalWindowGuide
from src.path_integral.volterra_excursion_guide import volterra_monitoring_operator
from tests.test_p1_recovery_p2_parallel import problem


def driver_shift(s, t, width, hurst, b):
    exponent = hurst+.5
    return b*math.sqrt(2*hurst)/(exponent*math.sqrt(width))*(max(s-t, 0)**exponent-max(s-t-width, 0)**exponent)


@pytest.mark.parametrize("kind", ["point", "mixed-window"])
def test_grid_aligned_price_window_blp_mean_matches_continuous_kernel_integral(kind):
    old = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        p = problem(32)
        guide = PhysicalWindowGuide.create(p, kind=kind, amplitudes=[2.], price_amplitudes=[1.75])
        actual = volterra_monitoring_operator(p) @ (1.75*guide.price_directions.T)
        sign = -1 if p.rho > 0 else 1
        expected = torch.tensor([[driver_shift(k*p.step_dt, c*p.step_dt, w*p.step_dt, p.hurst, sign*1.75)
                                  for c, w in zip(guide.centres, guide.price_window_steps, strict=True)]
                                 for k in range(1, 32)], dtype=torch.float64)
        torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)
    finally:
        torch.set_num_threads(old)


@pytest.mark.parametrize("hurst", [.05, .2, .45])
def test_price_window_kernel_integral_and_integrated_energy_bound(hurst):
    t, width, b, maturity = .25, .125, 1.7, 1.
    for s in (t+width/2, t+width, t+2*width, maturity):
        direct = b*math.sqrt(2*hurst/width)*quad(lambda r, upper=s: (upper-r)**(hurst-.5), t, min(s, t+width),
                                               epsabs=1e-10, limit=200)[0]
        assert driver_shift(s, t, width, hurst, b) == pytest.approx(direct, rel=1e-9, abs=1e-10)
    energy = quad(lambda s: driver_shift(s, t, width, hurst, b)**2, t, maturity, points=[t+width], epsabs=1e-10)[0]
    sharp_bound = b*b*quad(lambda r: (maturity-r)**(2*hurst), t, t+width)[0]
    assert energy <= sharp_bound*(1+1e-10) <= b*b*width*(maturity-t)**(2*hurst)*(1+1e-10)


def test_rough_peak_can_decay_much_slower_than_direct_price_effect():
    assert (32/4096)**.05 > .75
    assert math.sqrt(32/4096) < .1
