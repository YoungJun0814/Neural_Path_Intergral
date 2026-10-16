from dataclasses import replace

import pytest
import torch

from src.path_integral.fft_static_shift_reference import FFTStaticShiftReference
from src.path_integral.volterra_excursion_guide import build_volterra_excursion_guide
from tests.test_p1_recovery_p2_parallel import problem


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("rho", [-.7, .7])
def test_implicit_density_and_every_component_equal_original_dense_guide(rho):
    p = replace(problem(16), rho=rho)
    kwargs = {"amplitudes": [2., 4.], "price_amplitudes": [0., 2.], "defensive_mass": .1}
    fast, dense = FFTStaticShiftReference.create(p, **kwargs), build_volterra_excursion_guide(p, **kwargs)
    assert fast.component_count == len(dense.components)
    labels = [(t, a, b) for t in range(1, 16) for a in range(2) for b in range(2)]
    indices = torch.tensor(labels, dtype=torch.int64)
    shifted = fast.apply_labels(torch.zeros((len(labels), 32), dtype=torch.float64), monitors=indices[:, 0],
                                amplitudes=indices[:, 1], prices=indices[:, 2], natural=torch.zeros(len(labels), dtype=torch.bool))
    means = torch.stack([c.mean for c in dense.components[1:]])
    torch.testing.assert_close(shifted, means, rtol=2e-12, atol=2e-12)
    z = torch.cat((shifted, torch.randn((100, 32), dtype=torch.float64, generator=torch.Generator().manual_seed(6900))))
    torch.testing.assert_close(fast.log_q_over_p(z), dense.log_q_over_p(z), rtol=2e-12, atol=2e-12)
    assert (fast.log_q_over_p(z) >= torch.tensor(.1, dtype=torch.float64).log()-1e-12).all()


def test_sampler_is_exact_declared_gaussian_component_reconstruction():
    fast = FFTStaticShiftReference.create(problem(32), amplitudes=[2., 4.], price_amplitudes=[0., 2.])
    count, ps, ls = 128, 6901, 6902
    g = torch.Generator().manual_seed(ls)
    natural = torch.rand(count, dtype=torch.float64, generator=g) < .1
    monitors = torch.randint(1, 32, (count,), generator=g)
    a = torch.randint(2, (count,), generator=g)
    b = torch.randint(2, (count,), generator=g)
    base = torch.randn((count, 64), dtype=torch.float64, generator=torch.Generator().manual_seed(ps))
    expected = fast.apply_labels(base, monitors=monitors, amplitudes=a, prices=b, natural=natural)
    torch.testing.assert_close(fast.sample(count, path_seed=ps, label_seed=ls), expected, rtol=0, atol=0)
    torch.testing.assert_close(expected[natural], base[natural], rtol=0, atol=0)


def test_large_grid_preparation_has_no_dense_mean_or_driver_matrix():
    fast = FFTStaticShiftReference.create(problem(4096), amplitudes=[2., 4., 6., 8.], price_amplitudes=[0., 2., 4.])
    assert fast.component_count == 49141 and fast.dimension == 8192
    assert fast.cache.kernel.historical_kernel.numel() == 4096
    assert fast.cache.kernel.local_cholesky.shape == (2, 2)
    z = fast.sample(2, path_seed=6903, label_seed=6904)
    assert torch.isfinite(fast.log_q_over_p(z)).all()
    with pytest.raises(ValueError):
        fast.sample(True, path_seed=1, label_seed=2)
