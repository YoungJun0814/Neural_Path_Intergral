from dataclasses import replace

import pytest
import torch

from src.path_integral.cached_volterra_terminal import CachedVolterraTerminal
from src.path_integral.moment_matched_volterra_terminal import (
    MomentMatchedVolterraTerminal,
    historical_cell_squared_errors,
)
from src.path_integral.paired_volterra_gradient_reference import PairedVolterraProbability
from tests.test_p1_recovery_p2_parallel import problem


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("hurst", [.01, .05, .2, .49])
def test_moment_matching_is_exact_marginal_not_original_law(hurst):
    p = replace(problem(64), hurst=hurst)
    original, alternative = CachedVolterraTerminal.create(p), MomentMatchedVolterraTerminal.create(p)
    expected = (torch.arange(65, dtype=torch.float64)*p.step_dt).pow(2*hurst)
    torch.testing.assert_close(alternative.kernel.volterra_variance, expected, rtol=2e-12, atol=1e-14)
    assert torch.all(original.kernel.volterra_variance[2:] < expected[2:])
    torch.testing.assert_close(original.kernel.local_cholesky, alternative.kernel.local_cholesky, rtol=0, atol=0)
    # No historical original kernel mutation and distinct laws at nonzero eta.
    torch.testing.assert_close(original.kernel.historical_kernel, CachedVolterraTerminal.create(p).kernel.historical_kernel, rtol=0, atol=0)
    z = torch.randn((4, 128), dtype=torch.float64, generator=torch.Generator().manual_seed(6700))
    assert float((original.log_probability(z)-alternative.log_probability(z)).abs().max()) > 1e-9


def test_rms_kernel_l2_error_at_most_twice_the_mean_weight_error():
    for H in (.01, .05, .2, .49):
        for lag in (2, 3, 8, 64, 512):
            mse, rms = historical_cell_squared_errors(H, lag)
            assert mse <= rms <= 2*mse+1e-15
    with pytest.raises(ValueError):
        historical_cell_squared_errors(.05, 1)


def test_moment_matched_price_remains_adapted_and_joint_gradient_correct():
    p = problem(32)
    old = PairedVolterraProbability.create(p)
    paired = replace(old, fine=MomentMatchedVolterraTerminal.create(p),
                     coarse=MomentMatchedVolterraTerminal.create(replace(p, steps=16)))
    z = .3*torch.randn((3, paired.dimension), dtype=torch.float64, generator=torch.Generator().manual_seed(6701))
    values = paired.log_components(z)
    torch.testing.assert_close(values, torch.cat([paired.log_components(row[None]) for row in z]), rtol=2e-11, atol=2e-11)
    x = z.clone().requires_grad_(True)
    target = torch.logsumexp(paired.log_components(x), 1)
    grad = torch.autograd.grad(target.sum(), x)[0]
    direction = torch.randn(z.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(6702))
    direction /= torch.linalg.vector_norm(direction, dim=1, keepdim=True)
    eps = 1e-5
    fd = (torch.logsumexp(paired.log_components(z+eps*direction), 1)
          - torch.logsumexp(paired.log_components(z-eps*direction), 1))/(2*eps)
    torch.testing.assert_close(fd, (grad*direction).sum(1), rtol=2e-6, atol=2e-6)
    # The last local pair changes only the price Brownian increment, not left vol.
    changed = z[:, :64].clone()
    changed[:, -2:] += 3.
    _, integrated, variance = paired.fine.quantities(z[:, :64])
    _, changed_integrated, changed_variance = paired.fine.quantities(changed)
    torch.testing.assert_close(integrated, changed_integrated, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(variance[:, :-1], changed_variance[:, :-1], rtol=2e-12, atol=2e-12)


def test_constant_volatility_limit_keeps_fine_coarse_identity():
    p = replace(problem(32), eta=1e-12)
    old = PairedVolterraProbability.create(p)
    paired = replace(old, fine=MomentMatchedVolterraTerminal.create(p),
                     coarse=MomentMatchedVolterraTerminal.create(replace(p, steps=16)))
    z = torch.randn((256, paired.dimension), dtype=torch.float64, generator=torch.Generator().manual_seed(6703))
    values = paired.log_components(z)
    torch.testing.assert_close(values[:, 0], values[:, 1], rtol=0, atol=1e-9)
