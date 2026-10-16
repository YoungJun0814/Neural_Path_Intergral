from dataclasses import replace

import numpy as np
import pytest
import torch

from src.path_integral.cached_volterra_terminal import CachedVolterraTerminal
from src.path_integral.driver_conditioned_terminal import DriverConditionedTerminal
from src.path_integral.path_functionals import TerminalThresholdTask
from tests.test_p1_recovery_p2_parallel import problem


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("steps", [4, 16, 32])
def test_all_driver_conditional_cdf_matches_scalar_residual_quadrature(steps):
    p = replace(problem(steps), task=TerminalThresholdTask(80.))
    reduced, full = DriverConditionedTerminal.create(p), CachedVolterraTerminal.create(p)
    x = .6*torch.randn((4, steps-1), dtype=torch.float64, generator=torch.Generator().manual_seed(6800))
    z0 = x @ reduced.white_basis.T
    _, _, v = full.quantities(z0)
    price_direction = torch.zeros((4, 2*steps), dtype=torch.float64)
    price_direction[:, ::2] = v[:, :-1].sqrt()*p.step_dt**.5
    residual = price_direction-(price_direction @ reduced.white_basis) @ reduced.white_basis.T
    assert torch.linalg.vector_norm(residual, dim=1).min() > 0
    residual /= torch.linalg.vector_norm(residual, dim=1, keepdim=True)
    nodes, weights = np.polynomial.hermite.hermgauss(120)
    points = torch.tensor(2**.5*nodes, dtype=torch.float64)
    quadrature_weights = torch.tensor(weights/np.pi**.5, dtype=torch.float64)
    paths = (z0[:, None]+residual[:, None]*points[None, :, None]).reshape(-1, 2*steps)
    # All retained drivers stay fixed along the integrated direction.
    torch.testing.assert_close(reduced.project(paths).reshape(4, 120, -1), x[:, None].expand(-1, 120, -1), rtol=0, atol=2e-12)
    numerical = (full.log_probability(paths).exp().reshape(4, 120)*quadrature_weights).sum(1)
    torch.testing.assert_close(reduced.log_probability(x).exp(), numerical, rtol=1e-11, atol=1e-12)


def test_reduced_gaussian_geometry_row_separability_and_gradient():
    reduced = DriverConditionedTerminal.create(problem(32))
    assert reduced.dimension == 31
    torch.testing.assert_close(reduced.driver_factor.T @ reduced.driver_factor,
                               reduced.driver_rows @ reduced.driver_rows.T, rtol=2e-12, atol=2e-12)
    x = .3*torch.randn((3, 31), dtype=torch.float64, generator=torch.Generator().manual_seed(6801))
    torch.testing.assert_close(reduced.log_probability(x), torch.cat([reduced.log_probability(row[None]) for row in x]), rtol=2e-12, atol=2e-12)
    leaf = x.clone().requires_grad_(True)
    grad = torch.autograd.grad(reduced.log_probability(leaf).sum(), leaf)[0]
    direction = torch.randn(x.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(6802))
    direction /= torch.linalg.vector_norm(direction, dim=1, keepdim=True)
    eps = 1e-5
    finite = (reduced.log_probability(x+eps*direction)-reduced.log_probability(x-eps*direction))/(2*eps)
    torch.testing.assert_close(finite, (grad*direction).sum(1), rtol=2e-6, atol=2e-6)
    with pytest.raises(ValueError):
        DriverConditionedTerminal.create(problem(512))
    with pytest.raises(ValueError):
        reduced.log_probability(x[:, :-1])


def test_zero_correlation_reduces_to_integrated_variance_probability():
    p = replace(problem(16), rho=0.)
    reduced, full = DriverConditionedTerminal.create(p), CachedVolterraTerminal.create(p)
    x = torch.randn((10, 15), dtype=torch.float64, generator=torch.Generator().manual_seed(6803))
    torch.testing.assert_close(reduced.log_probability(x), full.log_probability(x @ reduced.white_basis.T), rtol=2e-12, atol=2e-12)
