import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from src.path_integral.cached_volterra_terminal import CachedVolterraTerminal
from src.path_integral.driver_conditioned_terminal import DriverConditionedTerminal
from src.path_integral.fft_driver_marginal_reference import (
    FFTDriverCovariance,
    FFTDriverMarginalReference,
)
from src.path_integral.gaussian_span_marginalization import GaussianMixtureShiftSpec
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.physical_window_guide import PhysicalWindowGuide
from tests.test_p1_recovery_p2_parallel import problem
from tests.test_physical_window_guide import KWARGS


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("n", [4, 16, 32])
def test_fft_covariance_projection_and_pcg_match_independent_dense_rows(n):
    p = problem(n)
    op, dense = FFTDriverCovariance.create(p), DriverConditionedTerminal.create(p)
    full = torch.randn((9, 2*n), dtype=torch.float64, generator=torch.Generator().manual_seed(27001))
    y = op.project(full)
    torch.testing.assert_close(y, full @ dense.driver_rows.T, rtol=1e-12, atol=1e-12)
    covariance = dense.driver_rows @ dense.driver_rows.T
    torch.testing.assert_close(op.covariance(y), y @ covariance, rtol=1e-12, atol=1e-12)
    solved = op.solve(y)
    expected = torch.linalg.solve(covariance, y.T).T
    torch.testing.assert_close(solved.solution, expected, rtol=1e-10, atol=1e-10)
    actual_error = torch.linalg.vector_norm(solved.solution-expected, dim=1)
    assert (actual_error <= solved.error_norm_upper+1e-12).all()
    zero = op.solve(torch.zeros((2, n-1), dtype=torch.float64))
    assert zero.iterations == 0 and torch.count_nonzero(zero.solution) == 0


@pytest.mark.parametrize("rho", [-.7, 0., .7])
def test_conditional_payoff_and_projected_mixture_density_match_independent_qr(rho):
    p = replace(problem(32), rho=rho, task=TerminalThresholdTask(80.))
    guide = PhysicalWindowGuide.create(p, kind="mixed-window", **KWARGS)
    ref, dense = FFTDriverMarginalReference.create(p, guide.spec), DriverConditionedTerminal.create(p)
    z = .5*torch.randn((16, 64), dtype=torch.float64, generator=torch.Generator().manual_seed(27002))
    x, y = dense.project(z), ref.operator.project(z)
    output = ref.evaluate(y)
    mean, variance = dense.conditional_price(x)
    torch.testing.assert_close(output.conditional_mean, mean, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(output.conditional_variance, variance, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(output.log_probability, dense.log_probability(x), rtol=1e-10, atol=1e-10)
    means = guide.spec.means @ dense.white_basis
    log_density = torch.logsumexp(x @ means.T-.5*means.square().sum(1)+guide.spec.weights.log(), 1)
    torch.testing.assert_close(output.log_density_ratio, log_density, rtol=1e-11, atol=1e-11)
    assert output.log_contribution_error_upper.max() < 1e-6
    assert (output.log_contribution <= math.log(10)+1e-10).all()


def test_solver_rejects_unconverged_run_and_autograd_instead_of_returning_approximate_silently():
    op = FFTDriverCovariance.create(problem(32))
    rhs = torch.randn((2, 31), dtype=torch.float64, generator=torch.Generator().manual_seed(27003))
    with pytest.raises(RuntimeError, match="no sample dropped"):
        op.solve(rhs, maximum_iterations=1)
    with pytest.raises(ValueError):
        op.solve(rhs.requires_grad_(True))


def test_marginal_is_is_conditional_raw_is_not_just_a_replaced_payoff():
    p = replace(problem(32), task=TerminalThresholdTask(80.))
    dense, raw = DriverConditionedTerminal.create(p), CachedVolterraTerminal.create(p)
    x = .4*torch.randn((1, 31), dtype=torch.float64, generator=torch.Generator().manual_seed(27004))
    z0 = x @ dense.white_basis.T
    _, _, variance = raw.quantities(z0)
    direction = torch.zeros_like(z0)
    direction[:, ::2] = variance[:, :-1].sqrt()*math.sqrt(p.step_dt)
    direction -= (direction @ dense.white_basis) @ dense.white_basis.T
    direction /= torch.linalg.vector_norm(direction, dim=1, keepdim=True)
    means = torch.stack((torch.zeros(64, dtype=torch.float64),
                         2*dense.white_basis[:, 0]+1.7*direction[0],
                         -dense.white_basis[:, 0]-.9*direction[0]))
    weights = torch.tensor([.1, .45, .45], dtype=torch.float64)
    ref = FFTDriverMarginalReference.create(p, GaussianMixtureShiftSpec(means, weights))
    marginal = ref.evaluate(ref.operator.project(z0)).log_contribution.exp()[0]
    projected = means @ dense.white_basis
    log_post = weights.log()+(x @ projected.T)[0]-.5*projected.square().sum(1)
    posterior = log_post.softmax(0)
    residual_means = means @ direction[0]
    nodes, quadrature_weights = np.polynomial.hermite.hermgauss(120)
    nodes = torch.tensor(nodes*math.sqrt(2), dtype=torch.float64)
    qw = torch.tensor(quadrature_weights/math.sqrt(math.pi), dtype=torch.float64)
    first, second = torch.zeros((), dtype=torch.float64), torch.zeros((), dtype=torch.float64)
    for label in range(3):
        paths = z0+(nodes+residual_means[label])[:, None]*direction
        full_d = torch.logsumexp(paths @ means.T-.5*means.square().sum(1)+weights.log(), 1)
        values = (raw.log_probability(paths)-full_d).exp()
        first += posterior[label]*(values*qw).sum()
        second += posterior[label]*(values.square()*qw).sum()
    torch.testing.assert_close(first, marginal, rtol=2e-10, atol=1e-12)
    assert float(second) >= float(marginal.square())-1e-12


@pytest.mark.parametrize("hurst", [.2, .45])
def test_solver_and_conditioning_remain_consistent_away_from_development_hurst(hurst):
    p = replace(problem(32), hurst=hurst, task=TerminalThresholdTask(80.))
    guide = PhysicalWindowGuide.create(p, kind="mixed-window", **KWARGS)
    ref, dense = FFTDriverMarginalReference.create(p, guide.spec), DriverConditionedTerminal.create(p)
    z = .1*torch.randn((3, 64), dtype=torch.float64, generator=torch.Generator().manual_seed(27005))
    actual = ref.evaluate(ref.operator.project(z))
    torch.testing.assert_close(actual.log_probability, dense.log_probability(dense.project(z)), rtol=1e-10, atol=1e-10)
