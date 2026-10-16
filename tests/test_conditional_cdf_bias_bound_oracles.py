import math

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_gaussian_lognormal_difference_identities_and_variance_bound():
    nodes, weights = np.polynomial.hermite.hermgauss(60)
    nodes = torch.tensor(2**.5*nodes, dtype=torch.float64)
    weights = torch.tensor(weights/math.pi**.5, dtype=torch.float64)
    mesh = torch.cartesian_prod(nodes, nodes)
    product_weights = torch.outer(weights, weights).flatten()
    eta, xi, M = 2., .04, 1.
    bound_coefficient = xi**2*math.exp(eta**2*M)*(eta**2+eta**4*M*math.cosh(.5*eta**2*M))
    for V, VN, C in ((.5, .8, .1), (.7, .7, .65), (.001, 1., .02), (.5, .8, -.5)):
        covariance = torch.tensor([[V, C], [C, VN]], dtype=torch.float64)
        y = mesh @ torch.linalg.cholesky(covariance).T
        v = xi*torch.exp(eta*y-.5*eta**2*torch.tensor([V, VN], dtype=torch.float64))
        e2 = V+VN-2*C
        sigma_error = float(((v[:, 0].sqrt()-v[:, 1].sqrt()).square()*product_weights).sum())
        expected_sigma = 2*xi*(-math.expm1(-eta**2*e2/8))
        assert sigma_error == pytest.approx(expected_sigma, rel=2e-12, abs=1e-14)
        assert sigma_error <= xi*eta**2*e2/4+1e-14
        variance_error = float(((v[:, 0]-v[:, 1]).square()*product_weights).sum())
        expected_v = xi**2*(math.exp(eta**2*V)+math.exp(eta**2*VN)-2*math.exp(eta**2*C))
        assert variance_error == pytest.approx(expected_v, rel=2e-12, abs=1e-14)
        assert variance_error <= bound_coefficient*e2+1e-14


def test_inverse_integrated_variance_jensen_and_gaussian_mgf_bound():
    xi, T, eta, M = .04, 1., 1.5, 1.
    A = torch.tensor([[.6, .2], [-.1, .5], [.4, .3]], dtype=torch.float64)
    variances = A.square().sum(1)
    z = torch.randn((100, 2), dtype=torch.float64, generator=torch.Generator().manual_seed(7200))
    y = z @ A.T
    integrated = xi*T*torch.exp(eta*y-.5*eta**2*variances).mean(1)
    lower = xi*T*torch.exp(eta*y.mean(1)-.5*eta**2*M)
    assert (integrated >= lower-1e-14).all()
    mean_variance = float(A.mean(0).square().sum())
    for r in (.5, 1., 2.):
        gaussian_upper = (xi*T)**-r*math.exp(.5*r*eta**2*M+.5*r*r*eta**2*mean_variance)
        B = (xi*T)**-r*math.exp(.5*eta**2*M*(r+r*r))
        assert gaussian_upper <= B


def test_conditional_cdf_partial_derivative_bounds():
    c = 1-.7**2
    a = torch.linspace(-10, 10, 51, dtype=torch.float64)[:, None].expand(-1, 41).clone().requires_grad_(True)
    integrated = torch.logspace(-3, 3, 41, dtype=torch.float64)[None].expand(51, -1).clone().requires_grad_(True)
    F = torch.special.ndtr(a/torch.sqrt(c*integrated))
    da, di = torch.autograd.grad(F.sum(), (a, integrated))
    assert (da.abs() <= 1/torch.sqrt(2*math.pi*c*integrated)+1e-12).all()
    assert (di.abs() <= 1/(2*math.sqrt(2*math.pi*math.e)*integrated)+1e-12).all()
