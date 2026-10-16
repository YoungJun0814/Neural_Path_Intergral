import math

import numpy as np
import pytest
import torch

from experiments.post_audit_canonical_last_pair_probability import seed
from src.path_integral.paired_normalizer_difference import paired_normalizer_difference
from src.path_integral.structural_v2_conditional_reference import LastPairCache


def test_last_pair_gaussian_cdf_convolution_matches_independent_quadrature():
    a = torch.tensor([.3, -.7, 1.5], dtype=torch.float64)
    b = torch.tensor([.5, -.8, 1.], dtype=torch.float64)
    variance = torch.tensor([.6, .8, .7], dtype=torch.float64)
    cache = LastPairCache(a, b, variance, torch.zeros((3, 1), dtype=torch.float64), torch.zeros((1, 2), dtype=torch.float64))
    nodes, weights = np.polynomial.hermite.hermgauss(100)
    x = torch.tensor(math.sqrt(2)*nodes, dtype=torch.float64)
    w = torch.tensor(weights/math.sqrt(math.pi), dtype=torch.float64)
    exact = (torch.special.ndtr((a[:, None]-b[:, None]*x)/torch.sqrt(variance[:, None]))*w).sum(1)
    torch.testing.assert_close(cache.log_mu_mean().exp(), exact, rtol=1e-12, atol=1e-12)
    streams = [seed(rep, role) for rep in range(8) for role in ("path", "label")]
    assert len(streams) == len(set(streams)) == 16


def test_paired_delta_method_uses_covariance_and_preserves_signed_difference():
    fine = torch.tensor([1., 2., 3., 4.], dtype=torch.float64)
    coarse = .95*fine
    result = paired_normalizer_difference(fine.log(), coarse.log(), comparisons=2)
    assert result["relative_difference"] == pytest.approx(.05)
    assert result["paired_relative_se"] < 1e-15
    assert result["pass"]
    reversed_result = paired_normalizer_difference(coarse.log(), fine.log(), comparisons=2)
    assert reversed_result["relative_difference"] == pytest.approx(-.05/.95)
    with pytest.raises(ValueError):
        paired_normalizer_difference(fine.log()[:1], coarse.log()[:1], comparisons=2)
