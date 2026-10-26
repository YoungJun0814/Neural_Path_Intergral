import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from src.path_integral.cached_volterra_terminal import CachedVolterraTerminal
from src.path_integral.residual_log_quadrature import log_gaussian_moments, slice_log_probability
from src.path_integral.residual_structural_oracle import LastVarianceResidualSlice
from tests.test_p1_recovery_p2_parallel import problem


@pytest.mark.parametrize("center,width", [(0., 1.), (10., .2), (-7., .75)])
def test_log_scaled_moments_match_independent_exact_gaussian_bump(center, width):
    result = log_gaussian_moments(lambda u: -.5*((u-center)/width)**2)
    assert result["relative_numerically_resolved"]
    for power, moment in enumerate(result["moments"], 1):
        s2 = width**2/power
        exact = .5*math.log(s2/(1+s2))-center**2/(2*(1+s2))
        assert moment["log_estimate"] == pytest.approx(exact, abs=1e-10)
        assert moment["log_lower_proxy"] <= exact+1e-12
        assert moment["log_upper_proxy"] >= exact-1e-12
    ratio = result["log_conditional_relative_second_moment"]
    assert result["log_conditional_ratio_lower_proxy"] <= ratio <= result["log_conditional_ratio_upper_proxy"]


def test_constant_probability_and_exact_zero_variance_identity():
    result = log_gaussian_moments(lambda u: np.full_like(u, math.log(.2)))
    assert result["relative_numerically_resolved"]
    assert result["moments"][0]["log_estimate"] == pytest.approx(math.log(.2), abs=1e-13)
    assert result["log_conditional_relative_second_moment"] == pytest.approx(0., abs=1e-13)


def test_underflow_does_not_become_zero_variance_or_absolute_only_success():
    result = log_gaussian_moments(lambda u: np.full_like(u, -1000.))
    assert not result["relative_numerically_resolved"]
    assert result["moments"][0]["log_estimate"] == pytest.approx(-1000., abs=1e-10)
    assert result["moments"][1]["log_estimate"] == pytest.approx(-2000., abs=1e-10)
    assert result["moments"][0]["log_lower_proxy"] is None
    assert result["log_conditional_ratio_lower_proxy"] is None


def test_tail_dominated_small_domain_cannot_pass_relative_gate():
    result = log_gaussian_moments(lambda u: -.5*((u-8)/.5)**2, cutoff=4.)
    assert result["status"] == "UNRESOLVED"


def test_scalar_evaluations_are_charged_before_callback():
    seen = []

    def charge(n):
        seen.append(n)
        raise TimeoutError("fixed cap")
    with pytest.raises(TimeoutError, match="fixed cap"):
        log_gaussian_moments(lambda u: pytest.fail("must not evaluate"), charge=charge)
    assert seen == [512]


@pytest.mark.parametrize("kwargs", [{"cutoff": 1}, {"cutoff": 16.5}, {"relative_tolerance": 0.}])
def test_invalid_quadrature_policy_rejected(kwargs):
    with pytest.raises(ValueError):
        log_gaussian_moments(lambda u: np.zeros_like(u), **kwargs)


@pytest.mark.parametrize("bad", [1., math.nan, -math.inf])
def test_invalid_log_probability_nodes_rejected(bad):
    with pytest.raises(ValueError):
        log_gaussian_moments(lambda u: np.full_like(u, bad))


def test_independent_numpy_slice_matches_torch_and_rejects_invalid_profile():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cache = CachedVolterraTerminal.create(problem(32))
        profile = LastVarianceResidualSlice.create(cache, torch.zeros((1, 64), dtype=torch.float64))
        u = np.linspace(-16, 16, 129)
        np.testing.assert_allclose(slice_log_probability(profile, u), profile.log_probability(torch.tensor(u)).numpy(), rtol=1e-12, atol=1e-10)
        with pytest.raises(ValueError):
            slice_log_probability(replace(profile, rho=1.), u)
    finally:
        torch.set_num_threads(old)
