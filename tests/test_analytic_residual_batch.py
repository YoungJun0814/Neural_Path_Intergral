import math
from dataclasses import replace

import numpy as np
import pytest
import torch
from scipy.special import log_ndtr

from src.path_integral.analytic_residual_batch import (
    JointBlindResidualSlice,
    batched_moments,
    critical_points,
)
from src.path_integral.cached_volterra_terminal import CachedVolterraTerminal
from src.path_integral.residual_log_quadrature import log_gaussian_moments, slice_log_probability
from src.path_integral.residual_structural_oracle import LastVarianceResidualSlice
from tests.test_p1_recovery_p2_parallel import problem


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def profile():
    return LastVarianceResidualSlice(32, 0., 1., -4., 1., .2, 1., -.5, 0.)


def test_two_turning_points_are_all_found_and_split():
    p = profile()
    roots = critical_points(p)
    assert len(roots["points"]) == 2 and not roots["near_multiple_root"]
    for u in roots["points"]:
        values = slice_log_probability(p, np.array([u-1e-5, u, u+1e-5]))
        assert abs(values[2]-values[0]) < 1e-8
    result = batched_moments([p])
    reference = log_gaussian_moments(lambda u: slice_log_probability(p, u))
    assert result["all_relative_numerically_resolved"]
    assert result["rows"][0]["panel_count"] == 34
    for m, old in zip(result["rows"][0]["moments"], reference["moments"], strict=True):
        assert abs(m["log_estimate"]-old["log_estimate"]) < 1e-9
        assert m["log_monotone_envelope_lower"] <= old["log_estimate"] <= m["log_monotone_envelope_upper"]


@pytest.mark.parametrize("log_level", [-2., 2.])
def test_beta_zero_known_answer_constant_CDF(log_level):
    p = replace(profile(), beta=0., log_level=log_level)
    x = (p.log_level-p.preceding_terminal+.5-p.rho*p.last_dw)/math.sqrt((1-p.rho**2)*2)
    result = batched_moments([p])
    assert result["all_relative_numerically_resolved"]
    for power, m in enumerate(result["rows"][0]["moments"], 1):
        assert m["log_estimate"] == pytest.approx(power*float(log_ndtr(x)), abs=1e-12)


def test_near_double_root_is_not_silently_called_certified():
    p = replace(profile(), preceding_terminal=-2.5, last_dw=2.)
    assert critical_points(p)["near_multiple_root"]
    assert not batched_moments([p])["all_relative_numerically_resolved"]


def test_tail_dominance_does_not_pass_with_tiny_positive_integral():
    p = replace(profile(), preceding_terminal=1000., beta=0.)
    result = batched_moments([p])
    assert not result["all_relative_numerically_resolved"]
    assert result["rows"][0]["moments"][0]["relative_error_proxy_plus_tail"] is None


def test_budget_charge_includes_padding_and_happens_before_evaluation():
    charges = []
    result = batched_moments([profile(), replace(profile(), beta=0.)], charge=charges.append)
    assert sum(charges) == result["scalar_evaluations_including_padding_and_endpoints"]
    assert charges == [2*34*16, 2*34*32, 2*34*64, 2*34*2]


@pytest.mark.parametrize("p", [replace(profile(), beta=-1.), replace(profile(), dt=0.), replace(profile(), rho=1.)])
def test_invalid_profile_rejected(p):
    with pytest.raises(ValueError):
        batched_moments([p])


def test_joint_57_61_slice_matches_full_hybrid_law_and_unused_63():
    cache = CachedVolterraTerminal.create(problem(32))
    z = torch.linspace(-.3, .3, 64, dtype=torch.float64)[None]
    joint = JointBlindResidualSlice.create(cache, z)
    pair = np.array([(a, b) for a in (-2., 0., 2.) for b in (-2., 0., 2.)])
    full = z.repeat(9, 1)
    full[:, [57, 61]] = torch.tensor(pair)
    np.testing.assert_allclose(joint.log_probability(pair), cache.log_probability(full).numpy(), rtol=1e-12, atol=1e-10)
    full[:, 63] = torch.arange(-4, 5, dtype=torch.float64)
    np.testing.assert_allclose(joint.log_probability(pair), cache.log_probability(full).numpy(), rtol=1e-12, atol=1e-10)


def test_local_odd_residual_changes_only_its_single_variance():
    cache = CachedVolterraTerminal.create(problem(32))
    z = torch.zeros((1, 64), dtype=torch.float64)
    for coordinate, index in ((57, 29), (61, 31), (63, 32)):
        points = z.repeat(2, 1)
        points[1, coordinate] = 1.
        _, _, variance = cache.quantities(points)
        changed = torch.where((variance[1]-variance[0]).abs() > 1e-12)[0].tolist()
        assert changed == [index]
