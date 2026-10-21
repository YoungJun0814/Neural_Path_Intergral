import math

import pytest
import torch
from scipy.integrate import quad

from experiments.post_audit_exact_residual_structure import Budget, exact_toy, scalar_moments
from src.path_integral.cached_volterra_terminal import CachedVolterraTerminal
from src.path_integral.gaussian_bump_oracle import GaussianBumpOracle
from src.path_integral.physical_window_guide import PhysicalWindowGuide
from src.path_integral.residual_structural_oracle import (
    LastVarianceResidualSlice,
    bump_shift_moments,
    gaussian_log_rho,
    untouched_bump_floor,
)
from tests.test_p1_recovery_p2_parallel import problem


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("center,width,mean", [(3., 1., 1.5), (-2., .75, .5), (0., 2., -1.)])
def test_exact_bump_shift_formula_matches_independent_density_integration(center, width, mean):
    result = bump_shift_moments((center,), width, (mean,))
    toy = GaussianBumpOracle((1.,), ((center,),), (width,))
    assert result["log_mu"] == pytest.approx(toy.log_endpoint(1), abs=1e-13)

    def integrand(z):
        return math.exp(-(z-center)**2/width**2-.5*z*z-mean*z+.5*mean*mean)/math.sqrt(2*math.pi)
    value = quad(integrand, -15, 15, epsabs=1e-13, epsrel=1e-10)[0]
    assert math.log(value) == pytest.approx(result["log_m2"], abs=1e-10)
    assert untouched_bump_floor((center,), width) == pytest.approx(toy.log_endpoint(2)-2*toy.log_endpoint(1))


def test_mean_error_and_covariance_floor_are_separate_exact_terms():
    optimum = bump_shift_moments((3., -1.), .75, (3/1.5625, -1/1.5625))
    wrong = bump_shift_moments((3., -1.), .75, (0., 0.))
    assert optimum["log_mean_mismatch_penalty"] == 0
    assert optimum["log_covariance_family_floor"] > 0
    assert wrong["log_covariance_family_floor"] == optimum["log_covariance_family_floor"]
    assert wrong["log_relative_second_moment"] > optimum["log_relative_second_moment"]


@pytest.mark.parametrize("variance,proposal_variance,a,b", [(.3, 1., 1., -2.), (1., 1., 0., 1.), (.5, .5, 1., 1.)])
def test_gaussian_rho_formula_and_finite_condition(variance, proposal_variance, a, b):
    log_rho = gaussian_log_rho((a,), variance, (b,), proposal_variance)

    def integrand(z):
        return math.sqrt(2*math.pi*proposal_variance)/(2*math.pi*variance)*math.exp(-(z-a)**2/variance+(z-b)**2/(2*proposal_variance))
    value = quad(integrand, -20, 20, epsabs=1e-10, epsrel=1e-10)[0]
    assert math.log(value) == pytest.approx(log_rho, abs=1e-9)
    assert math.isinf(gaussian_log_rho((0.,), 2., (0.,), 1.))


def test_exact_good_q_bad_G_counterexample_not_a_monte_carlo_signal():
    result = exact_toy(Budget())
    assert result["pass"]
    assert result["event_rse"] < .005
    assert result["risk_evaluation_rse_bad"] > 1
    assert result["risk_evaluation_rse_matched"] == pytest.approx(0, abs=1e-12)
    assert result["untouched_residual_log_floor"] > math.log(50)


@pytest.mark.parametrize("steps", [32, 64])
def test_last_used_residual_slice_matches_full_law_and_unused_coordinate(steps):
    cache = CachedVolterraTerminal.create(problem(steps))
    z = torch.linspace(-.3, .3, 2*steps, dtype=torch.float64)[None]
    profile = LastVarianceResidualSlice.create(cache, z)
    values = torch.tensor([-4., -2., 0., 2., 4.], dtype=torch.float64)
    points = z.repeat(5, 1)
    points[:, 2*steps-3] = values
    torch.testing.assert_close(cache.log_probability(points), profile.log_probability(values), rtol=1e-12, atol=1e-10)
    assert float(profile.log_probability(values).max()-profile.log_probability(values).min()) > 1
    points = z.repeat(5, 1)
    points[:, 2*steps-1] = values
    torch.testing.assert_close(cache.log_probability(points), profile.log_probability(z[0, 2*steps-3].repeat(5)), rtol=1e-12, atol=1e-10)


def test_physical_N32_means_leave_live_and_unused_residual_coordinates_unchanged():
    cache = CachedVolterraTerminal.create(problem(32))
    guide = PhysicalWindowGuide.create(cache.problem, kind="mixed-window", amplitudes=[2., 4., 6., 8.], price_amplitudes=[0., 2., 4.])
    assert torch.where(guide.spec.means.abs().max(0).values == 0)[0].tolist() == [57, 61, 63]
    z = torch.zeros((5, 64), dtype=torch.float64)
    z[:, 61] = torch.arange(-2, 3, dtype=torch.float64)
    logs = guide.log_q_over_p(z)
    assert torch.equal(logs, logs[0].expand_as(logs))
    assert not torch.equal(cache.log_probability(z), cache.log_probability(z)[0].expand(5))


def test_scalar_oracle_records_conditional_moment_and_resolution_not_global_risk():
    cache = CachedVolterraTerminal.create(problem(32))
    profile = LastVarianceResidualSlice.create(cache, torch.zeros((1, 64), dtype=torch.float64))
    budget = Budget()
    result = scalar_moments(profile, budget)
    assert result["adaptive"][1] >= result["adaptive"][0]**2-1e-14
    assert budget.physical == 0 and 336 <= budget.scalar < 50000
    assert result["not_global_m2_or_production_quadrature_accuracy"]


@pytest.mark.parametrize("action", ["physical", "scalar"])
def test_oracle_budget_never_extends(action):
    b = Budget()
    b.charge(**{action: 100 if action == "physical" else 50000})
    with pytest.raises(TimeoutError):
        b.charge(**{action: 1})


def test_bad_toy_and_profile_inputs_rejected():
    with pytest.raises(ValueError):
        bump_shift_moments((1.,), 0., (1.,))
    with pytest.raises(ValueError):
        gaussian_log_rho((1.,), 1., (0., 1.), 1.)
    cache = CachedVolterraTerminal.create(problem(32))
    with pytest.raises(ValueError):
        LastVarianceResidualSlice.create(cache, torch.zeros((2, 64), dtype=torch.float64))
    profile = LastVarianceResidualSlice.create(cache, torch.zeros((1, 64), dtype=torch.float64))
    with pytest.raises(ValueError):
        profile.log_probability(torch.tensor([math.nan], dtype=torch.float64))
