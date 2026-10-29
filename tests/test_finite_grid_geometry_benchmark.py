import math
from dataclasses import replace

import pytest
import torch
from scipy.integrate import quad
from scipy.special import logsumexp

from src.path_integral.cached_volterra_terminal import CachedVolterraTerminal
from src.path_integral.finite_grid_geometry_benchmark import (
    bounded_payoff,
    feasibility,
    normal_coordinates,
    reference_units,
    verify_reference_record,
)
from src.path_integral.finite_grid_geometry_protocol import EvaluationBudget
from src.path_integral.gaussian_bump_oracle import GaussianBumpOracle
from src.path_integral.volterra_conditional_payoffs import evaluate_rbergomi_conditional_terminal
from tests.test_finite_grid_geometry_protocol import protocol


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("index", [0, 1, 2, 3])
def test_same_integrand_matches_full_evaluator_for_every_development_task(index):
    task = protocol().tasks[index]
    z = normal_coordinates("conditional_mc", 64, 2*task.steps, 23434+index)
    cache = CachedVolterraTerminal.create(task.problem())
    actual = bounded_payoff(task, cache, z)
    full = evaluate_rbergomi_conditional_terminal(task.problem(), z)
    expected = (full.payoffs.left_probability if task.payoff == "digital"
                else full.payoffs.put_value/(task.spot*task.strike_ratio))
    torch.testing.assert_close(actual, expected, rtol=3e-11, atol=3e-14)
    changed = z.clone()
    changed[:, -1] += 3.
    torch.testing.assert_close(actual, bounded_payoff(task, cache, changed), rtol=1e-13, atol=1e-14)


def test_task_mismatched_cache_is_rejected_before_reusing_strike():
    t = protocol().tasks[0]
    cache = CachedVolterraTerminal.create(t.problem())
    with pytest.raises(ValueError, match="binding"):
        bounded_payoff(replace(t, strike_ratio=.9), cache, torch.zeros(2, 64, dtype=torch.float64))


def test_rqmc_uses_finite_open_cube_and_independent_scrambles():
    a = normal_coordinates("conditional_rqmc", 1024, 4, 534)
    b = normal_coordinates("conditional_rqmc", 1024, 4, 535)
    assert torch.isfinite(a).all() and not torch.equal(a, b)
    assert abs(float(a.mean())) < .005
    with pytest.raises(ValueError, match="power"):
        normal_coordinates("conditional_rqmc", 13, 4, 123)


def test_endpoint_failure_is_not_silently_clipped(monkeypatch):
    def endpoint(*args, **kwargs):
        return torch.zeros(2, 4, dtype=torch.float64)
    monkeypatch.setattr(torch.quasirandom.SobolEngine, "draw_base2", endpoint)
    with pytest.raises(FloatingPointError, match="endpoint"):
        normal_coordinates("conditional_rqmc", 2, 4, 123)


def test_qmc_standard_error_counts_scrambles_not_points():
    p = replace(protocol(), pilot_units=4, pilot_power=4)
    budget = EvaluationBudget(p.limit("FR2"))
    row = reference_units(p, "D01", "conditional_rqmc", budget)
    expected = torch.tensor(row["unit_means"], dtype=torch.float64).std()/math.sqrt(4)
    assert row["se"] == float(expected)
    assert row["independent_unit_count"] == 4
    assert row["path_count"] == 64 and budget.evaluations == 64
    assert row["unit_type"] == "independent_scramble"
    verify_reference_record(p, "D01", row)
    with pytest.raises(ValueError, match="statistics"):
        verify_reference_record(p, "D01", {**row, "se": row["se"]/100})
    with pytest.raises(ValueError, match="role/count/law"):
        verify_reference_record(p, "D01", {**row, "path_count": 1000})


def test_known_constant_law_limit_recovers_black_scholes_cdf():
    task = replace(protocol().tasks[0], eta=1e-12, rho=0.)
    cache = CachedVolterraTerminal.create(task.problem())
    z = normal_coordinates("conditional_mc", 64, 2*task.steps, 678)
    actual = bounded_payoff(task, cache, z)
    argument = (math.log(task.strike_ratio)+.5*task.xi*task.maturity)/math.sqrt(task.xi*task.maturity)
    expected = .5*math.erfc(-argument/math.sqrt(2))
    torch.testing.assert_close(actual, torch.full_like(actual, expected), rtol=1e-10, atol=1e-13)


def test_reference_diagnostic_cannot_promote_imprecise_or_wrong_estimands():
    p = protocol()
    mc = {"method": "conditional_mc", "mean": .1, "se": .0001, "rse": .001,
          "estimand_digest": "same"}
    rqmc = {**mc, "method": "conditional_rqmc"}
    assert feasibility(p, mc, rqmc)["status"] == "REFERENCE_FEASIBLE_DEVELOPMENT_ONLY"
    assert not feasibility(p, mc, rqmc)["P2_qualified"]
    assert feasibility(p, mc, {**rqmc, "rse": .5})["status"] == "UNRESOLVED_REFERENCE"
    with pytest.raises(ValueError, match="binding"):
        feasibility(p, mc, {**rqmc, "estimand_digest": "other"})


def test_small_event_mass_region_can_dominate_risk_with_a_defensive_density():
    # A deterministic negative control, not a new theorem or Volterra evidence.
    toy = GaussianBumpOracle((.001, .999), ((0.,), (4.5,)), (.5, .2))

    def integrand(z, power):
        log_g = float(logsumexp([math.log(.001)-z*z/(2*.5**2),
                                math.log(.999)-(z-4.5)**2/(2*.2**2)]))
        log_p = -.5*z*z-.5*math.log(2*math.pi)
        log_d = float(logsumexp([math.log(.1), math.log(.9/.4)+.5*z*z*(1-1/.4**2)]))
        return math.exp(power*log_g+log_p-(log_d if power == 2 else 0))

    mu, mu_error = quad(lambda z: integrand(z, 1), -12., 12., epsabs=1e-15, epsrel=1e-11)
    risk, risk_error = quad(lambda z: integrand(z, 2), -12., 12., epsabs=1e-15, epsrel=1e-11)
    event_tail = quad(lambda z: integrand(z, 1), 3.5, 12., epsabs=1e-15)[0]
    risk_tail = quad(lambda z: integrand(z, 2), 3.5, 12., epsabs=1e-15)[0]
    assert abs(mu-math.exp(toy.log_endpoint(1))) < 1e-13
    assert mu_error < 1e-12 and risk_error < 1e-12
    assert event_tail/mu < .05
    assert risk_tail/risk > .90
    # The omitted |z|>12 tail is bounded by 10*P(|Z|>12) for this q>=.1p.
    assert 10*math.erfc(12/math.sqrt(2)) < 1e-30
