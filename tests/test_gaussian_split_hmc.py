import math

import pytest
import torch

from src.path_integral.gaussian_split_hmc import gaussian_split_trajectory, probability_evaluator
from src.path_integral.hamiltonian_probability_smc import estimate_hamiltonian_probability
from src.path_integral.tail_weighted_continuum_bound import (
    log_cdf_weighted_envelope,
    observed_probability_root_diagnostic,
    probability_root_budget,
)
from src.path_integral.weighted_tempered_smc import WeightedSMCConfig


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def evaluate(cap=20.):
    return probability_evaluator(lambda z: (-.5*(z[:, :1]-1.7).square()), maximum_drift_norm=cap)


@pytest.mark.parametrize("cap", [20., .3])
def test_hamiltonian_involution_and_dense_energy_correction_with_capped_force(cap):
    g = torch.Generator().manual_seed(1942026)
    z = torch.randn((12, 6), dtype=torch.float64, generator=g)
    p = torch.randn(z.shape, dtype=torch.float64, generator=g)
    ev = evaluate(cap)
    comp, h, force = ev(z)
    result = gaussian_split_trajectory(z, p, comp, h, force, evaluate=ev, beta=.7, step_size=.35, steps=4)
    back = gaussian_split_trajectory(result.position, result.flipped_momentum,
        result.components, result.potential, result.drift, evaluate=ev, beta=.7, step_size=.35, steps=4)
    torch.testing.assert_close(back.position, z, atol=2e-14, rtol=0)
    torch.testing.assert_close(back.flipped_momentum, p, atol=2e-14, rtol=0)
    dense = (.7*(result.potential-h)-.5*(result.position.square()+result.flipped_momentum.square()
                                        -z.square()-p.square()).sum(1))
    torch.testing.assert_close(result.log_acceptance, dense, atol=1e-13, rtol=0)
    torch.testing.assert_close(back.log_acceptance, -result.log_acceptance, atol=1e-13, rtol=0)
    assert result.callback_rows == 48


def test_phase_space_jacobian_absolute_determinant_one():
    # Independent dense finite-difference oracle, including an active force cap.
    ev = evaluate(.3)
    x = torch.tensor([3., .4, -.2, .7], dtype=torch.float64)
    def transform(v):
        z, p = v[:2][None], v[2:][None]
        c, h, f = ev(z)
        r = gaussian_split_trajectory(z, p, c, h, f, evaluate=ev, beta=.8, step_size=.2, steps=3)
        return torch.cat((r.position[0], r.flipped_momentum[0]))
    eps = 1e-5
    jac = torch.stack([(transform(x+eps*e)-transform(x-eps*e))/(2*eps)
                       for e in torch.eye(4, dtype=torch.float64)], 1)
    assert abs(float(torch.linalg.det(jac))) == pytest.approx(1., abs=1e-9)


def test_prior_rotation_is_exact_in_any_dimension_without_artificial_rejections():
    g = torch.Generator().manual_seed(1952)
    z = torch.randn((4, 8192), dtype=torch.float64, generator=g)
    p = torch.randn(z.shape, dtype=torch.float64, generator=g)
    ev = probability_evaluator(lambda x: torch.zeros((len(x), 1), dtype=torch.float64))
    c, h, f = ev(z)
    r = gaussian_split_trajectory(z, p, c, h, f, evaluate=ev, beta=.8, step_size=.35, steps=4)
    assert torch.equal(r.log_acceptance, torch.zeros(4, dtype=torch.float64))
    torch.testing.assert_close(r.position, math.cos(1.4)*z+math.sin(1.4)*p, atol=3e-15, rtol=1e-14)


@pytest.mark.parametrize("hmc", [False, True])
def test_constant_normalizer_and_equal_work(hmc):
    cfg = WeightedSMCConfig(16, (0., .01, .1, .5, 1.), 4, .35, 1, 8, resample_every=2,
                            resampling_scheme="stratified")
    r = estimate_hamiltonian_probability(lambda z: torch.full((len(z), 1), math.log(.2), dtype=torch.float64),
        dimension=6, config=cfg, use_hamiltonian=hmc)
    assert r.log_normalizer == pytest.approx(math.log(.2), abs=1e-13)
    assert r.callback_rows == r.gradient_requests == 16*(1+3*4)
    assert all(s["initial_ancestors"] == 16 for s in r.stages)


def test_analytic_probability_normalizer_and_replica_mean():
    values = []
    for rep in range(32):
        cfg = WeightedSMCConfig(128, tuple((i/16)**3 for i in range(17)), 4, .35, 1, 1900+rep,
                                resample_every=4, resampling_scheme="stratified")
        r = estimate_hamiltonian_probability(lambda z: -.5*(z[:, :1]-2.).square(), dimension=4,
            config=cfg, use_hamiltonian=True)
        values.append(math.exp(r.log_normalizer))
    assert sum(values)/len(values) == pytest.approx(math.exp(-1)/math.sqrt(2), abs=.015)


@pytest.mark.parametrize("power,monomial", [(2.1, 0.), (2.1, 2.1), (4., 0.), (4., 4.), (10., 10.)])
def test_global_cdf_weighted_envelope_on_dense_s_grid(power, monomial):
    s = torch.linspace(-50, 50, 20001, dtype=torch.float64)
    log_ratio = monomial*s.abs().log()-.5*power*(math.log(2*math.pi)+s*s)-torch.special.log_ndtr(s)
    if monomial == 0:
        log_ratio = -.5*power*(math.log(2*math.pi)+s*s)-torch.special.log_ndtr(s)
    assert float(log_ratio.max()) <= log_cdf_weighted_envelope(power=power, monomial=monomial)


def test_root_bound_still_vacuous_and_observed_probability_cannot_certify():
    kwargs = dict(eta=1.5, xi=.04, rho=-.7, maturity=1., hurst=.05, holder_r=1.25)
    a = probability_root_budget(steps=4096, **kwargs)
    b = probability_root_budget(steps=8192, **kwargs)
    assert b["log_root_error_upper"]-a["log_root_error_upper"] == pytest.approx(-.05*math.log(2))
    assert a["root_error_upper_clipped_at_probability_range"] == 1.
    d = observed_probability_root_diagnostic(-18.1, exponent=a["root_exponent"],
                                             log_root_budget=a["log_root_error_upper"])
    assert d["plug_in_lower"] == 0 and d["plug_in_upper"] == 1
    assert not d["relative_bias_certificate_obtained"]


@pytest.mark.parametrize("r", [1., 0., True, math.inf, 33.])
def test_invalid_holder_exponents_rejected(r):
    with pytest.raises(ValueError):
        probability_root_budget(eta=1.5, xi=.04, rho=-.7, maturity=1., hurst=.05, steps=32, holder_r=r)


def test_invalid_hamiltonian_inputs_rejected():
    z = torch.zeros((2, 4), dtype=torch.float64)
    c, h, f = evaluate()(z)
    for changes in ({"steps": 0}, {"beta": 2.}, {"step_size": math.inf}):
        kw = dict(steps=4, beta=.5, step_size=.35)
        with pytest.raises(ValueError):
            gaussian_split_trajectory(z, z, c, h, f, evaluate=evaluate(), **{**kw, **changes})
