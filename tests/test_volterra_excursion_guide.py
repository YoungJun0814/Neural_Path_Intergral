import pytest
import torch

from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.finite_rank_gaussian_transport import (
    DefensiveFiniteRankGaussianMixture,
    FiniteRankGaussianComponent,
    combine_defensive_gaussian_mixtures,
)
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.volterra_excursion_guide import (
    build_volterra_excursion_guide,
    volterra_monitoring_operator,
)


def problem() -> RBergomiBaselineProblem:
    return RBergomiBaselineProblem("toy", TerminalThresholdTask(1.), 100., 1., 8, .05, 1.5, .04, -.7)


def test_driver_linearity_and_minimum_energy_tilt() -> None:
    p = problem()
    b = volterra_monitoring_operator(p)
    z = torch.randn((20, p.local_dimension), dtype=torch.float64, generator=torch.Generator().manual_seed(42))
    assert torch.allclose(z@b.T, p.simulate_local(z).volterra[:, 1:-1], atol=1e-12, rtol=1e-12)
    guide = build_volterra_excursion_guide(p, amplitudes=[2., 4.])
    assert len(guide.components) == 1+2*(p.steps-1)
    for i, row in enumerate(b):
        for j, amplitude in enumerate((2., 4.)):
            mean = guide.components[1+2*i+j].mean
            assert torch.isclose(mean.square().sum(), torch.tensor(amplitude**2, dtype=torch.float64))
            assert torch.isclose(row@mean, amplitude*torch.linalg.vector_norm(row))
    assert abs(guide.defensive_mass-.1) < 1e-14


def test_vectorized_mixed_density_and_gradients_match_components() -> None:
    components = (FiniteRankGaussianComponent.natural(3),
        FiniteRankGaussianComponent(torch.tensor([1., -2., .3], dtype=torch.float64),
            torch.empty((3, 0), dtype=torch.float64), torch.empty(0, dtype=torch.float64)),
        FiniteRankGaussianComponent(torch.ones(3, dtype=torch.float64), torch.eye(3, dtype=torch.float64)[:, :1],
            torch.tensor([.4], dtype=torch.float64)))
    q = DefensiveFiniteRankGaussianMixture(components, torch.tensor([.1, .5, .4], dtype=torch.float64))
    x = torch.randn((50, 3), dtype=torch.float64, generator=torch.Generator().manual_seed(4)).requires_grad_(True)
    actual = q.component_log_q_over_p(x)
    oracle = torch.stack([c.log_q_over_p(x) for c in components], dim=1)
    assert torch.allclose(actual, oracle, atol=1e-12, rtol=1e-12)
    ga = torch.autograd.grad(actual.sum(), x, retain_graph=True)[0]
    go = torch.autograd.grad(oracle.sum(), x)[0]
    assert torch.allclose(ga, go, atol=1e-12, rtol=1e-12)
    draw = q.sample(100, path_seed=3, label_seed=5)
    original_noise = torch.randn((100, 3), dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    for i, c in enumerate(components):
        selected = draw.labels == i
        assert torch.equal(draw.samples[selected], c.transform_standard_normal(original_noise[selected]))


def test_coupled_volterra_and_next_price_constraints() -> None:
    p = problem()
    b = volterra_monitoring_operator(p)
    guide = build_volterra_excursion_guide(p, amplitudes=[4.], price_amplitudes=[0., 2., 4.])
    for i, row in enumerate(b):
        for j, gamma in enumerate((0., 2., 4.)):
            mean = guide.components[1+3*i+j].mean
            assert abs(float(mean.square().sum())-(16+gamma**2)) < 1e-12
            assert abs(float(row@mean)-4*float(torch.linalg.vector_norm(row))) < 1e-12
            assert abs(float(mean[2*(i+1)])-gamma) < 1e-12


@pytest.mark.parametrize("rho", [-.7, 0., .7])
def test_price_orientation_and_safeguard_density_floor(rho: float) -> None:
    p = RBergomiBaselineProblem("orientation", TerminalThresholdTask(1.), 100., 1., 8, .05, 1.5, .04, rho)
    guide = build_volterra_excursion_guide(p, amplitudes=[4.], price_amplitudes=[2.])
    sign = -1. if rho > 0 else 1.
    for i in range(p.steps-1):
        assert guide.components[i+1].mean[2*(i+1)] == 2*sign
    natural = DefensiveFiniteRankGaussianMixture((FiniteRankGaussianComponent.natural(p.local_dimension),),
                                                torch.ones(1, dtype=torch.float64))
    blend = combine_defensive_gaussian_mixtures((natural, guide), (.5, .5))
    z = torch.randn((200, p.local_dimension), dtype=torch.float64,
                    generator=torch.Generator().manual_seed(63))
    assert torch.all(blend.log_q_over_p(z) >= guide.log_q_over_p(z)
                     -torch.log(torch.tensor(2., dtype=torch.float64))-1e-12)
    assert blend.defensive_mass >= .1
    almost_natural = FiniteRankGaussianComponent(torch.full((p.local_dimension,), 1e-15, dtype=torch.float64),
        torch.empty((p.local_dimension, 0), dtype=torch.float64), torch.empty(0, dtype=torch.float64))
    assert almost_natural.is_natural()  # diagnostic tolerance is not a density-floor proof
    with pytest.raises(ValueError, match="natural defensive"):
        DefensiveFiniteRankGaussianMixture((almost_natural,), torch.ones(1, dtype=torch.float64))


@pytest.mark.parametrize("kwargs", [
    {"amplitudes": []}, {"amplitudes": [float("nan")]}, {"amplitudes": [-1.]},
    {"amplitudes": [2.], "price_amplitudes": []},
    {"amplitudes": [2.], "price_amplitudes": [float("inf")]},
    {"amplitudes": [2.], "defensive_mass": 0.},
])
def test_invalid_excursion_guide_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        build_volterra_excursion_guide(problem(), **kwargs)
