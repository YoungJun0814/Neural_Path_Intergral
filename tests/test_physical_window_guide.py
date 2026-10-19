import math
from dataclasses import replace

import pytest
import torch

from src.path_integral.physical_window_guide import PhysicalWindowGuide
from src.path_integral.volterra_excursion_guide import (
    build_volterra_excursion_guide,
    volterra_monitoring_operator,
)
from tests.test_p1_recovery_p2_parallel import problem

KWARGS = {"amplitudes": [2., 4., 6., 8.], "price_amplitudes": [0., 2., 4.], "defensive_mass": .1}


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("rho", [-.7, .7])
def test_g0_reuses_original_gaussian_mixture_and_matches_every_dense_component(rho):
    p = replace(problem(32), rho=rho)
    guide = PhysicalWindowGuide.create(p, kind="point", **KWARGS)
    dense = build_volterra_excursion_guide(p, **KWARGS)
    torch.testing.assert_close(guide.spec.means, torch.stack([c.mean for c in dense.components]), rtol=2e-12, atol=2e-12)
    z = torch.randn((100, 64), dtype=torch.float64, generator=torch.Generator().manual_seed(7300))
    torch.testing.assert_close(guide.log_q_over_p(z), dense.log_q_over_p(z), rtol=2e-12, atol=2e-12)


def test_g1_adjoint_matches_independent_dense_rows_and_strict_causal_energy():
    p = problem(32)
    guide = PhysicalWindowGuide.create(p, kind="mixed-window", **KWARGS)
    expected_raw = guide.feature_weights[:, :-1] @ volterra_monitoring_operator(p)
    torch.testing.assert_close(guide.vol_directions*guide.feature_variances.sqrt()[:, None], expected_raw, rtol=2e-12, atol=2e-12)
    assert guide.template_kinds.count("point") == 15 and guide.template_kinds.count("window") == 16
    assert guide.spec.components == 373
    for i, centre in enumerate(guide.centres):
        assert torch.count_nonzero(guide.vol_directions[i, 2*centre:]) == 0
        assert torch.count_nonzero(guide.price_directions[i, :2*centre]) == 0
    norms = guide.spec.means[1:].square().sum(1).reshape(31, 4, 3)
    expected = torch.tensor(KWARGS["amplitudes"], dtype=torch.float64).square()[:, None]+torch.tensor(KWARGS["price_amplitudes"], dtype=torch.float64).square()[None]
    torch.testing.assert_close(norms, expected[None].expand_as(norms), rtol=2e-12, atol=2e-12)


def test_window_price_component_keeps_physical_effect_under_refinement():
    effects = []
    for n in (32, 64, 128):
        p = problem(n)
        g = PhysicalWindowGuide.create(p, kind="mixed-window", **KWARGS)
        price_effect = p.rho*math.sqrt(p.xi*p.step_dt)*2*g.price_directions[:, ::2].sum(1)
        durations = torch.tensor(g.price_window_steps, dtype=torch.float64)*p.step_dt
        torch.testing.assert_close(price_effect, -abs(p.rho)*torch.sqrt(p.xi*durations)*2, rtol=2e-12, atol=2e-12)
        effects.append(price_effect)
        point = PhysicalWindowGuide.create(p, kind="point", **KWARGS)
        point_effect = p.rho*math.sqrt(p.xi*p.step_dt)*2*point.price_directions[:, ::2].sum(1)
        torch.testing.assert_close(point_effect, torch.full((31,), -abs(p.rho)*math.sqrt(p.xi*p.step_dt)*2, dtype=torch.float64), rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(effects[0], effects[2], rtol=2e-12, atol=2e-12)


def test_brownian_price_embedding_separates_point_noise_from_physical_windows():
    errors = []
    for n in (32, 64, 128):
        for kind in ("point", "mixed-window"):
            coarse = PhysicalWindowGuide.create(problem(n), kind=kind, **KWARGS)
            fine = PhysicalWindowGuide.create(problem(2*n), kind=kind, **KWARGS)
            embedded = coarse.price_directions[:, ::2].repeat_interleave(2, dim=1)/math.sqrt(2)
            error = (embedded-fine.price_directions[:, ::2]).square().sum(1)
            if kind == "point":
                torch.testing.assert_close(error, torch.full((31,), 2-math.sqrt(2), dtype=torch.float64), rtol=2e-12, atol=2e-12)
            else:
                torch.testing.assert_close(error[:15], torch.zeros(15, dtype=torch.float64), rtol=0, atol=2e-12)
                assert float(error[15:].max()) <= 16/(2*n)+2e-12
                errors.append(float(error[15:].max()))
    assert errors[-1] <= errors[0]


def test_sampler_full_balance_density_and_large_grid_geometry():
    guide = PhysicalWindowGuide.create(problem(4096), kind="mixed-window", **KWARGS)
    assert guide.spec.means.shape == (373, 8192) and guide.maximum_causal_roundoff < 1e-11
    labels = torch.multinomial(guide.spec.weights, 8, replacement=True, generator=torch.Generator().manual_seed(7301))
    base = torch.randn((8, 8192), dtype=torch.float64, generator=torch.Generator().manual_seed(7302))
    torch.testing.assert_close(guide.sample(8, path_seed=7302, label_seed=7301), base+guide.spec.means[labels], rtol=0, atol=0)
    density = guide.log_q_over_p(base)
    assert torch.isfinite(density).all() and (density >= math.log(.1)-1e-12).all()
    with pytest.raises(ValueError):
        PhysicalWindowGuide.create(problem(16), kind="point", **KWARGS)
