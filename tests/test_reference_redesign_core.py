"""Analytic identities, marginal densities, SE units and invariant-kernel controls."""

from __future__ import annotations

import math

import pytest
import torch

from experiments.post_audit_r1_diagnostics import _problem
from src.path_integral.elliptical_slice_kernel import elliptical_slice_transition
from src.path_integral.finite_rank_gaussian_transport import (
    DefensiveFiniteRankGaussianMixture,
    FiniteRankGaussianComponent,
)
from src.path_integral.gaussian_mixture_marginal import ShiftMixtureMarginal
from src.path_integral.nested_reference_statistics import nested_variance_diagnostic
from src.path_integral.structural_v2_conditional_reference import (
    block_log_inner_risk,
    last_pair_cache,
    nested_log_means,
)
from src.path_integral.volterra_conditional_payoffs import evaluate_rbergomi_conditional_terminal
from src.path_integral.weighted_tempered_smc import (
    WeightedSMCConfig,
    estimate_weighted_tempered_normalizer,
)


def mixture(dimension: int) -> DefensiveFiniteRankGaussianMixture:
    return DefensiveFiniteRankGaussianMixture((FiniteRankGaussianComponent.natural(dimension),
        FiniteRankGaussianComponent(torch.linspace(-.4, .7, dimension, dtype=torch.float64),
                                    torch.empty((dimension, 0), dtype=torch.float64),
                                    torch.empty(0, dtype=torch.float64))),
        torch.tensor([.2, .8], dtype=torch.float64))


@pytest.mark.parametrize("steps", [1, 4, 32])
@pytest.mark.parametrize("rho", [-.7, 0., .7])
def test_cache_and_full_original_risk_match(steps: int, rho: float) -> None:
    problem = _problem({"spot": 100., "maturity": 1., "hurst": .1,
                        "eta": 3., "xi": .04, "rho": rho}, {"id": "toy", "threshold": 80.}, steps)
    q = mixture(2 * steps)
    gen = torch.Generator().manual_seed(993)
    a = torch.randn((10, 2 * steps - 2), dtype=torch.float64, generator=gen) * .3
    pairs = torch.randn((10, 4, 2), dtype=torch.float64, generator=gen)
    cached = last_pair_cache(problem, a, q)
    full = torch.cat((a[:, None, :].expand(-1, 4, -1), pairs), 2).reshape(40, 2 * steps)
    logg = evaluate_rbergomi_conditional_terminal(problem, full).payoffs.log_left_probability
    reference = (2 * logg - q.log_q_over_p(full)).reshape(10, 4)
    torch.testing.assert_close(cached.log_inner_risk(pairs), reference, rtol=5e-12, atol=5e-12)


def test_exact_marginal_floor_sample_moments_and_zero_dimension() -> None:
    q = mixture(4)
    marginal = ShiftMixtureMarginal.from_full(q, (0, 2))
    x = marginal.sample(40000, path_seed=11, label_seed=12)
    expected = .8 * q.components[1].mean[[0, 2]]
    torch.testing.assert_close(x.mean(0), expected, rtol=0., atol=.025)
    assert float(marginal.log_q_over_p(x).min()) >= math.log(.2) - 1e-13
    oracle = torch.logsumexp(torch.stack([torch.full((len(x),), math.log(.2), dtype=torch.float64),
        x @ q.components[1].mean[[0, 2]] - .5 * q.components[1].mean[[0, 2]].square().sum()
        + math.log(.8)], 1), 1)
    torch.testing.assert_close(marginal.log_q_over_p(x), oracle)
    empty = ShiftMixtureMarginal.from_full(q, ())
    torch.testing.assert_close(empty.log_q_over_p(empty.sample(5, path_seed=1, label_seed=2)),
                               torch.zeros(5, dtype=torch.float64), atol=1e-15, rtol=0.)
    with pytest.raises(ValueError):
        ShiftMixtureMarginal.from_full(q, (0, 0))
    with pytest.raises(ValueError):
        marginal.sample(5, path_seed=1, label_seed=1)


def test_nonidentity_covariance_is_rejected() -> None:
    q = DefensiveFiniteRankGaussianMixture((FiniteRankGaussianComponent.natural(2),
        FiniteRankGaussianComponent(torch.zeros(2, dtype=torch.float64),
                                    torch.eye(2, dtype=torch.float64),
                                    torch.tensor([2., 1.], dtype=torch.float64))),
        torch.tensor([.5, .5], dtype=torch.float64))
    with pytest.raises(ValueError, match="rank-zero"):
        ShiftMixtureMarginal.from_full(q, (0,))


@pytest.mark.parametrize("block", [0, 2, 3])
def test_block_rebuilds_actual_future_not_frozen_variance(block: int) -> None:
    problem = _problem({"spot": 100., "maturity": 1., "hurst": .1,
                        "eta": 1.5, "xi": .04, "rho": -.7}, {"id": "toy", "threshold": 80.}, 4)
    q = mixture(8)
    outer = torch.zeros((3, 6), dtype=torch.float64)
    pair = torch.tensor([[[0., 0.], [1., .5]]], dtype=torch.float64).expand(3, -1, -1)
    values = block_log_inner_risk(problem, outer, pair, q, block)
    full = torch.zeros((6, 8), dtype=torch.float64)
    full[:, 2 * block:2 * block + 2] = pair.reshape(6, 2)
    logg = evaluate_rbergomi_conditional_terminal(problem, full).payoffs.log_left_probability
    torch.testing.assert_close(values, (2 * logg - q.log_q_over_p(full)).reshape(3, 2))


def test_natural_q_nested_analytic_second_moment_not_square_mean() -> None:
    # N=1 makes variance deterministic: exact bivariate Gaussian CDF oracle.
    problem = _problem({"spot": 100., "maturity": 1., "hurst": .1,
                        "eta": 1.5, "xi": .04, "rho": -.7},
                       {"id": "toy", "threshold": 100. * math.exp(-.02)}, 1)
    q = DefensiveFiniteRankGaussianMixture((FiniteRankGaussianComponent.natural(2),),
                                           torch.ones(1, dtype=torch.float64))
    n, inner = 8192, 16
    cache = last_pair_cache(problem, torch.empty((n, 0), dtype=torch.float64), q)
    pair = torch.randn((n, inner, 2), dtype=torch.float64, generator=torch.Generator().manual_seed(819))
    logs = nested_log_means(cache.log_inner_risk(pair), torch.zeros(n, dtype=torch.float64))
    estimates = logs.exp()
    oracle = .25 + math.asin(problem.rho**2) / (2 * math.pi)
    se = float(estimates.std(unbiased=True)) / math.sqrt(n)
    assert abs(float(estimates.mean()) - oracle) < 5 * se
    torch.testing.assert_close(cache.log_mu_mean().exp(), torch.full((n,), .5, dtype=torch.float64),
                               rtol=1e-13, atol=1e-13)
    assert oracle > .25
    with pytest.raises(ValueError):
        nested_log_means(torch.full((2, 1), -torch.inf, dtype=torch.float64),
                         torch.zeros(2, dtype=torch.float64))


def test_nested_noise_decomposition_reports_negative_without_clipping() -> None:
    a = torch.tensor([1., 3., 1., 3.], dtype=torch.float64).log()
    b = torch.tensor([3., 1., 3., 1.], dtype=torch.float64).log()
    result = nested_variance_diagnostic(a, b)
    assert result["negative_outer_estimate"]
    assert result["outer_cv2_unclipped"] < 0


def test_ellipse_gaussian_stationarity_replay_and_variable_calls() -> None:
    k = 3.
    x = torch.randn((12000, 2), dtype=torch.float64, generator=torch.Generator().manual_seed(22)) / math.sqrt(1 + k)
    def potential(z: torch.Tensor) -> torch.Tensor:
        return -.5 * k * z.square().sum(1)
    def move():
        return elliptical_slice_transition(x, potential(x), beta=1., log_potential=potential,
                                            generator=torch.Generator().manual_seed(44))
    a, b = move(), move()
    torch.testing.assert_close(a.particles, b.particles, rtol=0., atol=0.)
    assert float(a.particles.mean().abs()) < .02
    torch.testing.assert_close(a.particles.var(0, unbiased=True), torch.full((2,), .25, dtype=torch.float64),
                               rtol=.06, atol=.002)
    assert a.potential_evaluations >= len(x)
    assert a.maximum_attempts > 1


def test_ellipse_cap_raises_not_truncated_transition() -> None:
    with pytest.raises(TimeoutError, match="no truncated kernel"):
        elliptical_slice_transition(torch.zeros((4, 2), dtype=torch.float64),
            torch.zeros(4, dtype=torch.float64), beta=1.,
            log_potential=lambda z: torch.full((len(z),), -1e100, dtype=torch.float64),
            generator=torch.Generator().manual_seed(81), maximum_attempts=1)


def test_ellipse_smc_known_normalizer_and_actual_work() -> None:
    calls = 0
    def potential(z: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += len(z)
        return -.5 * 2. * z.square().sum(1)
    result = estimate_weighted_tempered_normalizer(potential, dimension=2,
        config=WeightedSMCConfig(128, tuple(i / 8 for i in range(9)), 1, .5, 24, 49,
                                 mutation_kernel="elliptical_slice", resample_every=2))
    assert calls == result.potential_evaluations
    assert abs(result.mean - 1 / 3) < 5 * result.standard_error
    with pytest.raises(ValueError):
        WeightedSMCConfig(8, (0., 1.), 1, .5, 1, 3, mutation_kernel="elliptical_slice", independence_every=1)
