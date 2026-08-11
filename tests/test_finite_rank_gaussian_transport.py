import math

import torch

from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.cameron_martin_basis import build_blp_cameron_martin_basis
from src.path_integral.cameron_martin_modes import (
    ActionSolverConfig,
    ModeSearchConfig,
    find_rbergomi_conditional_modes,
)
from src.path_integral.finite_rank_gaussian_transport import (
    CurvatureTransportConfig,
    DefensiveFiniteRankGaussianMixture,
    FiniteRankGaussianComponent,
    build_curvature_transport,
)
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.volterra_action import RBergomiConditionalAction


def _component() -> FiniteRankGaussianComponent:
    angle = 0.37
    direction = torch.tensor(
        [[math.cos(angle)], [math.sin(angle)], [0.0]],
        dtype=torch.float64,
    )
    return FiniteRankGaussianComponent(
        mean=torch.tensor([0.3, -0.2, 0.1], dtype=torch.float64),
        directions=direction,
        variance_eigenvalues=torch.tensor([2.4], dtype=torch.float64),
    )


def test_component_density_matches_dense_gaussian_oracle() -> None:
    component = _component()
    samples = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, -0.5, 0.2], [-2.0, 0.3, 1.1]],
        dtype=torch.float64,
    )
    covariance = torch.eye(3, dtype=torch.float64) + component.directions @ (
        torch.diag(component.variance_eigenvalues - 1.0) @ component.directions.T
    )
    proposal = torch.distributions.MultivariateNormal(component.mean, covariance_matrix=covariance)
    reference = torch.distributions.MultivariateNormal(
        torch.zeros(3, dtype=torch.float64),
        torch.eye(3, dtype=torch.float64),
    )
    oracle = proposal.log_prob(samples) - reference.log_prob(samples)
    assert torch.max(torch.abs(component.log_q_over_p(samples) - oracle)) < 2e-12


def test_component_sampler_has_prescribed_mean_and_covariance() -> None:
    component = _component()
    generator = torch.Generator().manual_seed(8128)
    standard = torch.randn((120_000, 3), dtype=torch.float64, generator=generator)
    samples = component.transform_standard_normal(standard)
    empirical_mean = torch.mean(samples, dim=0)
    centered = samples - empirical_mean
    empirical_covariance = centered.T @ centered / samples.shape[0]
    target_covariance = torch.eye(3, dtype=torch.float64) + component.directions @ (
        torch.diag(component.variance_eigenvalues - 1.0) @ component.directions.T
    )
    assert torch.max(torch.abs(empirical_mean - component.mean)) < 0.012
    assert torch.max(torch.abs(empirical_covariance - target_covariance)) < 0.025


def test_defensive_mixture_exact_likelihood_and_normalization() -> None:
    mixture = DefensiveFiniteRankGaussianMixture(
        components=(FiniteRankGaussianComponent.natural(3), _component()),
        weights=torch.tensor([0.2, 0.8], dtype=torch.float64),
    )
    drawn = mixture.sample(160_000, path_seed=931, label_seed=932)
    likelihood = torch.exp(drawn.log_p_over_q)
    assert abs(float(torch.mean(likelihood)) - 1.0) < 0.008
    assert float(torch.max(likelihood)) <= 5.0 + 2e-12
    assert torch.max(
        torch.abs(mixture.log_q_over_p(drawn.samples) - drawn.log_q_over_p)
    ) < 2e-13
    frequencies = torch.bincount(drawn.labels, minlength=2).to(torch.float64) / drawn.labels.numel()
    assert torch.max(torch.abs(frequencies - mixture.weights)) < 0.004


def test_curvature_builder_creates_exact_positive_defensive_transport() -> None:
    problem = RBergomiBaselineProblem(
        task_id="v15-transport",
        task=TerminalThresholdTask(level=60.0),
        spot=100.0,
        maturity=1.0,
        steps=8,
        hurst=0.12,
        eta=1.5,
        xi=0.04,
        rho=-0.7,
    )
    basis = build_blp_cameron_martin_basis(steps=problem.steps, modes_per_driver=2)
    action = RBergomiConditionalAction(problem=problem, basis=basis, epsilon=1.0)
    modes = find_rbergomi_conditional_modes(
        action,
        config=ModeSearchConfig(
            methods=("lbfgs", "trust-ncg"),
            random_starts=1,
            random_seed=12,
            start_scale=1.0,
            solver=ActionSolverConfig(maximum_iterations=100, gradient_tolerance=1e-6),
        ),
    )
    transport = build_curvature_transport(
        action,
        modes,
        config=CurvatureTransportConfig(defensive_mass=0.15),
    )
    assert math.isclose(transport.defensive_mass, 0.15, abs_tol=1e-14)
    assert transport.components[0].is_natural()
    assert all(component.rank == basis.rank for component in transport.components[1:])
    sample = transport.sample(512, path_seed=881, label_seed=882)
    assert torch.isfinite(sample.log_p_over_q).all()
    assert float(torch.max(torch.exp(sample.log_p_over_q))) <= 1.0 / 0.15 + 2e-12
