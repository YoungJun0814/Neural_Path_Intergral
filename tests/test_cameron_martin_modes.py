import torch

from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.cameron_martin_basis import build_blp_cameron_martin_basis
from src.path_integral.cameron_martin_modes import (
    ActionSolverConfig,
    ModeSearchConfig,
    find_action_modes,
    find_rbergomi_conditional_modes,
    optimize_action,
)
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.volterra_action import (
    RBergomiConditionalAction,
    evaluate_action_derivatives,
)


def test_both_solvers_recover_quadratic_minimum() -> None:
    target = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)

    def action(point: torch.Tensor) -> torch.Tensor:
        difference = point - target
        return 0.5 * torch.dot(difference, difference)

    initial = torch.tensor([-3.0, 4.0, 2.0], dtype=torch.float64)
    for method in ("lbfgs", "trust-ncg"):
        result = optimize_action(
            action,
            initial,
            config=ActionSolverConfig(method=method, gradient_tolerance=1e-9),
        )
        assert result.success
        assert torch.max(torch.abs(result.coefficients - target)) < 1e-7
        assert result.gradient_norm < 1e-7


def test_multistart_search_recovers_both_modes_of_symmetric_action() -> None:
    def action(point: torch.Tensor) -> torch.Tensor:
        return (point[0] ** 2 - 1.0) ** 2 + 0.25 * point[1] ** 2

    config = ModeSearchConfig(
        methods=("lbfgs", "trust-ncg"),
        random_starts=12,
        random_seed=42,
        start_scale=2.5,
        merge_distance=1e-4,
        action_merge_tolerance=1e-7,
        solver=ActionSolverConfig(maximum_iterations=200, gradient_tolerance=1e-8),
    )
    result = find_action_modes(action, 2, config=config)
    minima = sorted(float(mode.coefficients[0]) for mode in result.modes if mode.action_value < 1e-10)
    assert len(minima) == 2
    assert abs(minima[0] + 1.0) < 1e-6
    assert abs(minima[1] - 1.0) < 1e-6
    assert all(len(mode.methods) == 2 for mode in result.modes if mode.action_value < 1e-10)


def test_supplied_only_search_does_not_charge_hidden_cold_or_random_starts() -> None:
    target = torch.tensor([1.0, -2.0], dtype=torch.float64)

    def action(point: torch.Tensor) -> torch.Tensor:
        difference = point - target
        return 0.5 * torch.dot(difference, difference)

    result = find_action_modes(
        action,
        2,
        config=ModeSearchConfig(
            methods=("lbfgs",),
            random_starts=0,
            include_zero_start=False,
            solver=ActionSolverConfig(gradient_tolerance=1e-9),
        ),
        supplied_starts=(torch.tensor([0.9, -1.8], dtype=torch.float64),),
    )
    assert len(result.attempts) == 1
    assert result.modes
    assert torch.max(torch.abs(result.modes[0].coefficients - target)) < 1e-7


def _problem() -> RBergomiBaselineProblem:
    return RBergomiBaselineProblem(
        task_id="v15-mode",
        task=TerminalThresholdTask(level=60.0),
        spot=100.0,
        maturity=1.0,
        steps=8,
        hurst=0.12,
        eta=1.5,
        xi=0.04,
        rho=-0.7,
    )


def test_rbergomi_action_gradient_matches_centered_finite_difference() -> None:
    problem = _problem()
    basis = build_blp_cameron_martin_basis(steps=problem.steps, modes_per_driver=2)
    action = RBergomiConditionalAction(problem=problem, basis=basis, epsilon=0.4)
    point = torch.tensor([0.2, -0.1, 0.05, 0.3], dtype=torch.float64)
    audited = evaluate_action_derivatives(action, point)
    step = 2e-5
    finite = []
    for index in range(point.numel()):
        direction = torch.zeros_like(point)
        direction[index] = step
        finite.append(float((action(point + direction) - action(point - direction)) / (2.0 * step)))
    finite_tensor = torch.tensor(finite, dtype=torch.float64)
    assert torch.max(torch.abs(audited.gradient - finite_tensor)) < 2e-6


def test_rbergomi_mode_reduces_action_and_has_finite_curvature() -> None:
    problem = _problem()
    basis = build_blp_cameron_martin_basis(steps=problem.steps, modes_per_driver=3)
    action = RBergomiConditionalAction(problem=problem, basis=basis, epsilon=1.0)
    initial_value = float(action(torch.zeros(basis.rank, dtype=torch.float64)))
    result = find_rbergomi_conditional_modes(
        action,
        config=ModeSearchConfig(
            methods=("lbfgs", "trust-ncg"),
            random_starts=2,
            random_seed=9,
            start_scale=1.0,
            solver=ActionSolverConfig(maximum_iterations=100, gradient_tolerance=1e-6),
        ),
    )
    assert result.modes
    best = result.modes[0]
    assert best.action_value < initial_value
    assert best.gradient_norm < 2e-5
    assert torch.isfinite(best.proposal_mean).all()
    assert best.hessian_eigenvalues is not None
    assert torch.isfinite(best.hessian_eigenvalues).all()
