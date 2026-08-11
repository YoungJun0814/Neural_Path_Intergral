import math

import torch

from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.cameron_martin_basis import build_blp_cameron_martin_basis
from src.path_integral.finite_grid_small_noise import RBergomiFiniteGridRateAction
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.volterra_action import RBergomiConditionalAction


def _problem() -> RBergomiBaselineProblem:
    return RBergomiBaselineProblem(
        task_id="v16-fixed-grid-rate",
        task=TerminalThresholdTask(level=70.0),
        spot=100.0,
        maturity=1.0,
        steps=6,
        hurst=0.12,
        eta=1.5,
        xi=0.04,
        rho=-0.7,
    )


def test_full_dct_basis_marks_the_full_fixed_grid_rate_problem() -> None:
    problem = _problem()
    full = build_blp_cameron_martin_basis(steps=problem.steps)
    reduced = build_blp_cameron_martin_basis(steps=problem.steps, modes_per_driver=2)
    assert RBergomiFiniteGridRateAction(problem, full).is_full_grid
    assert not RBergomiFiniteGridRateAction(problem, reduced).is_full_grid


def test_finite_noise_action_converges_to_contracted_fixed_grid_rate() -> None:
    problem = _problem()
    basis = build_blp_cameron_martin_basis(steps=problem.steps, modes_per_driver=2)
    rate = RBergomiFiniteGridRateAction(problem, basis)
    coefficient = torch.tensor([0.35, -0.1, -0.2, 0.15], dtype=torch.float64)
    target = float(rate(coefficient))
    errors = []
    for epsilon in (0.04, 0.01, 0.0004):
        finite = RBergomiConditionalAction(
            problem=problem,
            basis=basis,
            epsilon=epsilon,
        )
        errors.append(abs(float(finite(coefficient)) - target))
    assert errors[2] < errors[1] < errors[0]
    assert errors[-1] < 0.004


def test_contracted_independent_driver_cost_has_correct_active_set() -> None:
    problem = _problem()
    basis = build_blp_cameron_martin_basis(steps=problem.steps)
    rate = RBergomiFiniteGridRateAction(problem, basis)
    zero = rate.evaluate(torch.zeros(basis.rank, dtype=torch.float64))
    expected = math.log(problem.task.level / problem.spot) ** 2 / (
        2.0 * (1.0 - problem.rho**2) * problem.xi * problem.maturity
    )
    # At zero control the zero-noise Wick correction vanishes and V is xi.
    assert abs(float(zero.conditional_cost) - expected) < 2e-13
    assert abs(float(zero.integrated_variance) - problem.xi * problem.maturity) < 2e-14


def test_rate_value_is_finite_nonnegative_and_differentiable() -> None:
    problem = _problem()
    basis = build_blp_cameron_martin_basis(steps=problem.steps, modes_per_driver=2)
    rate = RBergomiFiniteGridRateAction(problem, basis)
    point = torch.tensor([0.1, -0.2, 0.3, -0.1], dtype=torch.float64, requires_grad=True)
    value = rate(point)
    (gradient,) = torch.autograd.grad(value, point)
    assert float(value.detach()) >= 0.0
    assert torch.isfinite(value)
    assert torch.isfinite(gradient).all()
