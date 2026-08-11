import math

import numpy as np
import pytest
import torch
from scipy.special import log_ndtr

from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.rbergomi_local_volterra_transport import (
    evaluate_conditional_terminal_local,
)
from src.path_integral.volterra_conditional_payoffs import (
    evaluate_lognormal_terminal_payoffs,
    evaluate_rbergomi_conditional_terminal,
)


def _problem(*, rho: float = -0.7, eta: float = 1.5) -> RBergomiBaselineProblem:
    return RBergomiBaselineProblem(
        task_id="v15-oracle",
        task=TerminalThresholdTask(level=70.0),
        spot=100.0,
        maturity=1.0,
        steps=16,
        hurst=0.12,
        eta=eta,
        xi=0.04,
        rho=rho,
    )


def test_epsilon_one_is_pathwise_identical_to_v14_terminal_conditioning() -> None:
    problem = _problem()
    local = torch.randn(
        64,
        problem.local_dimension,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(1729),
    )
    old = evaluate_conditional_terminal_local(problem, local)
    new = evaluate_rbergomi_conditional_terminal(problem, local, epsilon=1.0)
    assert torch.max(
        torch.abs(old.standardized_threshold - new.payoffs.standardized_left_threshold)
    ) < 2e-12
    assert torch.max(torch.abs(old.log_conditional_value - new.payoffs.log_left_probability)) < 2e-12
    assert torch.max(torch.abs(old.conditional_value - new.payoffs.left_probability)) < 2e-12


def test_lognormal_digitals_match_independent_scipy_oracle_in_extreme_tails() -> None:
    mean = torch.tensor([-2.0, -0.2, 0.0, 0.7, 3.0], dtype=torch.float64)
    variance = torch.tensor([0.01, 0.2, 1.0, 0.03, 0.04], dtype=torch.float64)
    strike = 1.0
    batch = evaluate_lognormal_terminal_payoffs(mean, variance, strike=strike)
    threshold = (math.log(strike) - mean.numpy()) / np.sqrt(variance.numpy())
    np.testing.assert_allclose(
        batch.log_left_probability.numpy(), log_ndtr(threshold), rtol=1e-13, atol=1e-13
    )
    np.testing.assert_allclose(
        batch.log_right_probability.numpy(), log_ndtr(-threshold), rtol=1e-13, atol=1e-13
    )


def test_lognormal_put_call_parity_and_complementarity() -> None:
    mean = torch.linspace(-3.0, 2.0, 41, dtype=torch.float64)
    variance = torch.linspace(0.01, 1.0, 41, dtype=torch.float64)
    strike = 1.3
    batch = evaluate_lognormal_terminal_payoffs(mean, variance, strike=strike)
    assert torch.max(torch.abs(batch.left_probability + batch.right_probability - 1.0)) < 5e-15
    parity = batch.call_value - batch.put_value
    assert torch.max(torch.abs(parity - (batch.conditional_forward - strike))) < 2e-13
    assert bool((batch.put_value >= 0.0).all())
    assert bool((batch.call_value >= 0.0).all())


def test_rho_zero_nearly_constant_volatility_recovers_black_scholes_law() -> None:
    problem = _problem(rho=0.0, eta=1e-12)
    local = torch.randn(
        8,
        problem.local_dimension,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(99),
    )
    epsilon = 0.35
    batch = evaluate_rbergomi_conditional_terminal(problem, local, epsilon=epsilon)
    expected_mean = math.log(problem.spot) - 0.5 * epsilon * problem.xi * problem.maturity
    expected_variance = epsilon * problem.xi * problem.maturity
    assert torch.max(torch.abs(batch.terminal_log_mean - expected_mean)) < 2e-12
    assert torch.max(torch.abs(batch.terminal_log_variance - expected_variance)) < 2e-12


def test_small_noise_family_converges_to_deterministic_left_tail() -> None:
    problem = _problem(rho=-0.4)
    local = torch.zeros(2, problem.local_dimension, dtype=torch.float64)
    moderate = evaluate_rbergomi_conditional_terminal(problem, local, epsilon=0.1)
    tiny = evaluate_rbergomi_conditional_terminal(problem, local, epsilon=1e-4)
    assert bool((tiny.payoffs.log_left_probability < moderate.payoffs.log_left_probability).all())
    assert bool((tiny.payoffs.left_probability < moderate.payoffs.left_probability).all())


@pytest.mark.parametrize("epsilon", [0.0, -0.1, 1.01, math.inf])
def test_invalid_small_noise_scale_is_rejected(epsilon: float) -> None:
    problem = _problem()
    local = torch.zeros(1, problem.local_dimension, dtype=torch.float64)
    with pytest.raises(ValueError, match="epsilon"):
        evaluate_rbergomi_conditional_terminal(problem, local, epsilon=epsilon)

