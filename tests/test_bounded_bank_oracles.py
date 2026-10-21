import math

import numpy as np
import pytest
import torch
from scipy.optimize import minimize_scalar

from experiments.post_audit_bounded_bank_oracles import Budget, cases, numpy_logs
from src.path_integral.defensive_mixture_allocation import (
    allocation_weights,
    log_second_moment_objective,
)
from src.path_integral.physical_dictionary_learning import (
    event_kl_objective,
    fit_dictionary_weights,
)
from tests.test_physical_dictionary_learning import bank


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_fixed_cases_numpy_and_torch_density_payoff_agree():
    z = np.array([[0., 0.], [3., 1.], [-5., 3.]], dtype=np.float64)
    assert [name for name, _, _ in cases()] == ["A", "B", "C", "D"]
    for _, toy, spec in cases():
        g, d = numpy_logs(z, toy, spec)
        torch.testing.assert_close(torch.tensor(g), toy.log_value(torch.tensor(z)))
        expected = torch.logsumexp(torch.tensor(z) @ spec.means.T-.5*spec.means.square().sum(1)+spec.weights.log(), 1)
        torch.testing.assert_close(torch.tensor(d), expected)
        assert 0 < math.exp(toy.log_endpoint(1)) <= 1


@pytest.mark.parametrize("objective", ["event-kl", "risk-m2"])
def test_independent_scalar_solver_verifies_finite_bank_fw_bound(objective):
    c, g, d = bank()
    fit = fit_dictionary_weights(c, g, d, objective=objective, maximum_iterations=12)

    def loss(v):
        w = allocation_weights(torch.tensor([v, 1-v], dtype=torch.float64))
        return event_kl_objective(c, g, d, w) if objective == "event-kl" else log_second_moment_objective(c, g, d, w)

    reference = minimize_scalar(loss, bounds=(0., 1.), method="bounded", options={"xatol": 1e-12})
    assert reference.success
    if objective == "event-kl":
        assert -1e-9 <= fit.final_objective-reference.fun <= fit.relative_empirical_gap+1e-9
    else:
        assert math.exp(reference.fun-fit.final_objective) >= 1-fit.relative_empirical_gap-1e-9


def test_weighted_training_proportional_rows_reduce_to_equal_row_objectives():
    c, g, d = bank()
    w = allocation_weights(torch.tensor([.35, .65], dtype=torch.float64))
    a = torch.full_like(g, 1/len(g))
    event = a*(g-d).exp()
    ld = torch.logsumexp(c+w.log(), 1)
    assert -float((event/event.sum()) @ ld) == pytest.approx(event_kl_objective(c, g, d, w))
    assert float((a*(2*g-d-ld).exp()).sum().log()) == pytest.approx(log_second_moment_objective(c, g, d, w))


def test_synthetic_budget_never_silently_extends():
    budget = Budget(maximum=10)
    budget.charge(10)
    with pytest.raises(TimeoutError):
        budget.charge(1)
    assert budget.count == 10
