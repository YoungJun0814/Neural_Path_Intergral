import pytest
import torch

from src.path_integral.marginal_risk_contract import (
    finite_risk_decomposition,
    marginal_risk_known_answer,
)


def test_exact_conditional_risk_improvement_cannot_remove_marginal_risk():
    result = marginal_risk_known_answer()
    a, b, c = result["plain"], result["ideal_conditional_only"], result["ideal_joint"]
    assert result["conditional_only_leaves_positive_marginal_variance"]
    assert result["ideal_conditional_variance_zero"] and result["ideal_joint_variance_zero"]
    assert a["marginal_variance"] == pytest.approx(b["marginal_variance"])
    assert a["M2_of_fixed_event_q"] == pytest.approx(c["M2_of_fixed_event_q"])
    assert a["total_risk_estimator_variance"] == pytest.approx(a["marginal_variance"]+a["conditional_variance"])
    assert b["total_risk_estimator_variance"] > 0


@pytest.mark.parametrize("invalid", ["proposal", "conditional", "payoff", "probability"])
def test_risk_oracle_rejects_unnormalized_or_invalid_density(invalid):
    px = torch.tensor([.5, .5], dtype=torch.float64)
    pu = px.clone()
    g = torch.full((2, 2), .2, dtype=torch.float64)
    dq = torch.ones(2, dtype=torch.float64)
    dg = dq.clone()
    dh = torch.ones_like(g)
    if invalid == "proposal":
        dq[0] = 2
    elif invalid == "conditional":
        dh[0, 0] = 2
    elif invalid == "payoff":
        g[0, 0] = 1.1
    else:
        px[0] = 0
    with pytest.raises(ValueError):
        finite_risk_decomposition(px, pu, g, dq, dg, dh)


def test_positive_finite_densities_do_not_hide_second_moment_overflow():
    px = torch.tensor([.5, .5], dtype=torch.float64)
    dh = torch.tensor([[1e-320, 2.], [1., 1.]], dtype=torch.float64)
    with pytest.raises(FloatingPointError, match="range unresolved"):
        finite_risk_decomposition(px, px, torch.ones_like(dh), torch.ones_like(px), torch.ones_like(px), dh)
