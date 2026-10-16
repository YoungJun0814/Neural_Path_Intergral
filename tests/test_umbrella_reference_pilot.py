import torch

from experiments.post_audit_umbrella_reference_pilot import (
    expected_work,
    identity,
    jobs,
    seed,
    weights_from_anchor,
)
from src.path_integral.baselines.weighted_conditional_ce import proposal_parameters
from src.path_integral.finite_rank_gaussian_transport import (
    DefensiveFiniteRankGaussianMixture,
    FiniteRankGaussianComponent,
)


def test_frozen_family_grid_and_work_accounting():
    rows = [{"cell": {"id": c}, "parent_training_rep": rep}
            for c in ("canonical", "high-eta") for rep in range(5)]
    grid = jobs(rows)
    names = [identity(j) for j in grid]
    assert len(grid) == len(set(names)) == 48
    assert len({seed(i) for i in names}) == 48
    assert 128+sum(expected_work(j) for j in grid) == 20365440


def test_inverse_anchor_weights_include_scaled_risk_and_are_normalized():
    q = DefensiveFiniteRankGaussianMixture((FiniteRankGaussianComponent.natural(2),),
                                           torch.ones(1, dtype=torch.float64))
    rows = [{"cell": {"id": "test"}, "parent_training_rep": rep,
             "q_parameters": proposal_parameters(q)} for rep in range(5)]
    anchor = {"summaries": [{"cell": "test", "method": "guide-iid", "target": target,
                             "parent_training_rep": rep, "summary": {"log_mean": value}}
                            for target, rep, value in [("mu", 0, -5.)]+[("risk", rep, -7.-rep) for rep in range(5)]]}
    w = torch.tensor(weights_from_anchor(anchor, rows)["test"], dtype=torch.float64)
    expected = torch.tensor([5.]+[7.+rep for rep in range(5)], dtype=torch.float64)
    torch.testing.assert_close(w, expected-torch.logsumexp(expected, 0), rtol=0, atol=1e-12)
    assert abs(float(torch.logsumexp(w, 0))) < 1e-12
