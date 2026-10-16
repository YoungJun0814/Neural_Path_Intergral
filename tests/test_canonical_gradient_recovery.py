import math

import pytest
import torch

from experiments.post_audit_canonical_gradient_recovery import (
    ARMS,
    component_targets,
    expected_work,
    family_weights,
    identity,
    jobs,
    seed,
    summarize,
)


def test_fresh_canonical_grid_and_separate_forward_backward_cost():
    grid = jobs()
    assert len(grid) == len({identity(j) for j in grid}) == len({seed(j) for j in grid}) == 32
    assert 128+sum(expected_work(j)[0] for j in grid) == 12222592
    assert sum(expected_work(j)[1] for j in grid) == 4079616
    assert [len(component_targets(a)) for a in ARMS] == [6, 5, 5, 1]


def test_target_separation_uses_renormalized_frozen_risk_weights():
    frozen = torch.tensor([.1, .2, .1, .1, .3, .2], dtype=torch.float64).log().tolist()
    w = family_weights("risks-pcnl", frozen)
    expected = torch.tensor(frozen[1:], dtype=torch.float64)-math.log(.9)
    torch.testing.assert_close(w, expected, rtol=0, atol=1e-12)
    assert family_weights("probability-pcnl", frozen).tolist() == [0.]
    with pytest.raises(ValueError):
        family_weights("unknown", frozen)


def test_complete_wholes_required_and_outputs_not_counted_as_replicates():
    weights = torch.tensor([.1, .2, .1, .1, .3, .2], dtype=torch.float64).log().tolist()
    rows = []
    for j in jobs():
        w = family_weights(j["arm"], weights)
        vector = -10.+.001*j["rep"]+torch.arange(len(w), dtype=torch.float64)*.05
        z = float(torch.logsumexp(vector+w, 0))
        rows.append({**j, "log_normalizer": z, "log_scaled_component_normalizers": vector.tolist(),
                     "log_estimand_values": vector.tolist(), "wall": 1.})
    summaries = summarize(rows)
    assert len(summaries) == 4
    for s in summaries:
        assert s["allocation_feasible"]
        assert all(c["summary"]["count"] == 8 for c in s["components"])
        assert s["forecast_backward_evaluations"] == (32*254976 if s["arm"].endswith("pcnl") else 0)
    with pytest.raises(ValueError):
        summarize(rows[:-1])
