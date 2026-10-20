import pytest
import torch

from experiments.post_audit_bridge_mutation_allocation import (
    ARMS,
    identity,
    jobs,
    joint_work,
    seed,
    summarize,
)


def test_allocation_fixed_grid_and_independent_seeds_cost():
    grid = jobs()
    assert len(grid) == len({identity(j) for j in grid}) == len({seed(j) for j in grid}) == 32
    assert sum(2*joint_work(j["arm"]) for j in grid) == 1028096
    assert sum(6144*joint_work(j["arm"]) for j in grid) == 3158310912
    assert [joint_work(a) for a in ARMS] == [15936, 16192]
    with pytest.raises(ValueError):
        summarize([])


def test_allocation_forecasts_do_not_open_gates_and_use_wholes():
    old = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        records = [{**j, "log_component_normalizers": [-18.+.01*j["rep"], -18.+.01*j["rep"]],
                    "stages": [{"surviving_initial_ancestors": 12, "mutation_acceptance": .2, "ess_before_resampling": 32.}]*3}
                   for j in jobs()]
        out = summarize(records)
        assert len(out["rows"]) == 4
        assert all(x["components"][0]["summary"]["count"] == 8 for x in out["rows"])
        assert not any(out[k] for k in ("p1_qualified", "p2_qualified", "large_training_authorized"))
    finally:
        torch.set_num_threads(old)
