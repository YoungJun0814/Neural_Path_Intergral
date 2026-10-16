from experiments.post_audit_canonical_gradient_recovery import jobs as previous_jobs
from experiments.post_audit_canonical_gradient_recovery import seed as previous_seed
from experiments.post_audit_canonical_gradient_resource_sweep import (
    DESIGNS,
    expected_work,
    identity,
    jobs,
    seed,
    selected_design,
)


def test_resource_design_grid_cost_and_disjoint_previous_streams():
    grid = jobs()
    streams = {seed(j) for j in grid}
    assert len(grid) == len({identity(j) for j in grid}) == len(streams) == 32
    assert not streams & {previous_seed(j) for j in previous_jobs()}
    assert sum(expected_work(j) for j in grid) == 14327808
    assert [expected_work({"design": d}) for d in DESIGNS] == [508928, 509952, 517120, 254976]


def test_selection_never_uses_infeasible_low_cost_or_opens_global_gates():
    rows = [{"design": "failed", "allocation_feasible": False, "selection_cost_proxy": 0.},
            {"design": "a", "allocation_feasible": True, "selection_cost_proxy": 2.},
            {"design": "b", "allocation_feasible": True, "selection_cost_proxy": 1.}]
    assert selected_design(rows) == "b"
    assert selected_design(rows[:1]) is None
