import pytest

from experiments.post_audit_p2_paired_fine_extension import (
    JOINT_WORK,
    identity,
    jobs,
    seed,
    summarize,
)
from experiments.post_audit_p2_paired_gradient import jobs as old_jobs
from experiments.post_audit_p2_paired_gradient import seed as old_seed


def test_finer_fixed_grid_seed_disjointness_and_physical_accounting():
    grid = jobs()
    assert len(grid) == len({identity(j) for j in grid}) == len({seed(j) for j in grid}) == 32
    assert {seed(j) for j in grid}.isdisjoint({old_seed(j) for j in old_jobs()})
    assert 2*JOINT_WORK*len(grid) == 2039808
    assert JOINT_WORK*len(grid) == 1019904
    assert sum(JOINT_WORK*(j["fine_steps"]+j["fine_steps"]//2) for j in grid) == 4699717632


def test_finer_summary_requires_complete_grid_and_paired_wholes():
    records = [{**j, "log_component_normalizers": [-10.-j["rep"]*.01, -10.-j["rep"]*.01],
                "stages": [{"surviving_initial_ancestors": 20}]} for j in jobs()]
    with pytest.raises(ValueError, match="complete"):
        summarize(records[:-1])
    out = summarize(records)
    assert len(out) == 4
    assert all(r["paired_difference"]["relative_difference"] == 0. for r in out)
    assert all(r["paired_difference"]["paired_relative_se"] == 0. for r in out)
    assert all(r["paired_difference"]["whole_count"] == 8 for r in out)
