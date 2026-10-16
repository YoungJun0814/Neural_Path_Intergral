from experiments.post_audit_guide_free_block_recovery import CALLS, identity, jobs, seed


def test_fixed_guide_free_grid_and_actual_work():
    rows = [{"cell": {"id": c}, "parent_training_rep": rep}
            for c in ("canonical", "high-eta") for rep in range(5)]
    grid = jobs(rows)
    units = [identity(j) for j in grid]
    assert len(grid) == len(set(units)) == 64
    assert len({seed(i) for i in units}) == 64
    assert CALLS == 1024*(1+62*8)
    assert len(grid)*CALLS == 32571392
    assert {j["rep"] for j in grid} == set(range(8))
