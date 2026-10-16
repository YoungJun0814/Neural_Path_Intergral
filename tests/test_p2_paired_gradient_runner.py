from experiments.post_audit_p2_paired_gradient import JOINT_WORK, identity, jobs, seed


def test_paired_mesh_fixed_grid_augmented_dimension_and_physical_work():
    grid = jobs()
    assert len(grid) == len({identity(j) for j in grid}) == len({seed(j) for j in grid}) == 32
    assert 2*JOINT_WORK*len(grid) == 2039808
    assert JOINT_WORK*len(grid) == 1019904
    assert sum(JOINT_WORK*(j["fine_steps"]+j["fine_steps"]//2) for j in grid) == 1174929408
