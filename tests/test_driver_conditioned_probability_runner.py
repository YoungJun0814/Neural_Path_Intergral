import pytest

from experiments.post_audit_driver_conditioned_probability import (
    WORK,
    check_unit,
    identity,
    jobs,
    seed,
    summarize,
)


def test_driver_probability_pilot_fixed_new_seeds_and_cost():
    grid = jobs()
    assert len(grid) == len({identity(j) for j in grid}) == len({seed(j) for j in grid}) == 32
    assert len(grid)*WORK == 8159232
    with pytest.raises(ValueError):
        summarize([])
    j = grid[0]
    unit = {**j, "seed": seed(j), "actual_work": WORK, "backward_evaluations": WORK,
            "dimension": 64, "guide_independent_kernel": True, "log_normalizer": -10., "log_components": [-10.]}
    check_unit(j, unit)
    with pytest.raises(ValueError):
        check_unit(j, {**unit, "dimension": 31})
