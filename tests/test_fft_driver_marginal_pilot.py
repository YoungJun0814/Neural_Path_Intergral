import pytest

from experiments.post_audit_fft_driver_marginal_pilot import (
    DRAW_COUNT,
    identity,
    jobs,
    seed,
    summarize,
)


def test_fixed_conditional_pilot_budget_and_separate_roles():
    grid = jobs()
    assert len(grid) == len({identity(j) for j in grid}) == 16
    seeds = [seed(j, k, role) for j in grid for k in range(64) for role in ("path", "label")]
    assert len(seeds) == len(set(seeds)) == 2048
    assert len(grid)*DRAW_COUNT == 16384
    with pytest.raises(ValueError):
        summarize([])
