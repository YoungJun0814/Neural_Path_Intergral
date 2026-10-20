import pytest

from experiments.post_audit_physical_window_recovery import (
    DRAW_COUNT,
    WORK,
    identity,
    jobs,
    validate_record,
)


def test_recovery_is_same_final_seed_and_preserves_conservative_discard_cost():
    grid = jobs()
    assert len(grid) == 32
    assert identity(grid[-1]) == "high-eta200-k1-mixed-window-7"
    assert 31*WORK+WORK == 1048576
    assert WORK == 32768 and DRAW_COUNT == 16384
    assert 4*WORK == 131072
    with pytest.raises((KeyError, ValueError)):
        validate_record(grid[-1], {}, {})
