import pytest
import torch

from experiments.post_audit_fft_static_fine_reference import (
    DRAW_COUNT,
    WORK,
    identity,
    jobs,
    merge_retaining_maximum,
    seed,
    summarize,
)
from src.path_integral.structural_v2_reference import log_moments, merge_moments


def test_static_fine_fixed_iid_blocks_joint_density_seeds_and_physical_cost():
    grid = jobs()
    assert len(grid) == len({identity(j) for j in grid}) == 16
    seeds = [seed(j, k, role) for j in grid for k in range(64) for role in ("path", "label", "augmentation")]
    assert len(seeds) == len(set(seeds)) == 3072
    assert 16*WORK == 1048576 and 16*DRAW_COUNT == 524288
    assert 16*DRAW_COUNT*6144 == 3221225472
    with pytest.raises(ValueError):
        summarize([])


def test_two_level_moments_keep_maximum_and_recover_legacy_metadata():
    old = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        values = torch.linspace(-20, -2, 256, dtype=torch.float64)
        chunks = [log_moments(v) for v in values.split(32)]
        original_blocks = [merge_moments(chunks[:4]), merge_moments(chunks[4:])]
        restored = merge_retaining_maximum(original_blocks)
        direct = log_moments(values)
        for key in ("log_mean", "log_sum", "log_sum_squares", "relative_se", "maximum_fraction", "log_maximum"):
            assert restored[key] == pytest.approx(direct[key], rel=2e-12, abs=2e-12)
        assert restored["count"] == 256 and "log_maximum" not in original_blocks[0]
    finally:
        torch.set_num_threads(old)


def test_complete_static_grid_legacy_recursive_aggregation_does_not_fail():
    old = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        chunk = log_moments(torch.linspace(-20, -19, 512, dtype=torch.float64))
        block = merge_moments([chunk]*64)
        records = [{**j, "moments": [dict(block), dict(block)]} for j in jobs()]
        rows = summarize(records)
        assert len(rows) == 2
        assert all(r["components"][0]["summary"]["count"] == 262144 for r in rows)
        assert all(r["paired_difference"]["relative_difference"] == 0. for r in rows)
        assert all(r["paired_difference"]["paired_relative_se"] == 0. for r in rows)
    finally:
        torch.set_num_threads(old)
