import pytest
import torch

from experiments.post_audit_physical_window_reference import (
    ANCHOR,
    DIRECTORY,
    DRAW_COUNT,
    RESOURCE_AUDIT,
    WORK,
    identity,
    jobs,
    seed,
    summarize,
)
from src.path_integral.structural_v2_reference import log_moments


def test_matched_window_fixed_grid_new_seeds_and_cost():
    grid = jobs()
    assert len(grid) == len({identity(j) for j in grid}) == 32
    seeds = [seed(j, k, r) for j in grid for k in range(32) for r in ("path", "label", "augmentation")]
    assert len(seeds) == len(set(seeds)) == 3072
    assert 32*WORK == 1048576 and 32*DRAW_COUNT == 524288
    assert 32*DRAW_COUNT*6144 == 3221225472
    assert "stable_recovery_v2/session_20261019" in ANCHOR
    assert RESOURCE_AUDIT == ANCHOR.replace("result.json", "audit.json")
    assert DIRECTORY.endswith("physical_window_reference_v1/session_20261019")
    with pytest.raises(ValueError):
        summarize([])


def test_complete_geometry_pilot_summary_is_not_p2_authorization():
    old = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        moment = log_moments(torch.linspace(-20, -19, DRAW_COUNT, dtype=torch.float64))
        out = summarize([{**j, "moments": [moment, moment]} for j in jobs()])
        assert len(out["rows"]) == 4 and len(out["within_guide_mean_agreements"]) == 2
        assert not any(out[k] for k in ("p1_qualified", "p2_qualified", "large_training_authorized"))
        assert all(r["components"][0]["summary"]["count"] == 8*DRAW_COUNT for r in out["rows"])
        assert all(r["paired_difference"]["relative_difference"] == 0 for r in out["rows"])
        assert all(r["fresh_precision_forecast"]["draws"] >= 262144 for r in out["rows"])
    finally:
        torch.set_num_threads(old)
