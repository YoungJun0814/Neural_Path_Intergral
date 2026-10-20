import pytest
import torch

from experiments.post_audit_matched_window_mechanism import (
    ARMS,
    DRAW_COUNT,
    WORK,
    identity,
    jobs,
    seed,
    summarize,
)
from src.path_integral.matched_window_diagnostics import contribution_diagnostics
from src.path_integral.structural_v2_reference import log_moments


def test_mechanism_grid_namespace_work_and_role_separation():
    grid = jobs()
    assert len(grid) == len({identity(j) for j in grid}) == 48
    seeds = [seed(j, k, role) for j in grid for k in range(32) for role in ("path", "label", "augmentation")]
    assert len(set(seeds)) == len(seeds) == 4608
    assert len(ARMS) == 3 and 48*WORK == 1572864
    assert 48*DRAW_COUNT*6144 == 4831838208
    with pytest.raises(ValueError):
        summarize([])


def test_mechanism_summary_preserves_mass_and_reports_all_six_contrasts_without_gates():
    old = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        logs = torch.linspace(-20., -19., DRAW_COUNT, dtype=torch.float64)
        moment = log_moments(logs)
        diagnostic = contribution_diagnostics(logs, torch.zeros(DRAW_COUNT, dtype=torch.int64),
            torch.ones((DRAW_COUNT, 6), dtype=torch.float64), components=373)
        out = summarize([{**j, "moments": [moment, moment], "diagnostics": diagnostic} for j in jobs()])
        assert len(out["rows"]) == len(out["contrasts"]) == 6
        assert all(r["diagnostics"]["count"] == 8*DRAW_COUNT for r in out["rows"])
        assert all(r["paired_difference"]["relative_difference"] == 0 for r in out["rows"])
        assert not any(out[k] for k in ("p1_qualified", "p2_qualified", "large_training_authorized"))
    finally:
        torch.set_num_threads(old)
