import pytest
import torch

from experiments.post_audit_bank_ensemble_coverage import (
    ARMS,
    CELLS,
    CHUNK,
    DRAW_COUNT,
    identity,
    jobs,
    seed,
    summarize,
    unit_values,
)
from src.path_integral.cached_volterra_terminal import CachedVolterraTerminal
from src.path_integral.physical_window_guide import PhysicalWindowGuide
from tests.test_p1_recovery_p2_parallel import problem
from tests.test_physical_window_guide import KWARGS


def test_all_arms_fixed_work_disjoint_seeds_and_complete_family_required():
    work = jobs()
    assert len(work) == 64 and len({identity(j) for j in work}) == 64
    values = [seed(j["cell"], j["arm"], j["block"], k, role)
              for j in work for k in range(DRAW_COUNT//CHUNK) for role in ("path", "label")]
    values += [seed(c, "preflight", 0, 0, role) for c in CELLS for role in ("path", "label")]
    assert len(set(values)) == len(values)
    assert set(ARMS) == {"uniform", "fit0", "fit1", "ensemble"}
    assert 64*DRAW_COUNT+2*CHUNK == 524800
    with pytest.raises(ValueError, match="complete"):
        summarize([])


def test_small_grid_production_path_reconstructs_probability_and_replays():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        p = problem(32)
        guide = PhysicalWindowGuide.create(p, kind="mixed-window", **KWARGS)
        prepared = CachedVolterraTerminal.create(p), {a: guide.spec for a in ARMS}
        job = {"cell": "independent-small-oracle", "arm": "ensemble", "block": 0}
        first = unit_values(job, prepared, count=CHUNK)
        second = unit_values(job, prepared, count=CHUNK)
        assert first["coverage"] == second["coverage"]
        assert first["moments"] == second["moments"]
        assert first["moments"]["count"] == CHUNK
        assert sum(first["coverage"]["bin_counts"]) == CHUNK
    finally:
        torch.set_num_threads(old)
