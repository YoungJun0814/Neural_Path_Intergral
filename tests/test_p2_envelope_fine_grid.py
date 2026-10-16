from dataclasses import replace

import pytest
import torch

from experiments.post_audit_p2_envelope_fine_grid import (
    BATCH,
    BATCHES,
    STEPS,
    batch_values,
    seed,
    unit_ids,
)
from experiments.post_audit_p2_fine_mesh_recovery import summarize
from src.path_integral.cached_volterra_terminal import CachedVolterraTerminal
from src.path_integral.fft_excursion_reference import (
    AdjacentBlockLatentAdapter,
    FFTCoverageEnvelopeReference,
)
from tests.test_p1_recovery_p2_parallel import problem


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_fixed_grid_work_and_disjoint_seeds():
    rows = [{"cell": {"id": c}, "parent_training_rep": rep}
            for c in ("canonical", "high-eta") for rep in range(5)]
    units = unit_ids(rows)
    seeds = [seed(u, s) for u in units for s in ("path", "label", "augmentation")]
    assert len(units) == len(set(units)) == 768
    assert len(seeds) == len(set(seeds))
    assert len(units)*2*BATCH == 786432
    assert 2*BATCH*BATCHES*sum(n+n//2 for n in STEPS) == 1409286144


def test_one_fine_likelihood_paired_summary_and_replay():
    p = replace(problem(), steps=16)
    r = FFTCoverageEnvelopeReference.create(p)
    c = CachedVolterraTerminal.create(replace(p, steps=8))
    a = AdjacentBlockLatentAdapter.create(p)
    original = batch_values(r, c, a, "unit-test", count=128)
    replay = batch_values(r, c, a, "unit-test", count=128)
    for key in ("fine", "coarse", "sign", "difference_log_absolute", "peak_cell_share", "seeds"):
        assert original[key] == replay[key]
    assert original["actual_work"] == 256
    assert original["time_step_evaluations"] == 3072
    stats = summarize([original])
    assert stats["paired_mean_identity_error_scaled"] < 1e-12
    assert stats["scope"] == "paired_delta_method_development_not_continuum_certificate"
