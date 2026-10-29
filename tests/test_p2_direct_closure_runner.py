import math

import pytest
import torch

from experiments.post_audit_p2_direct_closure import ARMS, jobs, seed_key, summarize
from experiments.post_audit_small_grid_model_restart import configuration
from src.path_integral.structural_v2_reference import log_moments


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def fixture(noisy=False):
    cells = [c["id"] for c in configuration()["cells"]]
    mapping, records = {}, []
    for c in cells:
        for a in ARMS:
            for grid in ("fine", "coarse"):
                mapping[seed_key({"cell": c, "arm": a, "rep": 0}, "bootstrap-"+grid)] = len(mapping)+1
    for job in jobs(cells):
        factor = (100. if job["rep"] == 0 else 1.) if noisy else 1+.001*job["rep"]
        vector = [math.log(.1*factor), math.log(.099*factor)]
        records.append({**job, "log_component_normalizers": vector,
            "stages": [{"surviving_initial_ancestors": 3, "ess_before_resampling": 20.,
                        "second_mutation_acceptance": .5}]*63, "wall_seconds": 1.})
    anchor = {"summary": {"iid_rows": [{"cell": c,
        "summary": log_moments(torch.full((100,), math.log(.10035), dtype=torch.float64)),
        "precision_and_sensitivity": {"pass": True}} for c in cells]}}
    return records, anchor, mapping


def test_fixed_paired_fine_design_counts_and_gaussian_augmentation():
    fixed = jobs(["a", "b"])
    assert len(fixed) == len({(j["cell"], j["arm"], j["rep"]) for j in fixed}) == 32
    assert 128*(1+62*2) == 16000
    assert 2*(32*16000+4*16*(1+14*2)) == 1027712 < 1050000
    assert 5*4096//2 == 10240


def test_empirical_success_cannot_invent_continuum_proof_or_open_p2():
    rows, anchor, mapping = fixture()
    result = summarize(rows, anchor, mapping)
    assert result["primary_fine_reference_development_qualified"]
    assert result["primary_adjacent_development_qualified"]
    assert all(r["reference_equivalence"]["pass"] for r in result["rows"])
    assert not result["P2_qualified"] and not result["large_training_authorized"]
    assert not result["p1_all_q_risk_measured"]


def test_unqualified_precision_does_not_compute_equivalence():
    rows, anchor, mapping = fixture(noisy=True)
    result = summarize(rows, anchor, mapping)
    assert not result["primary_fine_reference_development_qualified"]
    for r in result["rows"]:
        assert not r["reference_equivalence"]["evaluated"]
        assert not r["fine_reference_development_qualified"]


def test_missing_and_duplicate_wholes_rejected_before_summary():
    rows, anchor, mapping = fixture()
    for bad in (rows[:-1], rows+[rows[0]]):
        with pytest.raises(ValueError):
            summarize(bad, anchor, mapping)
