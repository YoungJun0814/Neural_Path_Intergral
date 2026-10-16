import math

import pytest
import torch

from experiments.post_audit_envelope_precision_validation import decision, grid, summarize
from experiments.post_audit_p1_recovery_p2_parallel import identity, seed
from src.path_integral.structural_v2_reference import log_moments


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_envelope_fixed_grid_and_work_budget():
    jobs = grid()
    assert len(jobs) == 2016
    assert len(set(identity(j) for j in jobs)) == len(jobs)
    streams = [seed(identity(j), s) for j in jobs for s in ("path", "label", "whole")]
    assert len(set(streams)) == len(streams)
    work = sum(4096 if j["target"] == "paired" else 254976 for j in jobs)+128
    assert work == 128680064
    assert work < 160000000


def test_bounded_moment_merge_matches_full_raw_vectors():
    records, raw = [], torch.linspace(-10., -9., 32, dtype=torch.float64)
    for method in ("guide-iid", "envelope-iid"):
        for rep in range(2):
            for batch in range(2):
                part = raw[rep*16+batch*8:rep*16+(batch+1)*8]
                records.append({"method": method, "target": "paired", "rep": rep, "wall": .01,
                                "moments": {"mu": log_moments(part), "risk": log_moments(part-2)}})
    for method in ("natural-smc", "proposal-smc"):
        for target in ("mu", "risk"):
            for rep in range(8):
                records.append({"method": method, "target": target, "rep": rep, "wall": .01,
                                "log_normalizer": float(raw[rep]-(2 if target == "risk" else 0))})
    summary = summarize(records)
    for s in summary:
        if "iid" in s["method"]:
            expected = log_moments(raw-(2 if s["target"] == "risk" else 0))
            for k in ("count", "log_mean", "relative_se", "maximum_fraction"):
                assert s["summary"][k] == pytest.approx(expected[k], rel=1e-11, abs=1e-12)
            assert s["sensitivity"]["unit_count"] == 2


def test_envelope_scoped_success_does_not_open_full_gates():
    rows = [{"method": method, "target": target,
             "summary": {"log_mean": -10., "relative_se": .001, "maximum_fraction": .001},
             "sensitivity": {"maximum_leave_one_out_relative_shift": .001}}
            for method in ("guide-iid", "envelope-iid", "natural-smc", "proposal-smc") for target in ("mu", "risk")]
    output = decision(rows)
    assert output["high_eta_parent0_development_precision_pass"]
    assert not any(output[k] for k in ("p1_qualified", "p2_qualified", "large_training_authorized"))
    assert len(output["agreements"]) == 12
    rows[0]["summary"]["log_mean"] += math.log(1.2)
    assert not decision(rows)["high_eta_parent0_development_precision_pass"]
