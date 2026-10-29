import math

import numpy as np
import pytest
import torch

from experiments.post_audit_grouped_window_coupling import (
    ARMS,
    Budget,
    agreement,
    jobs,
    kernel_decision,
    summarize,
    unit_id,
    values,
)
from src.path_integral.stratified_paired_risk import sufficient


def test_fixed_design_complete_fresh_cost_and_no_duplicate_jobs():
    fixed = jobs(["a", "b"])
    assert len(fixed) == len({unit_id(j) for j in fixed}) == 80
    count = sum({"direct": 4096, "heldout": 3*1024, "reference": 2048}[j["phase"]] for j in fixed)
    assert count == 278528
    assert count+2*3*64 == 278912 < 280000
    replays = [j for j in fixed if j["block"] == 0 and (j["phase"] != "direct" or j["arm"] == "window")]
    assert len(replays) == 6
    assert sum({"direct": 4096, "heldout": 3072, "reference": 2048}[j["phase"]] for j in replays) == 18432


@pytest.mark.parametrize("speed,error,passed", [(1.2, 1e-9, True), (1.199, 0., False),
                                               (5., 1.01e-9, False), (1.3, 0., True)])
def test_kernel_gate_is_conjunctive_not_numerical_or_fast_only(speed, error, passed):
    rows = [{"order": o, "old_all_seconds": speed, "grouped_all_seconds": 1.} for o in ("AB", "BA", "AB")]
    decision = kernel_decision(rows, error)
    assert decision["kernel_candidate"] is passed
    assert decision["fresh_window_pilot_allowed"] is passed


def test_kernel_rejects_incomplete_or_reordered_timing_family():
    with pytest.raises(ValueError):
        kernel_decision([], 0.)


def test_budget_fails_before_exceeding_fixed_work():
    budget = Budget(2, 60, 3)
    budget.charge(2, 3)
    with pytest.raises(TimeoutError):
        budget.charge(1)
    assert budget.physical == 2 and budget.old_passes == 3
    with pytest.raises(ValueError):
        budget.charge(True)


def test_agreement_not_evaluated_without_both_precisions_and_not_truth_certificate():
    assert not agreement({}, {}, 0, False)["evaluated"]
    same = {"log_mean": [-12.], "relative_se": [.001]}
    result = agreement(same, same, 0, True)
    assert result["development_equivalence"]
    assert result["not_finite_sample_truth_certificate"]
    noisy = {"log_mean": [-12.], "relative_se": [.5]}
    assert not agreement(noisy, noisy, 0, True)["development_equivalence"]


def rows():
    part = sufficient(torch.log(torch.linspace(.5, 1.5, 16, dtype=torch.float64))[:, None])
    risk = sufficient(torch.log(torch.stack([torch.linspace(.5, 1.5, 16, dtype=torch.float64),
                                            torch.linspace(.5, 1.5, 16, dtype=torch.float64),
                                            torch.linspace(.5, 1.5, 16, dtype=torch.float64)], 1)))
    return [{**j, "strata": ([part] if j["phase"] == "direct" else [risk]*(3 if j["phase"] == "heldout" else 1)),
             "event_strata": [part]*(3 if j["phase"] == "heldout" else 1)} for j in jobs(["a", "b"])]


def test_complete_decision_keeps_production_and_p2_closed_without_signal():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = summarize(rows(), ["a", "b"])
        assert result["decision"] == "NO_ADOPTION_WINDOW_PRECISION_OR_COVERAGE_UNRESOLVED"
        assert not result["production_adoption"] and not result["P2_qualified"]
        assert not result["large_training_authorized"]
        assert result["fitting_or_selection_runs"] == 0
    finally:
        torch.set_num_threads(old)


def test_duplicate_or_missing_units_cannot_pass_full_decision():
    fixed = rows()
    for broken in (fixed[:-1], fixed+[fixed[0]]):
        with pytest.raises(ValueError):
            summarize(broken, ["a", "b"])


@pytest.mark.parametrize("phase", ["direct", "heldout", "reference"])
def test_value_estimator_uses_full_mixture_and_known_strata(monkeypatch, phase):
    import experiments.post_audit_grouped_window_coupling as study

    g = -.5
    density = torch.tensor([1., 2., 3.], dtype=torch.float64)

    def fake_scores(model, z, budget):
        return torch.full((len(z),), g, dtype=torch.float64), density.expand(len(z), -1), torch.full(
            (len(z),), 4., dtype=torch.float64), {"density_all_arms_and_R": 0., "physics": 0.}

    monkeypatch.setattr(study, "sample", lambda model, n, job, arm, mapping: torch.zeros((n, 4), dtype=torch.float64))
    monkeypatch.setattr(study, "scores", fake_scores)
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        row, arrays = values({"cell": "a", "phase": phase, "arm": "window", "block": 0},
                             {"a": None}, {}, Budget(100000, 60))
    finally:
        torch.set_num_threads(old)
    denominator = 3. if phase == "direct" else (float(torch.logsumexp(density, 0))-math.log(3) if phase == "heldout" else 4.)
    expected = [g-denominator] if phase == "direct" else (2*g-denominator-density).tolist()
    assert len(row["strata"]) == (len(ARMS) if phase == "heldout" else 1)
    np.testing.assert_allclose(arrays["logs"][0, 0], expected, atol=1e-12)
    np.testing.assert_allclose(arrays["event"], g-denominator, atol=1e-12)
