import copy
import math

import pytest
import torch

from experiments.post_audit_residual_paired_pilot import PilotBudget, scores, summarize


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def moments(m1, m2, resolved=True):
    return {"moments": [{"log_estimate": math.log(m1), "relative_error_proxy_plus_tail": 1e-10},
                         {"log_estimate": math.log(m2), "relative_error_proxy_plus_tail": 1e-10}],
            "relative_numerically_resolved": resolved}


def records(quad_wall=.01):
    result = []
    for cell in ("a", "b"):
        for block in range(8):
            rows = []
            for i in range(16):
                mean = .2+.02*(i % 2)
                g = mean+(.1 if (i//2) % 2 else -.1)
                moment = moments(mean, mean**2+.01)
                rows.append({"moments": moment, "scores": scores(math.log(g), 0., moment)})
            result.append({"cell": cell, "block": block, "rows": rows,
                "draw_wall": .001, "density_wall": .001, "raw_payoff_wall": .1,
                "profile_wall": .02, "quadrature_wall": quad_wall})
    return result


def test_original_q_M2_and_conditional_event_M2_never_interchanged():
    result = scores(math.log(.4), math.log(2.), moments(.2, .1))
    assert math.exp(result[1]) == pytest.approx(.1)
    assert math.exp(result[3]) == pytest.approx(.025)
    assert math.exp(result[4]) == pytest.approx(.01)
    assert result[3] != result[4]


def test_paired_exact_toy_cost_gate_is_only_permission_for_a_new_study():
    result = summarize(records())
    assert result["decision"] == "CANDIDATE_FOR_SEPARATE_MATCHED_WALL_STUDY_ONLY"
    assert not result["P2_qualified"] and not result["large_training_authorized"]
    assert all(c["qualified_pilot_cell"] for c in result["cells"])
    assert all(c["not_matched_wall_or_fixed_precision_benchmark"] for c in result["cells"])


def test_expensive_quadrature_can_fail_practical_adoption_despite_variance_reduction():
    result = summarize(records(quad_wall=100.))
    assert result["decision"] == "STOP_SINGLE_COORDINATE_PRACTICAL_ADOPTION"
    assert all(c["event_variance_ratio_proxy"] < 1 for c in result["cells"])
    assert all(c["work_normalized_variance_proxy"] > 1 for c in result["cells"])


def test_unresolved_numerics_or_tail_concentration_veto_adoption():
    values = records()
    values[0]["rows"][0]["moments"]["relative_numerically_resolved"] = False
    assert summarize(values)["decision"] == "NO_ADOPTION_PRECISION_UNRESOLVED"
    values = records()
    values[0]["rows"][0]["scores"][0] += 20
    assert summarize(values)["decision"] == "NO_ADOPTION_PRECISION_UNRESOLVED"


def test_missing_blocks_or_contexts_rejected():
    with pytest.raises(ValueError, match="complete"):
        summarize(records()[:-1])
    values = records()
    values[0]["rows"].pop()
    with pytest.raises(ValueError, match="contexts"):
        summarize(values)


def test_negative_conditional_variance_is_not_silently_fixed():
    values = copy.deepcopy(records())
    for row in values:
        for p in row["rows"]:
            p["scores"][3] = math.log(.0001)
    with pytest.raises(FloatingPointError, match="Jensen"):
        summarize(values)


def test_production_and_replay_budget_caps_are_not_extended():
    budget = PilotBudget(10, 60, 2)
    budget.charge(10)
    budget.charge_physical(2)
    with pytest.raises(TimeoutError):
        budget.charge(1)
    with pytest.raises(TimeoutError):
        budget.charge_physical(1)
    assert budget.count == 10 and budget.physical == 2
