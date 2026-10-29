import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.path_integral.finite_grid_geometry_protocol import (
    EvaluationBudget,
    GeometryProtocol,
    GeometryTask,
    PhaseLimit,
)

CONFIG = Path(__file__).resolve().parents[1]/"configs/finite_grid_geometry/reset_v1.json"


def protocol():
    return GeometryProtocol.from_dict(json.loads(CONFIG.read_text(encoding="utf-8")))


def test_registered_development_and_holdout_roundtrip_are_exact():
    p = protocol()
    assert GeometryProtocol.from_dict(p.to_dict()) == p
    assert p.planned_pilot_evaluations == 2097152
    assert p.max_redesigns == 1
    assert len({t.estimand_digest for t in p.tasks}) == 8
    assert len(p.digest) == 64


@pytest.mark.parametrize("phase", ["FR1", "FR2", "FR3", "FR4"])
@pytest.mark.parametrize("role", ["reference_pilot", "training", "selection", "final", "heldout", "audit"])
def test_confirmation_cannot_leak_through_any_development_role(phase, role):
    with pytest.raises(ValueError, match="sealed"):
        protocol().authorize("C01", phase, role)


def test_confirmation_requires_new_candidate_and_allocation_contracts():
    with pytest.raises(ValueError, match="digests"):
        protocol().authorize("C01", "FR5", "final")
    assert protocol().authorize("C01", "FR5", "final", candidate_digest="a"*64,
                                allocation_digest="b"*64).split == "confirmation"
    with pytest.raises(ValueError, match="candidate-blind"):
        protocol().authorize("C01", "FR5", "reference_pilot")


@pytest.mark.parametrize("field,value", [("payoff", "call"), ("split", "test"),
                                         ("steps", True), ("eta", float("nan")),
                                         ("rho", 1.), ("strike_ratio", 0.)])
def test_invalid_task_and_unbounded_payoff_are_rejected(field, value):
    with pytest.raises(ValueError):
        replace(protocol().tasks[0], **{field: value})


def test_strict_schema_rejects_foreign_law_and_hidden_fields():
    raw = protocol().to_dict()
    for changed in ({**raw, "law": "euler"}, {**raw, "extra": True}):
        with pytest.raises(ValueError, match="schema"):
            GeometryProtocol.from_dict(changed)
    changed = copy.deepcopy(raw)
    changed["tasks"][1] = changed["tasks"][0]
    with pytest.raises(ValueError, match="distinct"):
        GeometryProtocol.from_dict(changed)


def test_future_job_count_is_checked_before_simulation():
    with pytest.raises(ValueError, match="cap"):
        replace(protocol(), pilot_power=20)


def test_seed_roles_and_methods_do_not_overlap_and_are_deterministic():
    p = protocol()
    values = [p.seed("D01", role, i, stream) for role in ("training", "final")
              for i in range(4) for stream in ("conditional_mc", "conditional_rqmc")]
    assert len(set(values)) == len(values)
    assert values[0] == p.seed("D01", "training", 0, "conditional_mc")


def test_budget_reserves_failed_operations_without_silent_extension(monkeypatch):
    b = EvaluationBudget(PhaseLimit("FR2", 8, 1.))
    b.reserve(5)
    with pytest.raises(RuntimeError, match="cap"):
        b.reserve(4)
    assert b.evaluations == 5
    monkeypatch.setattr("src.path_integral.finite_grid_geometry_protocol.time.perf_counter",
                        lambda: b.started+2.)
    with pytest.raises(TimeoutError):
        b.reserve(1)
    assert b.evaluations == 5


def test_followup_runner_does_not_reset_the_phase_budget():
    b = EvaluationBudget(PhaseLimit("FR2", 8000000, 3600.), already_spent=2162688)
    b.reserve(5837312)
    with pytest.raises(RuntimeError, match="cap"):
        b.reserve(1)


def test_seed_task_and_phase_binding_cannot_be_omitted():
    p = protocol()
    with pytest.raises(ValueError, match="registered"):
        p.authorize("D99", "FR2", "reference_pilot")
    with pytest.raises(ValueError, match="phase"):
        p.authorize("D01", "FR9", "final")
    with pytest.raises(ValueError, match="FR2"):
        p.authorize("D01", "FR3", "reference_pilot")


def test_task_constructor_uses_original_model_and_positive_strike():
    t = GeometryTask("D01", "development", .1, 1., -.7, 1., .8, 32, "digital")
    assert t.problem().local_dimension == 64
    assert t.problem().task.level == 80
