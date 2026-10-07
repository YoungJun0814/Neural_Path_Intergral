from __future__ import annotations

import copy

from src.path_integral.v10r1_protocol import semantic_equal


def test_semantic_equal_detects_nested_result_mutations() -> None:
    payload = {
        "work": {"total": 123.5, "units": 7},
        "decision": {"submission_authorized": False},
        "values": [1.0, 2.0],
    }
    assert semantic_equal(payload, copy.deepcopy(payload))

    work_mutation = copy.deepcopy(payload)
    work_mutation["work"]["total"] = 124.5  # type: ignore[index]
    assert not semantic_equal(payload, work_mutation)

    decision_mutation = copy.deepcopy(payload)
    decision_mutation["decision"]["submission_authorized"] = True  # type: ignore[index]
    assert not semantic_equal(payload, decision_mutation)
