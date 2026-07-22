"""Fail-closed aggregate gate tests for V5 reference generation."""

from __future__ import annotations

from copy import deepcopy

from experiments.g11_v5_reference import _reference_gates


def _cell() -> dict:
    method = {
        "final_samples": 100,
        "estimate": 0.1,
        "variance": 0.09,
        "standard_error": 0.03,
    }
    return {
        "methods": [deepcopy(method), deepcopy(method)],
        "gate": {
            "reference_se_contract": True,
            "independent_agreement": True,
            "eta_zero_agreement": True,
        },
    }


def test_reference_gates_accept_a_complete_finite_cell() -> None:
    gates = _reference_gates([_cell()], expected_cells=1)
    assert all(gates.values())


def test_reference_gates_reject_empty_or_incomplete_matrix() -> None:
    empty = _reference_gates([], expected_cells=1)
    incomplete = _reference_gates([_cell()], expected_cells=2)
    assert not empty["complete_reference_matrix"]
    assert not empty["all_method_summaries_finite"]
    assert not incomplete["complete_reference_matrix"]
    assert not incomplete["all_method_summaries_finite"]


def test_reference_gates_reject_nonfinite_method_and_failed_agreement() -> None:
    cell = _cell()
    cell["methods"][0]["estimate"] = float("nan")
    cell["gate"]["independent_agreement"] = False
    gates = _reference_gates([cell], expected_cells=1)
    assert not gates["all_method_summaries_finite"]
    assert not gates["all_independent_crosschecks_agree"]
