"""Fail-closed gate tests for the V5 fresh-training baseline protocol."""

from __future__ import annotations

from copy import deepcopy

from experiments.g11_v5_baseline_qualification import _qualification_gates


def _record() -> dict:
    return {
        "training_seed": 1,
        "evaluation_seed_roles_are_disjoint": True,
        "cem": {"converged": True},
        "pure_cem_slis": {"estimate": 0.1, "standard_error": 0.01},
        "defensive_cem_mixture": {
            "estimate": 0.1,
            "standard_error": 0.01,
            "maximum_full_likelihood_bound_violation": 0.0,
        },
        "crude_mc": {
            "estimate": 0.1,
            "standard_error": 0.01,
            "zero_hit_censored": False,
            "exact_binomial_interval": {"upper": 0.2},
        },
    }


def test_baseline_gates_accept_a_complete_valid_record() -> None:
    gates = _qualification_gates([_record()], expected_records=1)
    assert all(gates.values())


def test_baseline_gates_reject_nonconvergence_and_incomplete_matrix() -> None:
    record = deepcopy(_record())
    record["cem"]["converged"] = False
    gates = _qualification_gates([record], expected_records=2)
    assert not gates["all_cem_fits_converged"]
    assert not gates["complete_cluster_matrix"]


def test_baseline_gates_reject_claimed_seed_overlap_and_zero_hit_without_upper_bound() -> None:
    record = deepcopy(_record())
    record["evaluation_seed_roles_are_disjoint"] = False
    record["crude_mc"]["zero_hit_censored"] = True
    record["crude_mc"]["exact_binomial_interval"]["upper"] = 0.0
    gates = _qualification_gates([record], expected_records=1)
    assert not gates["training_and_evaluation_seeds_disjoint"]
    assert not gates["zero_hit_cases_use_exact_intervals"]
