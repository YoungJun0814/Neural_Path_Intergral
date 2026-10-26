import pytest

from experiments.post_audit_analytic_residual_batch import Budget, compare, inputs


def test_analytic_stage_budget_never_borrows_physics_from_scalar():
    budget = Budget(100, 2, 60)
    budget.charge(100)
    budget.physics(2)
    with pytest.raises(TimeoutError):
        budget.charge(1)
    with pytest.raises(TimeoutError):
        budget.physics(1)
    assert budget.scalar == 100 and budget.physical == 2


def test_comparison_needs_numerics_reference_agreement_and_envelope():
    fast = {"rows": [{"moments": [{"log_estimate": -3., "log_monotone_envelope_lower": -4., "log_monotone_envelope_upper": -2.}]}],
        "all_relative_numerically_resolved": True}
    assert compare(fast, [[-3.]])["pass"]
    assert not compare(fast, [[-2.5]])["pass"]
    fast["all_relative_numerically_resolved"] = False
    assert not compare(fast, [[-3.]])["pass"]
    with pytest.raises(ValueError):
        compare(fast, [])


def test_validation_cannot_omit_a_failed_or_uncomfortable_context():
    with pytest.raises(ValueError, match="all six"):
        inputs({"contexts": []}, {"records": []})
