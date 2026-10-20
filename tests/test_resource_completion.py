import pytest

from experiments.post_audit_p2_fine_resource_sweep import identity, jobs, joint_work
from experiments.post_audit_resource_completion import POLICY
from experiments.post_audit_resource_sleep_recovery import NAMESPACE, recovery_summary


def test_battery_independent_completion_preserves_nine_original_seeds_and_cost():
    pending = jobs(NAMESPACE)[23:]
    assert len(pending) == 9 and identity(pending[0]) == "high-eta200-k1-more-temperatures-7"
    assert sum(2*joint_work(j) for j in pending) == 1149184
    assert sum(6144*joint_work(j) for j in pending) == 3530293248
    assert POLICY == "battery_checks_disabled_by_explicit_user_request_2026-10-19"
    with pytest.raises(ValueError):
        recovery_summary([])
