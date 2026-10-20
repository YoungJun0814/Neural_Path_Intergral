import pytest

from experiments.post_audit_p2_fine_resource_sweep import identity, jobs, joint_work
from experiments.post_audit_resource_sleep_recovery import NAMESPACE
from experiments.post_audit_resource_stable_recovery import check_power_pair, validate_power


def test_stable_recovery_fixed_ten_original_units_and_cost():
    pending = jobs(NAMESPACE)[22:]
    assert len(pending) == 10
    assert identity(pending[0]) == "high-eta200-k1-more-temperatures-6"
    assert sum(2*joint_work(j) for j in pending) == 1278464
    assert sum(6144*joint_work(j) for j in pending) == 3927441408


@pytest.mark.parametrize("before,after", [(31, 31), (31, 32), (31, 30), (100, 100)])
def test_stable_ac_allows_charging_and_one_point_rounding(before, after):
    check_power_pair({"ac_connected": 1, "battery_percent": before},
                     {"ac_connected": 1, "battery_percent": after})


@pytest.mark.parametrize("after", [{"ac_connected": 1, "battery_percent": 28},
                                  {"ac_connected": 0, "battery_percent": 40},
                                  {"ac_connected": 1, "battery_percent": 24},
                                  {"ac_connected": 1, "battery_percent": 255}])
def test_power_drop_unknown_low_or_disconnect_cannot_qualify(after):
    with pytest.raises(RuntimeError):
        check_power_pair({"ac_connected": 1, "battery_percent": 31}, after)
    with pytest.raises(RuntimeError):
        validate_power({"ac_connected": 1, "battery_percent": 29}, minimum=30)
