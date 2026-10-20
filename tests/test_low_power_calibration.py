import pytest

from experiments.post_audit_low_power_calibration import assert_tree_close, validate_power_samples


def samples(capacities):
    return [{"native": {"ac_connected": 1, "battery_percent": 25},
             "wmi": {"PowerOnline": True, "Discharging": False, "RemainingCapacity": c}} for c in capacities]


def test_low_power_capacity_guard_is_not_just_an_ac_icon():
    validate_power_samples(samples([20000, 20200, 20400]))
    with pytest.raises(ValueError):
        validate_power_samples(samples([20000, 19800, 19600]))
    with pytest.raises(ValueError):
        validate_power_samples(samples([20000]))
    bad = samples([20000, 20200, 20400])
    bad[1]["wmi"]["Discharging"] = True
    with pytest.raises(ValueError):
        validate_power_samples(bad)


def test_backend_replay_requires_discrete_ancestry_equality_and_close_floats():
    assert_tree_close([{"ancestor": 12, "value": -10.}], [{"ancestor": 12, "value": -10.+1e-10}])
    with pytest.raises(ValueError):
        assert_tree_close([{"ancestor": 12}], [{"ancestor": 11}])
    with pytest.raises(ValueError):
        assert_tree_close([{"value": -10.}], [{"value": -9.}])
