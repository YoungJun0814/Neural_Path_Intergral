import pytest
import torch

from experiments.post_audit_p2_fine_resource_sweep import identity, jobs, joint_work, seed
from experiments.post_audit_resource_sleep_recovery import (
    NAMESPACE,
    recovery_summary,
    validate_power,
)


def test_recovery_preserves_original_seeds_and_full_grid_cost_without_prefix_qualification():
    grid = jobs(NAMESPACE)
    pending = grid[20:]
    assert identity(pending[0]) == "high-eta200-k1-more-temperatures-4"
    assert len(pending) == 12 and len({seed(j) for j in grid}) == 32
    assert sum(2*joint_work(j) for j in pending) == 1537024
    assert sum(6144*joint_work(j) for j in pending) == 4721737728
    assert sum(2*joint_work(j) for j in grid) == 4108288
    old = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        records = [{**j, "log_component_normalizers": [-10.+.01*j["rep"]]*2,
                    "stages": [{"surviving_initial_ancestors": 20}], "wall": 1.} for j in grid]
        out = recovery_summary(records)
        assert len(out["rows"]) == 4
        assert not any(out[k] for k in ("p1_qualified", "p2_qualified", "large_training_authorized"))
        assert all("wall_shared_laptop_seconds" not in r["fresh_precision_forecast"] for r in out["rows"])
        with pytest.raises(ValueError):
            recovery_summary(records[:20])
    finally:
        torch.set_num_threads(old)


@pytest.mark.parametrize("status", [{"ac_connected": 0, "battery_percent": 95},
                                    {"ac_connected": 1, "battery_percent": 19},
                                    {"ac_connected": 1, "battery_percent": 255}])
def test_ac_guard_rejects_battery_unknown_or_low_without_changing_power_policy(status):
    with pytest.raises(RuntimeError):
        validate_power(status)
    validate_power({"ac_connected": 1, "battery_percent": 27})
