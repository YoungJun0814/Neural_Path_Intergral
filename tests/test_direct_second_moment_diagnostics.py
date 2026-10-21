import copy
import math

import pytest
import torch

from src.path_integral.direct_second_moment_diagnostics import direct_m2_blocks
from src.path_integral.structural_v2_reference import log_moments


def sample():
    x = torch.arange(1, 81, dtype=torch.float64).reshape(8, 10)
    return x, [log_moments(b.log()) for b in x]


def test_direct_x2_blocks_not_probability_or_raw_x4_se():
    x, parts = sample()
    expected = float(x.square().mean().log())
    result = direct_m2_blocks(parts, bootstrap_seed=1, expected_log_second=expected)
    assert result["log_mean_m2"] == pytest.approx(expected)
    assert result["maximum_observed_squared_contribution_share"] == pytest.approx(float(x.square().max()/x.square().sum()))
    means = x.square().mean(1)
    assert result["sensitivity"]["between_unit_relative_se"] == pytest.approx(float(means.std()/math.sqrt(8)/means.mean()))
    assert result["raw_iid_m2_se_available"] is False
    assert result["count"] == 80


def test_missing_mismatched_or_nonfinite_evidence_rejected():
    x, parts = sample()
    log_second = float(x.square().mean().log())
    for changed in (parts[:-1], [*parts[:-1], {**parts[-1], "count": 11}]):
        with pytest.raises(ValueError):
            direct_m2_blocks(changed, bootstrap_seed=1, expected_log_second=log_second)
    with pytest.raises(ValueError, match="mismatch"):
        direct_m2_blocks(parts, bootstrap_seed=1, expected_log_second=log_second+.001)
    changed = copy.deepcopy(parts)
    changed[0]["log_sum_squares"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        direct_m2_blocks(changed, bootstrap_seed=1, expected_log_second=log_second)
