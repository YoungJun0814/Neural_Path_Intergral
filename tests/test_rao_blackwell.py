"""Oracle tests for paired Rao--Blackwell moment diagnostics."""

from __future__ import annotations

import pytest
import torch

from src.path_integral import rao_blackwell_pair_diagnostics


def test_orthogonal_pair_has_exact_variance_decomposition() -> None:
    dcs = torch.tensor([-1.0, -1.0, 1.0, 1.0], dtype=torch.float64)
    residual = torch.tensor([-1.0, 1.0, -1.0, 1.0], dtype=torch.float64)
    diagnostics = rao_blackwell_pair_diagnostics(dcs + residual, dcs)
    assert diagnostics.residual_mean == 0.0
    assert diagnostics.dcs_residual_covariance == 0.0
    assert diagnostics.dcs_residual_correlation == 0.0
    assert diagnostics.raw_over_dcs_variance_ratio == 2.0
    assert diagnostics.variance_decomposition_error == 0.0


def test_pair_diagnostics_reject_mismatched_or_nonfinite_data() -> None:
    values = torch.ones(4, dtype=torch.float64)
    with pytest.raises(ValueError, match="matching finite"):
        rao_blackwell_pair_diagnostics(values, values[:3])
    invalid = values.clone()
    invalid[0] = torch.inf
    with pytest.raises(ValueError, match="matching finite"):
        rao_blackwell_pair_diagnostics(values, invalid)
