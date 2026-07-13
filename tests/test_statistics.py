from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.statistics import repeated_estimate_report


def test_repeated_report_computes_bias_and_coverage() -> None:
    estimates = np.array([0.09, 0.10, 0.11, 0.10])
    errors = np.full(4, 0.01)
    report = repeated_estimate_report(estimates, errors, truth=0.10)
    assert report.mean_estimate == pytest.approx(0.10)
    assert report.bias_z_score == pytest.approx(0.0, abs=1e-12)
    assert report.reported_bias_z_score == pytest.approx(0.0, abs=1e-12)
    assert report.ci_coverage == pytest.approx(1.0)
    assert report.relative_rmse > 0.0


def test_repeated_report_rejects_nonpositive_truth() -> None:
    with pytest.raises(ValueError, match="truth"):
        repeated_estimate_report(np.ones(3), np.ones(3), truth=0.0)


def test_reported_bias_z_uses_within_run_standard_errors() -> None:
    report = repeated_estimate_report(np.array([0.11, 0.11]), np.array([0.02, 0.02]), truth=0.10)
    expected_se = np.sqrt(0.02**2 + 0.02**2) / 2.0
    assert report.reported_bias_z_score == pytest.approx(0.01 / expected_se)
