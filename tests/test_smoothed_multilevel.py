"""Statistical-accounting tests for paired smoothed MLMC diagnostics."""

from __future__ import annotations

import math

import pytest
import torch

from src.evaluation.smoothed_multilevel import (
    paired_level_diagnostics,
    paired_mlmc_diagnostics,
)


def test_paired_report_detects_conditional_variance_reduction_and_mean_equality() -> None:
    torch.manual_seed(9501)
    conditioning = torch.randn(4_000, dtype=torch.float64)
    removed_noise = 0.8 * torch.randn(4_000, dtype=torch.float64)
    smoothed = 1.0 + conditioning
    raw = smoothed + removed_noise
    report = paired_level_diagnostics(
        raw,
        smoothed,
        raw_cost_per_path=2.0,
        smoothed_cost_per_path=2.1,
        bootstrap_replicates=200,
        bootstrap_seed=9502,
    )
    assert abs(report.paired_mean_difference_z) < 3.0
    assert report.smoothed_over_raw_variance < 0.7
    assert report.variance_ratio_ci_upper < 0.75
    assert report.raw_over_smoothed_work_ratio > 1.0


def test_mlmc_work_coefficient_uses_square_root_allocation_identity() -> None:
    torch.manual_seed(9503)
    reports = []
    for scale, raw_cost, smooth_cost in ((1.0, 1.0, 1.1), (0.4, 2.0, 2.2)):
        base = scale * torch.randn(2_000, dtype=torch.float64)
        raw = base + scale * torch.randn(2_000, dtype=torch.float64)
        reports.append(
            paired_level_diagnostics(
                raw,
                base,
                raw_cost_per_path=raw_cost,
                smoothed_cost_per_path=smooth_cost,
                bootstrap_replicates=100,
                bootstrap_seed=int(100 * raw_cost),
            )
        )
    combined = paired_mlmc_diagnostics(reports)
    expected_raw = (
        sum(math.sqrt(report.raw_variance * report.raw_cost_per_path) for report in reports) ** 2
    )
    expected_smoothed = (
        sum(
            math.sqrt(report.smoothed_variance * report.smoothed_cost_per_path)
            for report in reports
        )
        ** 2
    )
    assert combined.raw_work_coefficient == pytest.approx(expected_raw)
    assert combined.smoothed_work_coefficient == pytest.approx(expected_smoothed)
    assert combined.raw_over_smoothed_work_ratio == pytest.approx(expected_raw / expected_smoothed)


def test_zero_raw_variance_is_rejected_as_unidentifiable_ratio() -> None:
    values = torch.ones(100, dtype=torch.float64)
    with pytest.raises(ValueError, match="variance"):
        paired_level_diagnostics(
            values,
            values,
            raw_cost_per_path=1.0,
            smoothed_cost_per_path=1.0,
            bootstrap_replicates=100,
        )
