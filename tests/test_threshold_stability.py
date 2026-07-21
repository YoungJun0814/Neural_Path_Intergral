"""Deterministic checks for margin-localized scalar-threshold bounds."""

from __future__ import annotations

import math

import pytest
import torch

from src.path_integral import (
    NORMAL_DENSITY_MAXIMUM,
    aggregate_threshold_stability,
    combine_common_and_mesh_defect,
    defensive_moment_upper_bounds,
    ratio_candidate_stability,
)


def test_ratio_candidate_bound_holds_on_denominator_good_event() -> None:
    coarse_n = torch.tensor([[-2.0, 1.0], [4.0, -3.0]], dtype=torch.float64)
    fine_n = coarse_n + torch.tensor([[0.1, -0.2], [0.3, 0.1]], dtype=torch.float64)
    coarse_b = torch.tensor([[1.0, 2.0], [0.2, 1.5]], dtype=torch.float64)
    fine_b = coarse_b + torch.tensor([[0.05, -0.1], [0.05, 0.2]], dtype=torch.float64)
    result = ratio_candidate_stability(
        fine_n,
        fine_b,
        coarse_n,
        coarse_b,
        denominator_floor=0.5,
    )
    assert torch.equal(result.good_event, torch.tensor([True, False]))
    assert result.maximum_good_event_violation <= 2e-15
    assert result.observed_candidate_error[0] <= result.candidate_error_bound[0]


@pytest.mark.parametrize("kind", ["max", "min"])
def test_aggregate_bound_separates_common_error_and_mesh_enrichment(kind: str) -> None:
    fine_common = torch.tensor([[1.1, 2.2], [-1.1, 0.4]], dtype=torch.float64)
    coarse = torch.tensor([[1.0, 2.0], [-1.0, 0.5]], dtype=torch.float64)
    extra = (
        torch.tensor([[3.0], [0.3]], dtype=torch.float64)
        if kind == "max"
        else torch.tensor([[0.2], [-2.0]], dtype=torch.float64)
    )
    result = aggregate_threshold_stability(
        torch.cat((fine_common, extra), dim=1),
        fine_common,
        coarse,
        kind=kind,
    )
    assert result.maximum_violation <= 2e-15
    assert torch.all(result.threshold_error <= result.threshold_error_bound + 2e-15)
    assert bool((result.mesh_enrichment_defect > 0.0).any())


def test_common_and_mesh_bounds_reject_negative_or_mismatched_terms() -> None:
    common = torch.tensor([0.2, 0.4], dtype=torch.float64)
    mesh = torch.tensor([0.1, 0.3], dtype=torch.float64)
    assert torch.equal(combine_common_and_mesh_defect(common, mesh), common + mesh)
    with pytest.raises(ValueError, match="nonnegative"):
        combine_common_and_mesh_defect(common, torch.tensor([-0.1, 0.3]))


def test_defensive_moment_bounds_are_linear_raw_and_quadratic_dcs() -> None:
    epsilon = 0.04
    result = defensive_moment_upper_bounds(
        good_event_l1_threshold_bound=epsilon,
        good_event_l2_threshold_bound_squared=epsilon**2,
        bad_event_probability=0.0,
        natural_weight=0.1,
    )
    assert result.raw_second_moment_upper_bound == pytest.approx(
        NORMAL_DENSITY_MAXIMUM * epsilon / 0.1
    )
    assert result.dcs_second_moment_upper_bound == pytest.approx(
        NORMAL_DENSITY_MAXIMUM**2 * epsilon**2 / 0.1
    )
    assert result.dcs_second_moment_upper_bound < result.raw_second_moment_upper_bound


def test_bad_event_probability_enters_both_bounds_without_a_margin_claim() -> None:
    result = defensive_moment_upper_bounds(
        good_event_l1_threshold_bound=0.0,
        good_event_l2_threshold_bound_squared=0.0,
        bad_event_probability=0.02,
        natural_weight=0.1,
    )
    assert math.isclose(result.raw_second_moment_upper_bound, 0.2)
    assert math.isclose(result.dcs_second_moment_upper_bound, 0.2)
