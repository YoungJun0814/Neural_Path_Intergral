"""Profile-map and defensive-bound tests for rBergomi hybrid sampling."""

from __future__ import annotations

import pytest
import torch

from src.path_integral import (
    RBergomiHybridTermSampler,
    TerminalThresholdTask,
    TimePiecewiseTwoDriverControl,
    rbergomi_hybrid_candidate_profiles,
    rbergomi_hybrid_profile_ids,
)
from src.physics_engine import RBergomiSimulator


def _sampler() -> RBergomiHybridTermSampler:
    simulator = RBergomiSimulator(H=0.12, eta=1.1, xi=0.04, rho=-0.6, device="cpu")
    controls = (
        TimePiecewiseTwoDriverControl(((0.0, 0.0), (0.0, 0.0)), maturity=0.25),
        TimePiecewiseTwoDriverControl(((-0.4, -1.2), (-0.25, -0.7)), maturity=0.25),
    )
    return RBergomiHybridTermSampler(
        simulator,
        controls,
        torch.tensor([0.2, 0.8], dtype=torch.float64),
        TerminalThresholdTask(94.0),
        spot=100.0,
        maturity=0.25,
        coarsest_steps=8,
        finest_level=2,
    )


def test_profile_map_contains_every_start_level_telescope() -> None:
    assert rbergomi_hybrid_profile_ids(2) == (
        "single_0",
        "single_1",
        "single_2",
        "correction_1",
        "correction_2",
    )
    assert rbergomi_hybrid_candidate_profiles(2) == {
        "start_0": ("single_0", "correction_1", "correction_2"),
        "start_1": ("single_1", "correction_2"),
        "start_2": ("single_2",),
    }


@pytest.mark.parametrize("profile_id", rbergomi_hybrid_profile_ids(2))
def test_each_profile_obeys_frozen_cost_and_defensive_bound(profile_id: str) -> None:
    sampler = _sampler()
    batch = sampler(
        profile_id,
        "pilot",
        128,
        {"proposal": 12345, "labels": 23456},
    )
    assert batch.work_units == 128 * sampler.cost_per_sample(profile_id)
    assert float(torch.amax(torch.abs(batch.values))) <= (sampler.defensive_absolute_bound + 1e-12)


def test_profile_sampler_rejects_out_of_hierarchy_identifier() -> None:
    sampler = _sampler()
    with pytest.raises(ValueError, match="outside"):
        sampler(
            "correction_3",
            "pilot",
            2,
            {"proposal": 1, "labels": 2},
        )


def test_defensive_bound_uses_declared_zero_component_not_smallest_weight() -> None:
    sampler = _sampler()
    assert sampler.declared_natural_component_weight == pytest.approx(0.2)
    assert sampler.defensive_absolute_bound == pytest.approx(5.0)
