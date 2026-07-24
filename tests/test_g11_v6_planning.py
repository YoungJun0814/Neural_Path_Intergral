"""Replicated direct and hybrid planning contract tests."""

from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

import torch

from experiments.g11_v6_result_audit import (
    _audit_replicated_hybrid_selection,
)
from experiments.g11_v6_routed_policy import (
    _replicated_planning_selection,
)
from src.path_integral import (
    LevelBatch,
    SeedLedger,
    rbergomi_hybrid_candidate_profiles,
    rbergomi_hybrid_profile_ids,
)


class _ToyProfileSampler:
    defensive_absolute_bound = 5.0

    def __call__(
        self, profile_id: str, role: str, count: int, seeds: dict[str, int]
    ) -> LevelBatch:
        assert role == "pilot"
        generator = torch.Generator().manual_seed(seeds["proposal"])
        level = int(profile_id.split("_")[1])
        scale = 0.05 / (level + 1)
        values = scale * torch.randn(
            count, generator=generator, dtype=torch.float64
        )
        values = torch.clamp(values, -4.0, 4.0)
        return LevelBatch(
            values,
            count * self.cost_per_sample(profile_id),
            wall_seconds=0.01,
        )

    def cost_per_sample(self, profile_id: str) -> float:
        level = int(profile_id.split("_")[1])
        return float(8 * 2**level)


def test_hybrid_replicated_selection_is_independently_replayed() -> None:
    finest = 2
    profile_ids = rbergomi_hybrid_profile_ids(finest)
    candidates = rbergomi_hybrid_candidate_profiles(finest)
    config = {
        "protocol_id": "g11-v6-test-hybrid-planning",
        "selector": {
            "planning_replicates": 3,
            "samples_per_replicate": 64,
            "planning_variance_statistic": "mean_replicate_variance",
            "familywise_alpha": 0.05,
            "practical_equivalence_relative_tolerance": 0.05,
        },
        "sampling": {"allocation_safety_factor": 2.0},
    }
    state, selector_work, _wall, _cpu = _replicated_planning_selection(
        config=config,
        dcs_sampler=_ToyProfileSampler(),  # type: ignore[arg-type]
        ledger=SeedLedger(),
        cell=SimpleNamespace(cell_id="cell"),
        cluster=0,
        profile_ids=profile_ids,
        candidate_profiles=candidates,
        preprocessing_work=100.0,
        sampling_variance_target=0.01,
        minimum_final_samples=32,
        smoke=True,
    )
    selected = state.frozen_decision.selected_candidate
    selected_ids = candidates[selected]
    record = {
        "selection": asdict(state),
        "preparation": {
            "preprocessing_work": {
                "records": [
                    {"category": "routing", "work_units": 100.0},
                    {
                        "category": "selector_profile",
                        "work_units": selector_work,
                    },
                ]
            },
            "core": {
                "target": {
                    "nominal_probability": 0.2,
                    "relative_sampling_rmse": 0.5,
                },
                "allocations": [
                    {
                        "profile_id": profile_id,
                        "design_variance": (
                            state.allocation_safety_factor
                            * state.profile_planning_variances[profile_id]
                        ),
                    }
                    for profile_id in selected_ids
                ],
            },
        },
    }
    assert _audit_replicated_hybrid_selection(
        record, relative=1e-13, absolute=1e-12, required=True
    )
    record["selection"]["profile_planning_variances"][profile_ids[0]] *= 1.1
    assert not _audit_replicated_hybrid_selection(
        record, relative=1e-13, absolute=1e-12, required=True
    )
