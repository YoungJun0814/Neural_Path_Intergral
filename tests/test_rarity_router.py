"""Decision-boundary and independence tests for the V6 rarity router."""

from __future__ import annotations

import math

import torch

from src.path_integral.rarity_router import (
    HybridProfileOpportunity,
    RarityRouterConfig,
    RoutingWorkInterval,
    freeze_rarity_route,
)


def _config(**changes) -> RarityRouterConfig:
    return RarityRouterConfig(**changes)


def _work(method, lower, point, upper) -> RoutingWorkInterval:
    return RoutingWorkInterval(method, lower, point, upper)


def test_moderate_event_skips_hybrid_and_defaults_to_crude() -> None:
    decision = freeze_rarity_route(
        successes=80,
        trials=256,
        screening_work=256.0,
        config=_config(),
        crude_work=_work("crude", 900.0, 1000.0, 1100.0),
        dcs_work=_work("dcs_slis", 1000.0, 1200.0, 1400.0),
        hybrid_opportunity=HybridProfileOpportunity(10.0, 20.0, 100.0),
    )
    assert decision.rarity_class == "moderate"
    assert decision.action == "crude"


def test_moderate_event_can_use_certified_cheaper_dcs_without_hybrid() -> None:
    decision = freeze_rarity_route(
        successes=80,
        trials=256,
        screening_work=256.0,
        config=_config(minimum_certified_relative_saving=0.10),
        crude_work=_work("crude", 1000.0, 1100.0, 1200.0),
        dcs_work=_work("dcs_slis", 600.0, 700.0, 800.0),
    )
    assert decision.action == "dcs_slis"


def test_rare_event_profiles_hybrid_only_when_the_economic_cap_allows_it() -> None:
    common = dict(
        successes=0,
        trials=256,
        screening_work=256.0,
        config=_config(maximum_hybrid_profile_work=500.0, maximum_profile_fraction=0.25),
        crude_work=_work("crude", 9000.0, 10000.0, 11000.0),
        dcs_work=_work("dcs_slis", 7000.0, 8000.0, 9000.0),
    )
    profiled = freeze_rarity_route(
        **common,
        hybrid_opportunity=HybridProfileOpportunity(400.0, 6000.0, 450.0),
    )
    assert profiled.rarity_class == "rare"
    assert profiled.action == "profile_hybrid"
    assert profiled.effective_profile_work_cap == 450.0

    capped = freeze_rarity_route(
        **common,
        hybrid_opportunity=HybridProfileOpportunity(451.0, 6000.0, 450.0),
    )
    assert capped.action == "dcs_slis"
    assert "exceeds" in capped.reason


def test_ambiguous_probability_uses_one_more_look_then_frozen_fallback() -> None:
    early = freeze_rarity_route(
        successes=13,
        trials=256,
        screening_work=256.0,
        config=_config(maximum_screening_trials=512),
    )
    assert early.rarity_class == "ambiguous"
    assert early.action == "continue_screening"

    final = freeze_rarity_route(
        successes=26,
        trials=512,
        screening_work=512.0,
        config=_config(maximum_screening_trials=512),
    )
    assert final.action == "dcs_slis"


def test_route_hash_is_deterministic_and_changes_with_pilot_evidence() -> None:
    arguments = dict(
        trials=256,
        screening_work=256.0,
        config=_config(),
    )
    left = freeze_rarity_route(successes=0, **arguments)
    repeated = freeze_rarity_route(successes=0, **arguments)
    right = freeze_rarity_route(successes=1, **arguments)
    assert left.decision_hash == repeated.decision_hash
    assert left.decision_hash != right.decision_hash


def test_independent_final_samples_remain_unbiased_after_random_routing() -> None:
    """A Monte Carlo oracle for the conditioning argument used by Theorem V5-4."""

    probability = 0.04
    replications = 4000
    final_count = 128
    generator = torch.Generator().manual_seed(26_072_301)
    estimates = []
    for _ in range(replications):
        pilot = torch.rand(256, generator=generator) < probability
        route = freeze_rarity_route(
            successes=int(pilot.sum()),
            trials=256,
            screening_work=256.0,
            config=_config(),
        )
        assert route.action in {"continue_screening", "dcs_slis", "crude"}
        final = torch.rand(final_count, generator=generator) < probability
        estimates.append(float(final.to(torch.float64).mean()))
    mean = sum(estimates) / replications
    standard_error = math.sqrt(probability * (1.0 - probability) / (final_count * replications))
    assert abs(mean - probability) <= 5.0 * standard_error
