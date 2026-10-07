from __future__ import annotations

import torch

from src.path_integral.defensive_proposal_selection import (
    select_defensive_proposal_by_second_moment,
)
from src.path_integral.finite_rank_gaussian_transport import (
    DefensiveFiniteRankGaussianMixture,
    FiniteRankGaussianComponent,
)


def test_exact_risk_selector_prefers_natural_law_for_constant_payoff() -> None:
    natural = FiniteRankGaussianComponent.natural(2)
    shifted = FiniteRankGaussianComponent(
        mean=torch.tensor((2.5, 0.0), dtype=torch.float64),
        directions=torch.empty((2, 0), dtype=torch.float64),
        variance_eigenvalues=torch.empty(0, dtype=torch.float64),
    )
    proposals = (
        DefensiveFiniteRankGaussianMixture(
            components=(natural,),
            weights=torch.ones(1, dtype=torch.float64),
        ),
        DefensiveFiniteRankGaussianMixture(
            components=(natural, shifted),
            weights=torch.tensor((0.2, 0.8), dtype=torch.float64),
        ),
    )
    selected = select_defensive_proposal_by_second_moment(
        proposals,
        ("natural", "shifted"),
        lambda sample: torch.ones(sample.shape[0], dtype=torch.float64),
        sample_count=50_000,
        batch_size=5_000,
        path_seed=8001,
        label_seed=8002,
    )
    assert selected.selected_id == "natural"
    assert selected.proposal is proposals[0]
    assert selected.estimates[0].second_moment_estimate < selected.estimates[1].second_moment_estimate
    assert selected.simultaneous_oracle_excess_bound > 0.0


def test_exact_risk_selector_validates_streams_and_payoff_range() -> None:
    proposal = DefensiveFiniteRankGaussianMixture(
        components=(FiniteRankGaussianComponent.natural(1),),
        weights=torch.ones(1, dtype=torch.float64),
    )
    try:
        select_defensive_proposal_by_second_moment(
            (proposal,),
            ("candidate",),
            lambda sample: torch.full(
                (sample.shape[0],), 2.0, dtype=torch.float64
            ),
            sample_count=8,
            batch_size=4,
            path_seed=9,
            label_seed=10,
        )
    except ValueError as error:
        assert "in [0, 1]" in str(error)
    else:
        raise AssertionError("out-of-range validation payoff was accepted")
