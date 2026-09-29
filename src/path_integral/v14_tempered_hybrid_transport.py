"""Exact balance mixture of V14 residual and tempered target transports."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from src.path_integral.baseline_framework import BaselineCostLedger
from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.cameron_martin_basis import CameronMartinBasis
from src.path_integral.finite_rank_gaussian_transport import (
    DefensiveFiniteRankGaussianMixture,
    FiniteRankGaussianComponent,
    combine_defensive_gaussian_mixtures,
)
from src.path_integral.rbergomi_local_volterra_transport import (
    LocalVolterraTransportTrainingConfig,
    train_local_volterra_transport,
)
from src.path_integral.tempered_target_transport import (
    TemperedTargetTransportConfig,
    TemperedTargetTransportResult,
    fit_tempered_target_transport,
)


@dataclass(frozen=True)
class V14TemperedHybridConfig:
    target: TemperedTargetTransportConfig
    local: LocalVolterraTransportTrainingConfig
    target_mass: float = 0.5

    def __post_init__(self) -> None:
        if not math.isfinite(self.target_mass) or not 0.0 < self.target_mass < 1.0:
            raise ValueError("target mass must lie strictly between zero and one")


@dataclass(frozen=True)
class V14TemperedHybridResult:
    proposal: DefensiveFiniteRankGaussianMixture
    training_cost: BaselineCostLedger
    target: TemperedTargetTransportResult
    local_component_count: int
    target_mass: float


def convert_local_proposal(
    means: tuple[tuple[float, ...], ...],
    weights: tuple[float, ...],
) -> DefensiveFiniteRankGaussianMixture:
    if not means:
        raise ValueError("local proposal has no components")
    dimension = len(means[0])
    components = tuple(
        FiniteRankGaussianComponent(
            mean=torch.tensor(mean, dtype=torch.float64),
            directions=torch.empty((dimension, 0), dtype=torch.float64),
            variance_eigenvalues=torch.empty(0, dtype=torch.float64),
        )
        for mean in means
    )
    return DefensiveFiniteRankGaussianMixture(
        components=components,
        weights=torch.tensor(weights, dtype=torch.float64),
    )


def fit_v14_tempered_hybrid_transport(
    problem: RBergomiBaselineProblem,
    basis: CameronMartinBasis,
    *,
    target_seed: int,
    local_seed: int,
    config: V14TemperedHybridConfig,
) -> V14TemperedHybridResult:
    """Train both families; final evaluation remains ordinary exact IS."""

    if target_seed == local_seed:
        raise ValueError("target and local training seeds must be distinct")
    target = fit_tempered_target_transport(
        problem,
        basis,
        config=config.target,
    )
    local = train_local_volterra_transport(
        problem,
        training_seed=local_seed,
        config=config.local,
    )
    local_proposal = convert_local_proposal(
        local.proposal.component_means,
        local.proposal.component_weights,
    )
    proposal = combine_defensive_gaussian_mixtures(
        (target.proposal, local_proposal),
        (config.target_mass, 1.0 - config.target_mass),
    )
    return V14TemperedHybridResult(
        proposal=proposal,
        training_cost=target.training_cost.plus(local.proposal.training_cost),
        target=target,
        local_component_count=len(local_proposal.components),
        target_mass=config.target_mass,
    )
