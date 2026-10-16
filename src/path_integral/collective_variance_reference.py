"""Normalized covariance transport of causal Volterra/next-price directions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .baselines.rbergomi_common import RBergomiBaselineProblem
from .finite_rank_gaussian_transport import (
    DefensiveFiniteRankGaussianMixture,
    FiniteRankGaussianComponent,
)
from .local_variance_reference import validate_seeds
from .volterra_excursion_guide import volterra_monitoring_operator


@dataclass(frozen=True)
class CollectiveVarianceReference:
    units: torch.Tensor
    price_indices: torch.Tensor
    variance: float = 25.
    defensive_mass: float = .1

    @classmethod
    def create(cls, problem: RBergomiBaselineProblem) -> CollectiveVarianceReference:
        operator = volterra_monitoring_operator(problem)
        units = operator/torch.linalg.vector_norm(operator, dim=1, keepdim=True)
        return cls(units, 2*torch.arange(1, problem.steps))

    def __post_init__(self) -> None:
        u, j = self.units, self.price_indices
        if (u.ndim != 2 or u.shape[0] < 1 or u.shape[1] < 2 or u.dtype != torch.float64
                or u.device.type != "cpu" or not torch.isfinite(u).all() or j.shape != (u.shape[0],)
                or j.dtype != torch.int64 or j.device.type != "cpu" or bool((j < 0).any())
                or bool((j >= u.shape[1]).any()) or j.unique().numel() != j.numel()
                or torch.max(torch.abs(u.square().sum(1)-1)) > 1e-12
                or torch.max(torch.abs(u[torch.arange(len(j)), j])) > 1e-12
                or isinstance(self.variance, bool) or not math.isfinite(self.variance) or self.variance <= 1
                or isinstance(self.defensive_mass, bool) or not 0 < self.defensive_mass < 1):
            raise ValueError("invalid causal orthogonal collective scale mixture")

    @property
    def dimension(self) -> int:
        return self.units.shape[1]

    def log_q_over_p(self, z: torch.Tensor) -> torch.Tensor:
        if (z.ndim != 2 or z.shape[1] != self.dimension or z.dtype != torch.float64
                or z.device.type != "cpu" or not torch.isfinite(z).all()):
            raise ValueError("invalid collective coordinates")
        norms = (z @ self.units.T).square()+z[:, self.price_indices].square()
        components = .5*(1-1/self.variance)*norms-math.log(self.variance)
        inflated = torch.logsumexp(components, 1)+math.log((1-self.defensive_mass)/len(self.units))
        result = torch.logaddexp(torch.full_like(inflated, math.log(self.defensive_mass)), inflated)
        if not torch.isfinite(result).all():
            raise FloatingPointError("collective density overflow; no clipping")
        return result

    def sample(self, count: int, *, path_seed: int, label_seed: int) -> torch.Tensor:
        validate_seeds(path_seed, label_seed)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("positive count required")
        weights = torch.full((len(self.units)+1,), (1-self.defensive_mass)/len(self.units), dtype=torch.float64)
        weights[0] = self.defensive_mass
        labels = torch.multinomial(weights, count, replacement=True, generator=torch.Generator().manual_seed(label_seed))
        z = torch.randn((count, self.dimension), dtype=torch.float64, generator=torch.Generator().manual_seed(path_seed))
        rows = torch.arange(count)[labels > 0]
        chosen = labels[labels > 0]-1
        u, j = self.units[chosen], self.price_indices[chosen]
        coefficient = math.sqrt(self.variance)-1
        price = z[rows, j].clone()
        projection = (z[rows]*u).sum(1)
        z[rows] += coefficient*projection[:, None]*u
        z[rows, j] += coefficient*price
        return z

    def as_gaussian_mixture(self) -> DefensiveFiniteRankGaussianMixture:
        components = [FiniteRankGaussianComponent.natural(self.dimension)]
        for u, j in zip(self.units, self.price_indices, strict=True):
            v = torch.zeros(self.dimension, dtype=torch.float64)
            v[j] = 1
            components.append(FiniteRankGaussianComponent(torch.zeros(self.dimension, dtype=torch.float64),
                torch.stack((u, v), 1), torch.full((2,), self.variance, dtype=torch.float64)))
        weights = torch.full((len(components),), (1-self.defensive_mass)/(len(components)-1), dtype=torch.float64)
        weights[0] = self.defensive_mass
        return DefensiveFiniteRankGaussianMixture(tuple(components), weights)
