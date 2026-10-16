"""Normalized coverage envelope for spikes, moderate excursions and global tails."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .baselines.rbergomi_common import RBergomiBaselineProblem
from .collective_variance_reference import CollectiveVarianceReference
from .conditional_excursion_reference import ConditionalExcursionReference
from .local_variance_reference import validate_seeds


@dataclass(frozen=True)
class CoverageEnvelopeReference:
    excursion: ConditionalExcursionReference
    collective: CollectiveVarianceReference

    @classmethod
    def create(cls, p: RBergomiBaselineProblem) -> CoverageEnvelopeReference:
        geometry = CollectiveVarianceReference.create(p)
        excursion = ConditionalExcursionReference(geometry.units, geometry.price_indices,
                                                  -1. if p.rho > 0 else 1., (2., 3., 4., 5., 6.))
        return cls(excursion, geometry)

    def __post_init__(self) -> None:
        if (self.excursion.dimension != self.collective.dimension
                or self.excursion.defensive_mass != .1 or self.collective.defensive_mass != .1):
            raise ValueError("coverage envelope requires matching dimension and .1 floors")

    @property
    def dimension(self) -> int:
        return self.excursion.dimension

    @property
    def units(self) -> torch.Tensor:
        return self.collective.units

    @property
    def price_indices(self) -> torch.Tensor:
        return self.collective.price_indices

    def log_radial_over_p(self, z: torch.Tensor) -> torch.Tensor:
        if (z.ndim != 2 or z.shape[1] != self.dimension or z.dtype != torch.float64
                or z.device.type != "cpu" or not torch.isfinite(z).all()):
            raise ValueError("invalid radial coordinates")
        inflated = -.5*self.dimension*math.log(2.)+.25*z.square().sum(1)
        return torch.logaddexp(torch.full_like(inflated, math.log(.1)), math.log(.9)+inflated)

    def log_q_over_p(self, z: torch.Tensor) -> torch.Tensor:
        result = torch.logsumexp(torch.stack((math.log(.4)+self.excursion.log_q_over_p(z),
                                             math.log(.4)+self.collective.log_q_over_p(z),
                                             math.log(.2)+self.log_radial_over_p(z)), 1), 1)
        if not torch.isfinite(result).all():
            raise FloatingPointError("coverage envelope density overflow")
        return result

    def sample(self, count: int, *, path_seed: int, label_seed: int) -> torch.Tensor:
        validate_seeds(path_seed, label_seed)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("positive integer count required")
        labels_generator = torch.Generator().manual_seed(label_seed)
        labels = torch.multinomial(torch.tensor([.4, .4, .2], dtype=torch.float64), count,
                                   replacement=True, generator=labels_generator)
        path_generator = torch.Generator().manual_seed(path_seed)
        child_paths = torch.randint(1, 2**62, (3,), generator=path_generator).tolist()
        child_labels = torch.randint(1, 2**62, (3,), generator=labels_generator).tolist()
        if len(set(child_paths+child_labels)) != 6:
            raise ValueError("child stream seed collision")
        result = torch.empty((count, self.dimension), dtype=torch.float64)
        for k in range(3):
            rows = torch.arange(count)[labels == k]
            if not len(rows):
                continue
            if k == 0:
                result[rows] = self.excursion.sample(len(rows), path_seed=child_paths[k], label_seed=child_labels[k])
            elif k == 1:
                result[rows] = self.collective.sample(len(rows), path_seed=child_paths[k], label_seed=child_labels[k])
            else:
                z = torch.randn((len(rows), self.dimension), dtype=torch.float64,
                                generator=torch.Generator().manual_seed(child_paths[k]))
                inflated = torch.rand(len(rows), dtype=torch.float64,
                                      generator=torch.Generator().manual_seed(child_labels[k])) >= .1
                scales = torch.ones(len(rows), dtype=torch.float64)
                scales[inflated] = math.sqrt(2.)
                result[rows] = z*scales[:, None]
        return result
