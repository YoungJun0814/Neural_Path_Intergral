"""Exactly normalized Gaussian conditional-excursion mixture, no mean fitting."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .baselines.rbergomi_common import RBergomiBaselineProblem
from .collective_variance_reference import CollectiveVarianceReference
from .local_variance_reference import validate_seeds


@dataclass(frozen=True)
class ConditionalExcursionReference:
    units: torch.Tensor
    price_indices: torch.Tensor
    price_sign: float = 1.
    thresholds: tuple[float, ...] = (4., 5., 6.)
    price_threshold: float = 1.
    defensive_mass: float = .1

    @classmethod
    def create(cls, p: RBergomiBaselineProblem) -> ConditionalExcursionReference:
        geometry = CollectiveVarianceReference.create(p)
        return cls(geometry.units, geometry.price_indices, -1. if p.rho > 0 else 1.)

    def __post_init__(self) -> None:
        CollectiveVarianceReference(self.units, self.price_indices)
        if (self.price_sign not in (-1., 1.) or not self.thresholds
                or any(isinstance(x, bool) or not math.isfinite(x) or not 0 < x <= 10 for x in self.thresholds)
                or len(set(self.thresholds)) != len(self.thresholds)
                or isinstance(self.price_threshold, bool) or not math.isfinite(self.price_threshold)
                or not 0 < self.price_threshold <= 10 or isinstance(self.defensive_mass, bool)
                or not 0 < self.defensive_mass < 1):
            raise ValueError("invalid exact conditional-excursion mixture")

    @property
    def dimension(self) -> int:
        return self.units.shape[1]

    def log_q_over_p(self, z: torch.Tensor) -> torch.Tensor:
        if (z.ndim != 2 or z.shape[1] != self.dimension or z.dtype != torch.float64
                or z.device.type != "cpu" or not torch.isfinite(z).all()):
            raise ValueError("invalid excursion coordinates")
        y = z @ self.units.T
        price = self.price_sign*z[:, self.price_indices]
        tail_price = float(torch.special.log_ndtr(torch.tensor(-self.price_threshold, dtype=torch.float64)))
        terms = []
        weight = math.log((1-self.defensive_mass)/(2*len(self.units)*len(self.thresholds)))
        for threshold in self.thresholds:
            active = y >= threshold
            tail = float(torch.special.log_ndtr(torch.tensor(-threshold, dtype=torch.float64)))
            single = active.sum(1).to(torch.float64)
            joint = (active & (price >= self.price_threshold)).sum(1).to(torch.float64)
            terms += [single.log()+weight-tail, joint.log()+weight-tail-tail_price]
        terms.append(torch.full((len(z),), math.log(self.defensive_mass), dtype=torch.float64))
        return torch.logsumexp(torch.stack(terms, 1), 1)

    def sample(self, count: int, *, path_seed: int, label_seed: int) -> torch.Tensor:
        validate_seeds(path_seed, label_seed)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("positive count required")
        components = 2*len(self.units)*len(self.thresholds)
        weights = torch.full((components+1,), (1-self.defensive_mass)/components, dtype=torch.float64)
        weights[0] = self.defensive_mass
        labels = torch.multinomial(weights, count, replacement=True, generator=torch.Generator().manual_seed(label_seed))
        generator = torch.Generator().manual_seed(path_seed)
        z = torch.randn((count, self.dimension), dtype=torch.float64, generator=generator)
        rows = torch.arange(count)[labels > 0]
        labels = labels[labels > 0]-1
        times = labels//(2*len(self.thresholds))
        threshold_indices = (labels//2) % len(self.thresholds)
        thresholds = torch.tensor(self.thresholds, dtype=torch.float64)[threshold_indices]
        u = self.units[times]
        uniforms = torch.rand((len(rows), 2), dtype=torch.float64, generator=generator)
        if bool((uniforms <= 0).any()):
            raise FloatingPointError("zero inverse-CDF uniform; no clipping/replacement")
        values = -torch.special.ndtri(uniforms[:, 0]*torch.special.ndtr(-thresholds))
        z[rows] += (values-(z[rows]*u).sum(1))[:, None]*u
        joint = labels % 2 == 1
        selected_rows, j = rows[joint], self.price_indices[times[joint]]
        price_values = -torch.special.ndtri(uniforms[joint, 1]*torch.special.ndtr(torch.tensor(-self.price_threshold, dtype=torch.float64)))
        z[selected_rows, j] = self.price_sign*price_values
        return z
