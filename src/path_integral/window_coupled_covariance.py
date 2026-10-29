"""Finite-grid rank-one, equal-trace covariance transport over time windows."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .gaussian_span_marginalization import GaussianMixtureShiftSpec
from .local_variance_reference import validate_seeds
from .residual_brownian_covariance import ResidualBrownianCovarianceFamily, validate_points


@dataclass(frozen=True)
class WindowCoupledCovariance:
    family: ResidualBrownianCovarianceFamily
    width: int
    windows: tuple[tuple[int, int], ...]
    directions: torch.Tensor
    projected_means: torch.Tensor

    @classmethod
    def create(cls, family: ResidualBrownianCovarianceFamily, width: int) -> WindowCoupledCovariance:
        if type(width) is not int or not 1 <= width <= family.times:
            raise ValueError("in-range positive integer window width required")
        base = GaussianMixtureShiftSpec(family.base.means.detach().clone(),
                                       family.base.weights.detach().clone())
        family = ResidualBrownianCovarianceFamily(base, family.variance, family.alpha,
                                                  family.price_sign, family.chunk)
        windows = tuple((start, min(start+width, family.times+1))
                        for start in range(1, family.times+1, width))
        directions = torch.zeros((len(windows), family.dimension), dtype=torch.float64)
        for row, (start, stop) in enumerate(windows):
            indices = torch.arange(start, stop)
            directions[row, 2*indices-1] = 1/math.sqrt(2*(stop-start))
            directions[row, 2*indices] = family.price_sign/math.sqrt(2*(stop-start))
        projected = family.base.means @ directions.T
        return cls(family, width, windows, directions, projected)

    @property
    def eigenvalue(self) -> float:
        return 2*self.family.variance-1

    def log_density(self, z: torch.Tensor) -> torch.Tensor:
        family = self.family
        validate_points(z, family.dimension)
        intercept = family.base.weights.log()-.5*family.base.means.square().sum(1)
        parts = []
        for begin in range(0, len(z), family.chunk):
            batch = z[begin:begin+family.chunk]
            logs = batch @ family.base.means.T+intercept
            if family.alpha == 0:
                parts.append(torch.logsumexp(logs, 1))
                continue
            projection = batch @ self.directions.T
            expanded = torch.full((len(batch),), -math.inf, dtype=torch.float64)
            for j in range(len(self.windows)):
                x = projection[:, j, None]-self.projected_means[None, 1:, j]
                add = .5*(1-1/self.eigenvalue)*x.square()-.5*math.log(self.eigenvalue)
                expanded = torch.logaddexp(expanded, torch.logsumexp(logs[:, 1:]+add, 1))
            retained = torch.logaddexp(logs[:, 0], torch.logsumexp(logs[:, 1:], 1)+math.log1p(-family.alpha))
            parts.append(torch.logaddexp(retained, expanded+math.log(family.alpha/len(self.windows))))
        result = torch.cat(parts)
        if not torch.isfinite(result).all():
            raise FloatingPointError("window covariance density unresolved")
        return result

    def sample(self, count: int, *, path_seed: int, label_seed: int) -> torch.Tensor:
        validate_seeds(path_seed, label_seed)
        if type(count) is not int or count < 1:
            raise ValueError("positive window sample count required")
        generator = torch.Generator().manual_seed(label_seed)
        labels = torch.multinomial(self.family.base.weights, count, replacement=True, generator=generator)
        z = torch.randn((count, self.family.dimension), dtype=torch.float64,
                        generator=torch.Generator().manual_seed(path_seed))
        active = (labels > 0) & (torch.rand(count, generator=generator, dtype=torch.float64) < self.family.alpha)
        rows = torch.arange(count)[active]
        windows = torch.randint(len(self.windows), (len(rows),), generator=generator)
        w = self.directions[windows]
        z[rows] += (math.sqrt(self.eigenvalue)-1)*(z[rows]*w).sum(1)[:, None]*w
        return z+self.family.base.means[labels]

    def reference_log_density(self, z: torch.Tensor) -> torch.Tensor:
        validate_points(z, self.family.dimension)
        x = z @ self.directions.T
        logs = .5*(1-1/self.eigenvalue)*x.square()-.5*math.log(self.eigenvalue)
        result = torch.logaddexp(torch.full((len(z),), math.log(.1), dtype=torch.float64),
                                 torch.logsumexp(logs, 1)+math.log(.9/len(self.windows)))
        if not torch.isfinite(result).all():
            raise FloatingPointError("window reference density unresolved")
        return result

    def reference_sample(self, count: int, *, path_seed: int, label_seed: int) -> torch.Tensor:
        validate_seeds(path_seed, label_seed)
        if type(count) is not int or count < 1:
            raise ValueError("positive window reference count required")
        weights = torch.full((len(self.windows)+1,), .9/len(self.windows), dtype=torch.float64)
        weights[0] = .1
        labels = torch.multinomial(weights, count, replacement=True,
                                   generator=torch.Generator().manual_seed(label_seed))
        z = torch.randn((count, self.family.dimension), dtype=torch.float64,
                        generator=torch.Generator().manual_seed(path_seed))
        rows = torch.arange(count)[labels > 0]
        w = self.directions[labels[rows]-1]
        z[rows] += (math.sqrt(self.eigenvalue)-1)*(z[rows]*w).sum(1)[:, None]*w
        return z
