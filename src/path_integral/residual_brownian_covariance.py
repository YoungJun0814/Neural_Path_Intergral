"""Normalized, mean-preserving, multi-time residual/next-Brownian proposals.

These are finite-grid Gaussian mixtures, not continuous-time drift controls.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.special import logsumexp

from .gaussian_span_marginalization import GaussianMixtureShiftSpec
from .local_variance_reference import validate_seeds

ARMS = ("base", "residual", "independent", "coupled")


def validate_points(z: torch.Tensor, dimension: int) -> None:
    if (z.ndim != 2 or z.shape[1] != dimension or len(z) < 1 or z.dtype != torch.float64
            or z.device.type != "cpu" or z.requires_grad or not torch.isfinite(z).all()):
        raise ValueError("finite detached nonempty CPU float64 coordinates required")


def correction(a: np.ndarray, b: np.ndarray, variance: float, sign: int) -> tuple[np.ndarray, ...]:
    coefficient = .5*(1-1/variance)
    return (coefficient*a*a-.5*math.log(variance),
            coefficient*(a*a+b*b)-math.log(variance),
            .25*(1-1/(2*variance-1))*(a+sign*b)**2-.5*math.log(2*variance-1))


def transform(epsilon: torch.Tensor, rows: torch.Tensor, times: torch.Tensor,
              arm: str, variance: float, sign: int) -> None:
    odd, even = 2*times-1, 2*times
    if arm == "residual":
        epsilon[rows, odd] *= math.sqrt(variance)
    elif arm == "independent":
        epsilon[rows, odd] *= math.sqrt(variance)
        epsilon[rows, even] *= math.sqrt(variance)
    elif arm == "coupled":
        coefficient = (math.sqrt(2*variance-1)-1)/2*(epsilon[rows, odd]+sign*epsilon[rows, even])
        epsilon[rows, odd] += coefficient
        epsilon[rows, even] += sign*coefficient
    else:
        raise ValueError("declared covariance arm required")


@dataclass(frozen=True)
class ResidualBrownianCovarianceFamily:
    base: GaussianMixtureShiftSpec
    variance: float = 5.
    alpha: float = .5
    price_sign: int = 1
    chunk: int = 256

    def __post_init__(self) -> None:
        b = self.base
        if (b.dimension < 4 or b.dimension % 2 or b.means.dtype != torch.float64
                or b.means.device.type != "cpu" or b.means.requires_grad or b.weights.requires_grad
                or b.components < 2 or not bool((b.means[0] == 0).all())
                or bool((b.means[1:] == 0).all(1).any()) or float(b.weights[0]) < .1-1e-14
                or isinstance(self.variance, bool) or not math.isfinite(self.variance) or self.variance <= 1
                or isinstance(self.alpha, bool) or not math.isfinite(self.alpha) or not 0 <= self.alpha < 1
                or type(self.price_sign) is not int or self.price_sign not in (-1, 1)
                or type(self.chunk) is not int or self.chunk < 1):
            raise ValueError("exact natural-first shifted CPUfloat64 mixture and valid parameters required")

    @property
    def dimension(self) -> int:
        return self.base.dimension

    @property
    def times(self) -> int:
        return self.dimension//2-1

    def log_densities(self, z: torch.Tensor) -> torch.Tensor:
        """Columns base/residual/independent/coupled, all against standard p."""
        validate_points(z, self.dimension)
        means = self.base.means.numpy()
        weights = self.base.weights.numpy()
        intercept = np.log(weights)-.5*np.square(means).sum(1)
        output = []
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            for start in range(0, len(z), self.chunk):
                batch = z[start:start+self.chunk]
                points = batch.numpy()
                components = (batch @ self.base.means.T).numpy()+intercept
                base = logsumexp(components, axis=1)
                if self.alpha == 0:
                    output.append(np.repeat(base[:, None], 4, axis=1))
                    continue
                enlarged = np.full((len(points), 3), -np.inf, dtype=np.float64)
                for j in range(1, self.times+1):
                    a = points[:, 2*j-1, None]-means[None, 1:, 2*j-1]
                    b = points[:, 2*j, None]-means[None, 1:, 2*j]
                    for index, add in enumerate(correction(a, b, self.variance, self.price_sign)):
                        block = logsumexp(components[:, 1:]+add, axis=1)
                        enlarged[:, index] = np.logaddexp(enlarged[:, index], block)
                retained = np.logaddexp(components[:, 0],
                    logsumexp(components[:, 1:], axis=1)+math.log1p(-self.alpha))
                transported = enlarged+math.log(self.alpha/self.times)
                candidates = np.logaddexp(retained[:, None], transported)
                output.append(np.concatenate((base[:, None], candidates), axis=1))
        result = torch.from_numpy(np.concatenate(output))
        if not torch.isfinite(result).all():
            raise FloatingPointError("covariance mixture density unresolved; no clipping")
        return result

    def sample(self, count: int, arm: str, *, path_seed: int, label_seed: int) -> torch.Tensor:
        validate_seeds(path_seed, label_seed)
        if type(count) is not int or count < 1 or arm not in ARMS:
            raise ValueError("positive count and declared arm required")
        generator = torch.Generator().manual_seed(label_seed)
        labels = torch.multinomial(self.base.weights, count, replacement=True, generator=generator)
        z = torch.randn((count, self.dimension), dtype=torch.float64,
                        generator=torch.Generator().manual_seed(path_seed))
        if arm != "base":
            active = (labels > 0) & (torch.rand(count, generator=generator, dtype=torch.float64) < self.alpha)
            rows = torch.arange(count)[active]
            times = torch.randint(1, self.times+1, (len(rows),), generator=generator)
            transform(z, rows, times, arm, self.variance, self.price_sign)
        return z+self.base.means[labels]

    def reference_log_density(self, z: torch.Tensor) -> torch.Tensor:
        validate_points(z, self.dimension)
        odd, even = z[:, 1:-1:2].numpy(), z[:, 2::2].numpy()
        values = correction(odd, even, self.variance, self.price_sign)[2]
        result = np.logaddexp(math.log(.1), logsumexp(values, axis=1)+math.log(.9/self.times))
        if not np.isfinite(result).all():
            raise FloatingPointError("guide-free reference density unresolved")
        return torch.from_numpy(result)

    def reference_sample(self, count: int, *, path_seed: int, label_seed: int) -> torch.Tensor:
        validate_seeds(path_seed, label_seed)
        if type(count) is not int or count < 1:
            raise ValueError("positive reference sample count required")
        generator = torch.Generator().manual_seed(label_seed)
        weights = torch.full((self.times+1,), .9/self.times, dtype=torch.float64)
        weights[0] = .1
        labels = torch.multinomial(weights, count, replacement=True, generator=generator)
        z = torch.randn((count, self.dimension), dtype=torch.float64,
                        generator=torch.Generator().manual_seed(path_seed))
        rows = torch.arange(count)[labels > 0]
        transform(z, rows, labels[rows], "coupled", self.variance, self.price_sign)
        return z
