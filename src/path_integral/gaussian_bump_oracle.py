"""Analytic endpoints for bounded shifted/multimodal Gaussian bump toys."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GaussianBumpOracle:
    amplitudes: tuple[float, ...]
    centers: tuple[tuple[float, ...], ...]
    scales: tuple[float, ...]

    def __post_init__(self) -> None:
        if (not self.amplitudes or len(self.amplitudes) != len(self.centers)
                or len(self.scales) != len(self.centers) or not self.centers[0]
                or any(len(m) != len(self.centers[0]) for m in self.centers)
                or any(not math.isfinite(x) for m in self.centers for x in m)
                or any(not math.isfinite(x) or x <= 0 for x in (*self.amplitudes, *self.scales))
                or sum(self.amplitudes) > 1):
            raise ValueError("invalid bounded Gaussian bump mixture")

    @property
    def dimension(self) -> int:
        return len(self.centers[0])

    def log_value(self, z: torch.Tensor) -> torch.Tensor:
        if (z.ndim != 2 or z.shape[1] != self.dimension or z.dtype != torch.float64
                or z.device.type != "cpu" or not torch.isfinite(z).all()):
            raise ValueError("invalid oracle coordinates")
        means = torch.tensor(self.centers, dtype=torch.float64)
        scales = torch.tensor(self.scales, dtype=torch.float64)
        amplitudes = torch.tensor(self.amplitudes, dtype=torch.float64)
        return torch.logsumexp(amplitudes.log() - (z[:, None, :] - means).square().sum(2)
                               / (2 * scales.square()), 1)

    def log_endpoint(self, power: int) -> float:
        d = self.dimension
        if power == 1:
            terms = [math.log(a) + .5 * d * math.log(s*s/(1+s*s))
                     - sum(x*x for x in m)/(2*(1+s*s))
                     for a, m, s in zip(self.amplitudes, self.centers, self.scales, strict=True)]
        elif power == 2:
            terms = []
            for a, m, s in zip(self.amplitudes, self.centers, self.scales, strict=True):
                for b, n, t in zip(self.amplitudes, self.centers, self.scales, strict=True):
                    precision = 1 + 1/s**2 + 1/t**2
                    vector = [x/s**2 + y/t**2 for x, y in zip(m, n, strict=True)]
                    constant = sum(x*x for x in m)/s**2 + sum(x*x for x in n)/t**2
                    terms.append(math.log(a*b) - .5*d*math.log(precision)
                                 - .5*(constant - sum(x*x for x in vector)/precision))
        else:
            raise ValueError("fractional bridge endpoint is not analytic here")
        return float(torch.logsumexp(torch.tensor(terms, dtype=torch.float64), 0))
