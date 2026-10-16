"""Guide-independent defensive, disjoint-block Gaussian scale mixture."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def validate_seeds(path_seed: int, label_seed: int) -> None:
    if (any(isinstance(s, bool) or not isinstance(s, int) or not 0 < s < 2**63
            for s in (path_seed, label_seed)) or path_seed == label_seed):
        raise ValueError("distinct positive integer path/label seeds required")


@dataclass(frozen=True)
class LocalVarianceReference:
    dimension: int
    block_size: int = 2
    variance: float = 25.
    defensive_mass: float = .1

    def __post_init__(self) -> None:
        if (any(isinstance(v, bool) or not isinstance(v, int) or v < 1
                for v in (self.dimension, self.block_size))
                or self.dimension % self.block_size
                or isinstance(self.variance, bool) or not math.isfinite(self.variance)
                or self.variance <= 1 or isinstance(self.defensive_mass, bool)
                or not math.isfinite(self.defensive_mass) or not 0 < self.defensive_mass < 1):
            raise ValueError("invalid defensive block-scale mixture")

    @property
    def blocks(self) -> int:
        return self.dimension // self.block_size

    def log_q_over_p(self, z: torch.Tensor) -> torch.Tensor:
        if (z.ndim != 2 or z.shape[1] != self.dimension or z.dtype != torch.float64
                or z.device.type != "cpu" or not torch.isfinite(z).all()):
            raise ValueError("invalid scale-mixture coordinates")
        norms = z.reshape(-1, self.blocks, self.block_size).square().sum(2)
        component = .5*(1-1/self.variance)*norms-.5*self.block_size*math.log(self.variance)
        inflated = torch.logsumexp(component, 1) + math.log((1-self.defensive_mass)/self.blocks)
        return torch.logaddexp(torch.full_like(inflated, math.log(self.defensive_mass)), inflated)

    def sample(self, count: int, *, path_seed: int, label_seed: int) -> torch.Tensor:
        validate_seeds(path_seed, label_seed)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("positive integer sample count required")
        weights = torch.full((self.blocks+1,), (1-self.defensive_mass)/self.blocks, dtype=torch.float64)
        weights[0] = self.defensive_mass
        labels = torch.multinomial(weights, count, replacement=True,
                                   generator=torch.Generator().manual_seed(label_seed))
        z = torch.randn((count, self.blocks, self.block_size), dtype=torch.float64,
                        generator=torch.Generator().manual_seed(path_seed))
        active = labels > 0
        rows = torch.arange(count)[active]
        z[rows, labels[active]-1, :] *= math.sqrt(self.variance)
        return z.reshape(count, self.dimension)
