"""Exact marginals of identity-covariance defensive shift mixtures."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.path_integral.finite_rank_gaussian_transport import DefensiveFiniteRankGaussianMixture


def require_shift_mixture(q: DefensiveFiniteRankGaussianMixture) -> None:
    if any(c.rank != 0 for c in q.components):
        raise ValueError("conditional reference currently requires rank-zero covariance")


@dataclass(frozen=True)
class ShiftMixtureMarginal:
    means: torch.Tensor
    weights: torch.Tensor
    indices: tuple[int, ...]

    @classmethod
    def from_full(cls, q: DefensiveFiniteRankGaussianMixture,
                  indices: tuple[int, ...]) -> ShiftMixtureMarginal:
        require_shift_mixture(q)
        if (len(set(indices)) != len(indices) or any(isinstance(i, bool) or not isinstance(i, int)
                                                   or not 0 <= i < q.dimension for i in indices)):
            raise ValueError("invalid marginal indices")
        means = torch.stack([c.mean for c in q.components])[:, list(indices)].clone()
        return cls(means, q.weights.clone(), indices)

    def __post_init__(self) -> None:
        if (self.means.ndim != 2 or self.means.shape != (self.weights.numel(), len(self.indices))
                or self.means.dtype != torch.float64 or self.means.device.type != "cpu"
                or self.weights.dtype != torch.float64 or self.weights.device.type != "cpu"
                or self.weights.ndim != 1 or not torch.isfinite(self.means).all()
                or not torch.isfinite(self.weights).all() or bool((self.weights <= 0).any())
                or abs(float(self.weights.sum()) - 1) > 1e-12):
            raise ValueError("invalid normalized marginal mixture")
        if self.defensive_mass <= 0:
            raise ValueError("marginal requires an exact natural component")

    @property
    def defensive_mass(self) -> float:
        return float(self.weights[(self.means == 0).all(1)].sum())

    def log_q_over_p(self, samples: torch.Tensor) -> torch.Tensor:
        if (samples.ndim != 2 or samples.shape[1] != len(self.indices)
                or samples.dtype != torch.float64 or samples.device.type != "cpu"
                or not torch.isfinite(samples).all()):
            raise ValueError("invalid marginal samples")
        return torch.logsumexp(samples @ self.means.T - .5 * self.means.square().sum(1)
                               + self.weights.log(), 1)

    def sample(self, count: int, *, path_seed: int, label_seed: int) -> torch.Tensor:
        if isinstance(count, bool) or not isinstance(count, int) or count < 1 or path_seed == label_seed:
            raise ValueError("invalid marginal count or distinct seeds")
        labels = torch.multinomial(self.weights, count, replacement=True,
                                   generator=torch.Generator().manual_seed(label_seed))
        return torch.randn((count, len(self.indices)), dtype=torch.float64,
                           generator=torch.Generator().manual_seed(path_seed)) + self.means[labels]
