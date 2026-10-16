"""Exact fixed-orthogonal conditional laws of identity-covariance mixtures.

Coordinates x=z@rotation; A precedes B. No data-dependent rotations, numerical
quadrature, or covariance-rank approximation are performed here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .finite_rank_gaussian_transport import DefensiveFiniteRankGaussianMixture
from .gaussian_mixture_marginal import require_shift_mixture
from .local_variance_reference import validate_seeds


@dataclass(frozen=True)
class JointShiftConditional:
    rotation: torch.Tensor
    means: torch.Tensor
    weights: torch.Tensor
    inner_dimension: int

    @classmethod
    def create(cls, r: DefensiveFiniteRankGaussianMixture, rotation: torch.Tensor,
               inner_dimension: int) -> JointShiftConditional:
        require_shift_mixture(r)
        if (rotation.ndim != 2 or rotation.shape != (r.dimension, r.dimension)
                or rotation.dtype != torch.float64 or rotation.device.type != "cpu"):
            raise ValueError("rotation must match CPU float64 full dimension")
        obj = cls(rotation.clone(), torch.stack([c.mean for c in r.components]) @ rotation,
                  r.weights.clone(), inner_dimension)
        return obj

    def __post_init__(self) -> None:
        u, m, w = self.rotation, self.means, self.weights
        if (u.ndim != 2 or u.shape[0] != u.shape[1] or u.shape[0] < 2
                or m.ndim != 2 or m.shape != (w.numel(), u.shape[0]) or w.ndim != 1
                or any(t.dtype != torch.float64 or t.device.type != "cpu"
                       or not torch.isfinite(t).all() for t in (u, m, w))
                or isinstance(self.inner_dimension, bool) or not isinstance(self.inner_dimension, int)
                or not 1 <= self.inner_dimension < u.shape[0]
                or bool((w <= 0).any()) or abs(float(w.sum())-1) > 1e-12
                or float(torch.max(torch.abs(u.T @ u-torch.eye(u.shape[0], dtype=torch.float64)))) > 2e-10
                or not bool((m == 0).all(1).any())):
            raise ValueError("invalid fixed orthogonal defensive conditional mixture")

    @property
    def outer_dimension(self) -> int:
        return self.rotation.shape[0]-self.inner_dimension

    def _validate(self, z: torch.Tensor, dimension: int) -> None:
        if (z.ndim != 2 or z.shape[1] != dimension or z.dtype != torch.float64
                or z.device.type != "cpu" or not torch.isfinite(z).all()):
            raise ValueError("invalid conditional coordinates")

    def split(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate(z, self.rotation.shape[0])
        x = z @ self.rotation
        return x[:, :self.outer_dimension], x[:, self.outer_dimension:]

    def join(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        self._validate(a, self.outer_dimension)
        self._validate(b, self.inner_dimension)
        if len(a) != len(b):
            raise ValueError("conditional batch counts differ")
        return torch.cat((a, b), 1) @ self.rotation.T

    def outer_component_logs(self, a: torch.Tensor) -> torch.Tensor:
        self._validate(a, self.outer_dimension)
        ma = self.means[:, :self.outer_dimension]
        return a @ ma.T-.5*ma.square().sum(1)+self.weights.log()

    def log_marginal_over_prior(self, a: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(self.outer_component_logs(a), 1)

    def conditional_log_weights(self, a: torch.Tensor) -> torch.Tensor:
        logs = self.outer_component_logs(a)
        return logs-torch.logsumexp(logs, 1, keepdim=True)

    def log_conditional_over_prior(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        self._validate(b, self.inner_dimension)
        if len(a) != len(b):
            raise ValueError("conditional batch counts differ")
        mb = self.means[:, self.outer_dimension:]
        return torch.logsumexp(self.conditional_log_weights(a)+b @ mb.T-.5*mb.square().sum(1), 1)

    def sample_inner(self, a: torch.Tensor, *, path_seed: int, label_seed: int) -> torch.Tensor:
        validate_seeds(path_seed, label_seed)
        probabilities = self.conditional_log_weights(a).exp()
        labels = torch.multinomial(probabilities, 1,
                                   generator=torch.Generator().manual_seed(label_seed)).squeeze(1)
        return (torch.randn((len(a), self.inner_dimension), dtype=torch.float64,
                            generator=torch.Generator().manual_seed(path_seed))
                + self.means[labels, self.outer_dimension:])
