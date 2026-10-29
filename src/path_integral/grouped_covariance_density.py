"""Exact local-mean grouping for the frozen covariance mixture density.

No rounding, pruning, density fitting, or change to the proposal distribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.special import logsumexp

from .gaussian_span_marginalization import GaussianMixtureShiftSpec
from .residual_brownian_covariance import (
    ARMS,
    ResidualBrownianCovarianceFamily,
    correction,
    validate_points,
)


@dataclass(frozen=True)
class LocalMeanGroups:
    pair_means: np.ndarray
    order: np.ndarray
    starts: np.ndarray
    sorted_group_ids: np.ndarray

    @classmethod
    def create(cls, pair_means: np.ndarray) -> LocalMeanGroups:
        if (pair_means.ndim != 2 or pair_means.shape[1] != 2 or not len(pair_means)
                or pair_means.dtype != np.float64 or not np.isfinite(pair_means).all()):
            raise ValueError("finite nonempty float64 component pair means required")
        unique, inverse = np.unique(pair_means, axis=0, return_inverse=True)
        order = np.argsort(inverse, kind="stable")
        ids = inverse[order]
        starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1])+1]
        return cls(unique, order, starts, ids)

    def aggregate(self, logs: np.ndarray) -> np.ndarray:
        if (logs.ndim != 2 or logs.shape[1] != len(self.order) or not np.isfinite(logs).all()):
            raise ValueError("finite batch-by-component log terms required")
        # np.take keeps component rows contiguous; advanced column indexing
        # otherwise creates a strided layout for every segment reduction.
        ordered = np.take(logs, self.order, axis=1)
        maximum = np.maximum.reduceat(ordered, self.starts, axis=1)
        # Every group has its own scale: no global rescaling loses a small group
        # that may later receive a large covariance correction.
        totals = np.add.reduceat(np.exp(ordered-maximum[:, self.sorted_group_ids]), self.starts, axis=1)
        return maximum+np.log(totals)


@dataclass(frozen=True)
class GroupedCovarianceDensity:
    family: ResidualBrownianCovarianceFamily
    groups: tuple[LocalMeanGroups, ...]
    intercept: np.ndarray

    @classmethod
    def create(cls, family: ResidualBrownianCovarianceFamily) -> GroupedCovarianceDensity:
        original = family.base
        base = GaussianMixtureShiftSpec(original.means.detach().clone(), original.weights.detach().clone())
        copied = ResidualBrownianCovarianceFamily(base, family.variance, family.alpha,
                                                  family.price_sign, family.chunk)
        groups = tuple(LocalMeanGroups.create(base.means[1:, [2*j-1, 2*j]].numpy())
                       for j in range(1, copied.times+1))
        intercept = np.log(base.weights.numpy())-.5*np.square(base.means.numpy()).sum(1)
        return cls(copied, groups, intercept)

    def log_densities(self, z: torch.Tensor, arms: tuple[str, ...] = ARMS) -> torch.Tensor:
        validate_points(z, self.family.dimension)
        if not arms or len(set(arms)) != len(arms) or any(arm not in ARMS for arm in arms):
            raise ValueError("nonempty distinct declared density arms required")
        indices = [ARMS.index(arm)-1 for arm in arms if arm != "base"]
        output = []
        family = self.family
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            for begin in range(0, len(z), family.chunk):
                batch = z[begin:begin+family.chunk]
                points = batch.numpy()
                logs = (batch @ family.base.means.T).numpy()+self.intercept
                base = logsumexp(logs, axis=1)
                if not indices or family.alpha == 0:
                    output.append(np.repeat(base[:, None], len(arms), axis=1))
                    continue
                expanded = np.full((len(batch), len(indices)), -np.inf)
                for j, group in enumerate(self.groups, 1):
                    grouped = group.aggregate(logs[:, 1:])
                    a = points[:, 2*j-1, None]-group.pair_means[None, :, 0]
                    b = points[:, 2*j, None]-group.pair_means[None, :, 1]
                    additions = correction(a, b, family.variance, family.price_sign)
                    for column, index in enumerate(indices):
                        expanded[:, column] = np.logaddexp(expanded[:, column],
                            logsumexp(grouped+additions[index], axis=1))
                retained = np.logaddexp(logs[:, 0], logsumexp(logs[:, 1:], axis=1)+math.log1p(-family.alpha))
                candidates = np.logaddexp(retained[:, None], expanded+math.log(family.alpha/family.times))
                columns, candidate = [], 0
                for arm in arms:
                    columns.append(base if arm == "base" else candidates[:, candidate])
                    candidate += arm != "base"
                output.append(np.stack(columns, axis=1))
        result = torch.from_numpy(np.concatenate(output))
        if not torch.isfinite(result).all():
            raise FloatingPointError("grouped covariance density numerical range unresolved")
        return result

    def log_density(self, z: torch.Tensor, arm: str) -> torch.Tensor:
        return self.log_densities(z, (arm,))[:, 0]

    def group_counts(self) -> list[int]:
        return [len(group.pair_means) for group in self.groups]
