"""Training-only PCA geometry with independent split-conformal calibration.

Outside means outside typical *proposal geometry*, never outside Gaussian support
or proof of a newly discovered event mode. SMC particles have no iid coverage claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PathGeometry:
    mean: torch.Tensor
    directions: torch.Tensor
    scales: torch.Tensor
    projected_centers: torch.Tensor
    distance_threshold: float
    complement_threshold: float

    def scores(self, samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _samples(samples, self.mean.numel())
        centered = samples-self.mean
        coordinates = centered@self.directions
        projected = coordinates/self.scales
        distance = torch.cdist(projected, self.projected_centers).square().amin(1)
        residual = centered-coordinates@self.directions.T
        complement = residual.square().sum(1)/(self.mean.numel()-self.directions.shape[1])
        return distance, complement

    def outside(self, samples: torch.Tensor) -> torch.Tensor:
        distance, complement = self.scores(samples)
        return (distance > self.distance_threshold) | (complement > self.complement_threshold)


def _samples(samples: torch.Tensor, dimension: int) -> None:
    if (samples.ndim != 2 or samples.shape[1] != dimension
            or samples.device.type != "cpu" or samples.dtype != torch.float64
            or not torch.isfinite(samples).all()):
        raise ValueError("geometry requires finite CPU float64 samples")


def calibrate_path_geometry(
    training: torch.Tensor, calibration: torch.Tensor, centers: torch.Tensor,
    *, rank: int, outside_probability: float = .05,
) -> PathGeometry:
    """Fixed training geometry; two upper calibration ranks with Bonferroni alpha/2.

    Given frozen q and independent iid calibration/test draws, the marginal
    false-outside probability is <= alpha. This is not conditional coverage for
    the one realized calibration set, or coverage for event/risk-tilted samples.
    """
    if training.ndim != 2:
        raise ValueError("invalid training geometry")
    d = training.shape[1]
    for x in (training, calibration, centers):
        _samples(x, d)
    if (training.shape[0] < 2 or centers.shape[0] < 1 or not 1 <= rank < d
            or not 0 < outside_probability < 1):
        raise ValueError("invalid geometry calibration contract")
    n = calibration.shape[0]
    k = math.ceil((n+1)*(1-outside_probability/2))
    if k > n or n < 2:
        raise ValueError("not enough independent calibration samples for alpha")
    mean = training.mean(0)
    centered = training-mean
    covariance = centered.T@centered/(training.shape[0]-1)
    values, vectors = torch.linalg.eigh(covariance)
    directions = vectors[:, -rank:].contiguous()
    scales = values[-rank:].clamp_min(.1).sqrt()
    projected_centers = ((centers-mean)@directions)/scales
    provisional = PathGeometry(mean, directions, scales, projected_centers, math.inf, math.inf)
    distance, complement = provisional.scores(calibration)
    return PathGeometry(mean, directions, scales, projected_centers,
                        float(distance.sort().values[k-1]), float(complement.sort().values[k-1]))
