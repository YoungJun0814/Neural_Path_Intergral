"""Mesh-aware orthonormal bases for finite-grid Cameron--Martin controls."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CameronMartinBasis:
    """An orthonormal coefficient-to-whitened-control map.

    The BLP reference variables are independent standard normals.  A deterministic
    shift in those whitened coordinates has Cameron--Martin energy equal to its
    Euclidean squared norm.  Requiring orthonormal columns therefore makes coefficient
    energy exactly equal to control energy.
    """

    matrix: torch.Tensor
    steps: int
    drivers: int
    modes_per_driver: int

    def __post_init__(self) -> None:
        matrix = self.matrix
        if matrix.device.type != "cpu" or matrix.dtype != torch.float64:
            raise ValueError("Cameron--Martin basis must be CPU float64")
        if matrix.ndim != 2 or matrix.shape != (
            self.steps * self.drivers,
            self.modes_per_driver * self.drivers,
        ):
            raise ValueError("Cameron--Martin basis has the wrong shape")
        if not torch.isfinite(matrix).all():
            raise ValueError("Cameron--Martin basis must be finite")
        gram = matrix.T @ matrix
        identity = torch.eye(matrix.shape[1], dtype=torch.float64)
        if float(torch.amax(torch.abs(gram - identity))) > 2e-12:
            raise ValueError("Cameron--Martin basis columns must be orthonormal")

    @property
    def dimension(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def rank(self) -> int:
        return int(self.matrix.shape[1])

    def expand(self, coefficients: torch.Tensor) -> torch.Tensor:
        if coefficients.shape[-1] != self.rank:
            raise ValueError("Cameron--Martin coefficients have the wrong dimension")
        if coefficients.device.type != "cpu" or coefficients.dtype != torch.float64:
            raise ValueError("Cameron--Martin coefficients must be CPU float64")
        return coefficients @ self.matrix.T

    def project(self, control: torch.Tensor) -> torch.Tensor:
        if control.shape[-1] != self.dimension:
            raise ValueError("Cameron--Martin control has the wrong dimension")
        if control.device.type != "cpu" or control.dtype != torch.float64:
            raise ValueError("Cameron--Martin control must be CPU float64")
        return control @ self.matrix

    def projection_residual(self, control: torch.Tensor) -> torch.Tensor:
        return control - self.expand(self.project(control))


def _orthonormal_dct_basis(steps: int, modes: int) -> torch.Tensor:
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if isinstance(modes, bool) or not isinstance(modes, int) or not 1 <= modes <= steps:
        raise ValueError("modes must lie between one and steps")
    times = torch.arange(steps, dtype=torch.float64) + 0.5
    frequencies = torch.arange(modes, dtype=torch.float64)
    basis = torch.cos(math.pi * times[:, None] * frequencies[None, :] / steps)
    basis[:, 0] *= math.sqrt(1.0 / steps)
    if modes > 1:
        basis[:, 1:] *= math.sqrt(2.0 / steps)
    return basis


def build_blp_cameron_martin_basis(
    *,
    steps: int,
    modes_per_driver: int | None = None,
    drivers: int = 2,
) -> CameronMartinBasis:
    """Build a channel-separated DCT basis in BLP standard-normal coordinates.

    For the primary BLP local law, ``drivers=2`` means the two orthonormal
    within-cell coordinates of one volatility Brownian motion, not two independent
    continuum Brownian drivers.  See ``blp_cameron_martin_embedding.py`` for the
    exact isometry.
    """

    if isinstance(drivers, bool) or not isinstance(drivers, int) or drivers < 1:
        raise ValueError("drivers must be a positive integer")
    modes = steps if modes_per_driver is None else modes_per_driver
    temporal = _orthonormal_dct_basis(steps, modes)
    matrix = torch.zeros(steps * drivers, modes * drivers, dtype=torch.float64)
    for driver in range(drivers):
        matrix[driver::drivers, driver * modes : (driver + 1) * modes] = temporal
    return CameronMartinBasis(
        matrix=matrix,
        steps=steps,
        drivers=drivers,
        modes_per_driver=modes,
    )
