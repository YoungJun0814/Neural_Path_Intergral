"""Exact coarse whitened Gaussian coordinates conditional on fine local noise."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .baselines.rbergomi_common import RBergomiBaselineProblem
from .rbergomi_coupling import adjacent_local_gaussian_coefficients
from .rbergomi_fft import blp_fft_kernel


@dataclass(frozen=True)
class AdjacentLatentAdapter:
    fine_matrix: torch.Tensor
    augmentation_matrix: torch.Tensor

    @classmethod
    def create(cls, problem: RBergomiBaselineProblem) -> AdjacentLatentAdapter:
        n = problem.steps
        if n < 2 or n % 2:
            raise ValueError("even positive fine grid required")
        coefficients = adjacent_local_gaussian_coefficients(problem.simulator(), fine_dt=problem.step_dt)
        coarse = blp_fft_kernel(problem.simulator(), n_steps=n//2, step_dt=2*problem.step_dt).local_cholesky
        local = coefficients.first_cell_cholesky
        fine = coefficients.second_cell_cholesky
        raw = torch.stack((torch.cat((local[0, :2], fine[0, :2], local[0, 2:])),
                           torch.cat((local[2, :2], fine[1, :2], local[2, 2:]))))
        white = torch.linalg.solve_triangular(coarse, raw, upper=False)
        a, b = torch.zeros((n, 2*n), dtype=torch.float64), torch.zeros((n, n//2), dtype=torch.float64)
        for k in range(n//2):
            a[2*k:2*k+2, 4*k:4*k+4] = white[:, :4]
            b[2*k:2*k+2, k] = white[:, 4]
        return cls(a, b)

    def __post_init__(self) -> None:
        a, b = self.fine_matrix, self.augmentation_matrix
        if (a.ndim != 2 or b.ndim != 2 or a.shape[0] < 2 or a.shape[0] % 2
                or a.shape[1] != 2*a.shape[0] or b.shape != (a.shape[0], a.shape[0]//2)
                or any(x.dtype != torch.float64 or x.device.type != "cpu" or not torch.isfinite(x).all() for x in (a, b))
                or torch.max(torch.abs(a @ a.T+b @ b.T-torch.eye(len(a), dtype=torch.float64))) > 2e-10):
            raise ValueError("invalid exact Gaussian adjacent whitening")

    def coarse(self, fine: torch.Tensor, augmentation: torch.Tensor) -> torch.Tensor:
        if (fine.ndim != 2 or fine.shape[1] != self.fine_matrix.shape[1]
                or augmentation.shape != (len(fine), self.augmentation_matrix.shape[1])
                or any(x.dtype != torch.float64 or x.device.type != "cpu" or not torch.isfinite(x).all()
                       for x in (fine, augmentation))):
            raise ValueError("invalid fine/independent-augmentation coordinates")
        return fine @ self.fine_matrix.T+augmentation @ self.augmentation_matrix.T
