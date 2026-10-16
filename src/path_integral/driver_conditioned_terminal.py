"""Exact Gaussian price-residual integration conditional on all left drivers.

This is an alternative probability estimator for the same finite-grid law,
not a replacement for the original proposal's raw second-moment integrand.
Dense preparation/scoring is deliberately bounded to small-grid development.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from src.physics_engine import strict_lognormal_variance

from .baselines.rbergomi_common import RBergomiBaselineProblem
from .cached_volterra_terminal import CachedVolterraTerminal
from .path_functionals import TerminalThresholdTask


@dataclass(frozen=True)
class DriverConditionedTerminal:
    problem: RBergomiBaselineProblem
    driver_rows: torch.Tensor
    white_basis: torch.Tensor
    driver_factor: torch.Tensor
    brownian_mean_factor: torch.Tensor

    @classmethod
    def create(cls, problem: RBergomiBaselineProblem) -> DriverConditionedTerminal:
        if not 2 <= problem.steps <= 256:
            raise ValueError("dense driver conditioning is limited to 2..256 steps")
        cache = CachedVolterraTerminal.create(problem)
        n, kernel = problem.steps, cache.kernel
        rows = torch.zeros((n-1, 2*n), dtype=torch.float64)
        for i in range(1, n):
            if i > 1:
                rows[i-1, :2*(i-1):2] = math.sqrt(2*problem.hurst)*kernel.local_cholesky[0, 0]*kernel.historical_kernel[1:i].flip(0)
            rows[i-1, 2*(i-1):2*i] = math.sqrt(2*problem.hurst)*kernel.local_cholesky[1]
        basis, factor = torch.linalg.qr(rows.T, mode="reduced")
        if float(factor.diagonal().abs().min()) <= 1e-12:
            raise ValueError("driver projection is numerically rank deficient")
        torch.testing.assert_close(basis.T @ basis, torch.eye(n-1, dtype=torch.float64), rtol=0, atol=2e-12)
        torch.testing.assert_close(basis @ factor, rows.T, rtol=2e-12, atol=2e-12)
        torch.testing.assert_close(rows.square().sum(1), kernel.volterra_variance[1:n], rtol=2e-12, atol=1e-14)
        # B maps the original local normals to N Brownian increments.
        brownian = torch.zeros((n, 2*n), dtype=torch.float64)
        brownian[torch.arange(n), 2*torch.arange(n)] = kernel.local_cholesky[0, 0]
        return cls(problem, rows, basis, factor, brownian @ basis)

    @property
    def dimension(self) -> int:
        return self.problem.steps-1

    def project(self, original: torch.Tensor) -> torch.Tensor:
        self.problem._validate_latent(original, self.problem.local_dimension)
        return original @ self.white_basis

    def conditional_price(self, white: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if (white.ndim != 2 or white.shape[1] != self.dimension or len(white) < 1
                or white.dtype != torch.float64 or white.device.type != "cpu" or not torch.isfinite(white).all()):
            raise ValueError("finite CPU float64 whitened driver rows required")
        p = self.problem
        driver = white @ self.driver_factor
        # The full vector of future monitoring drivers is a conditioning
        # statistic, not a future-looking control in the original SDE.
        compensation = .5*p.eta**2*self.driver_rows.square().sum(1)
        after = strict_lognormal_variance(p.eta*driver-compensation, xi=p.xi)
        left_variance = torch.cat((torch.full((len(white), 1), p.xi, dtype=torch.float64), after), 1)
        roots = left_variance.sqrt()
        integrated = p.step_dt*left_variance.sum(1)
        conditional_dw = white @ self.brownian_mean_factor.T
        mean = math.log(p.spot)-.5*integrated+p.rho*(roots*conditional_dw).sum(1)
        removed = (roots @ self.brownian_mean_factor).square().sum(1)
        variance = integrated-p.rho**2*removed
        if (not torch.isfinite(mean).all() or not torch.isfinite(variance).all()
                or not (variance > 0).all()
                or (variance < (1-p.rho**2)*integrated*(1-2e-12)).any()):
            raise FloatingPointError("conditional Gaussian price variance contract failed")
        return mean, variance

    def log_probability(self, white: torch.Tensor) -> torch.Tensor:
        mean, variance = self.conditional_price(white)
        task = self.problem.task
        assert isinstance(task, TerminalThresholdTask)
        return torch.special.log_ndtr((math.log(task.level)-mean)/variance.sqrt())
