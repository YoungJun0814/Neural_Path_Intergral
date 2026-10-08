"""Scoped immutable preparation for the original CPU float64 terminal law."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.finite_rank_gaussian_transport import DefensiveFiniteRankGaussianMixture
from src.path_integral.gaussian_mixture_marginal import require_shift_mixture
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.rbergomi_fft import BLPFFTKernel, blp_fft_kernel
from src.path_integral.structural_v2_conditional_reference import LastPairCache
from src.physics_engine import strict_lognormal_variance


@dataclass(frozen=True)
class CachedShiftDensity:
    means: torch.Tensor
    intercepts: torch.Tensor

    @classmethod
    def create(cls, q: DefensiveFiniteRankGaussianMixture) -> CachedShiftDensity:
        require_shift_mixture(q)
        means = torch.stack([c.mean for c in q.components]).clone()
        return cls(means, q.weights.log()-.5*means.square().sum(1))

    def log_q_over_p(self, z: torch.Tensor) -> torch.Tensor:
        if (z.ndim != 2 or z.shape[1] != self.means.shape[1] or z.dtype != torch.float64
                or z.device.type != "cpu" or not torch.isfinite(z).all()):
            raise ValueError("invalid cached shift density coordinates")
        return torch.logsumexp(z @ self.means.T+self.intercepts, 1)


@dataclass(frozen=True)
class CachedVolterraTerminal:
    problem: RBergomiBaselineProblem
    kernel: BLPFFTKernel
    kernel_transform: torch.Tensor
    fft_length: int

    @classmethod
    def create(cls, problem: RBergomiBaselineProblem) -> CachedVolterraTerminal:
        if not isinstance(problem.task, TerminalThresholdTask):
            raise ValueError("cache is terminal downside epsilon=1 only")
        kernel = blp_fft_kernel(problem.simulator(), n_steps=problem.steps,
                                step_dt=problem.step_dt, dtype=torch.float64)
        length = 1 << ((2*problem.steps-1)-1).bit_length()
        return cls(problem, kernel, torch.fft.rfft(kernel.historical_kernel, n=length), length)

    def quantities(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p = self.problem
        p._validate_latent(z, p.local_dimension)
        local = z.reshape(-1, p.steps, 2) @ self.kernel.local_cholesky.T
        dw, integral = local[:, :, 0], local[:, :, 1]
        history = torch.fft.irfft(torch.fft.rfft(dw, n=self.fft_length, dim=1)
                                 * self.kernel_transform, n=self.fft_length, dim=1)[:, :p.steps]
        driver = math.sqrt(2*p.hurst)*(history + integral)
        after = strict_lognormal_variance(p.eta*driver - .5*p.eta**2*self.kernel.volterra_variance[1:], xi=p.xi)
        variance = torch.cat((torch.full((len(z), 1), p.xi, dtype=torch.float64), after), 1)
        increments = -.5*variance[:, :-1]*p.step_dt + torch.sqrt(variance[:, :-1])*(p.rho*dw)
        terminal = math.log(p.spot) + torch.cumsum(increments, 1)[:, -1]
        integrated = p.step_dt*variance[:, :-1].sum(1)
        if not torch.isfinite(terminal).all() or not torch.isfinite(integrated).all():
            raise FloatingPointError("nonfinite cached terminal law")
        return terminal, integrated, variance

    def log_probability(self, z: torch.Tensor) -> torch.Tensor:
        terminal, integrated, _ = self.quantities(z)
        task = self.problem.task
        assert isinstance(task, TerminalThresholdTask)
        return torch.special.log_ndtr((math.log(task.level)-terminal)
                                      / torch.sqrt((1-self.problem.rho**2)*integrated))

    def last_pair(self, prefix: torch.Tensor, q: DefensiveFiniteRankGaussianMixture) -> LastPairCache:
        require_shift_mixture(q)
        if prefix.ndim != 2 or prefix.shape[1] != self.problem.local_dimension-2 or q.dimension != self.problem.local_dimension:
            raise ValueError("invalid last-pair dimensions")
        terminal, integrated, variance = self.quantities(torch.cat((prefix, torch.zeros((len(prefix), 2), dtype=torch.float64)), 1))
        means = torch.stack([c.mean for c in q.components])
        task = self.problem.task
        assert isinstance(task, TerminalThresholdTask)
        return LastPairCache(math.log(task.level)-terminal,
                             self.problem.rho*torch.sqrt(variance[:, -2]*self.problem.step_dt),
                             (1-self.problem.rho**2)*integrated,
                             prefix @ means[:, :-2].T - .5*means.square().sum(1) + q.weights.log(),
                             means[:, -2:])
