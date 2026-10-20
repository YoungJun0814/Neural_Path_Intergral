"""Matrix-free Gaussian driver marginalization for the original finite-grid law.

Analytic formulas are exact; float64 PCG solves carry residual-based numerical
sensitivity bounds, not directed-rounding certificates. Probability only, not
the original proposal's M2. No gradients through iterative solves are supported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from src.physics_engine import strict_lognormal_variance

from .baselines.rbergomi_common import RBergomiBaselineProblem
from .cached_volterra_terminal import CachedVolterraTerminal
from .gaussian_span_marginalization import GaussianMixtureShiftSpec
from .path_functionals import TerminalThresholdTask


@dataclass(frozen=True)
class DriverSolve:
    solution: torch.Tensor
    residual_norm: torch.Tensor
    error_norm_upper: torch.Tensor
    iterations: int
    covariance_row_evaluations: int


@dataclass(frozen=True)
class FFTDriverCovariance:
    problem: RBergomiBaselineProblem
    coefficients: torch.Tensor
    transform: torch.Tensor
    fft_length: int
    independent_variance: float
    diagonal: torch.Tensor

    @classmethod
    def create(cls, problem: RBergomiBaselineProblem) -> FFTDriverCovariance:
        if not 2 <= problem.steps <= 4096 or not isinstance(problem.task, TerminalThresholdTask):
            raise ValueError("terminal original BLP grid 2..4096 required")
        cache = CachedVolterraTerminal.create(problem)
        kernel, n = cache.kernel, problem.steps-1
        b = math.sqrt(2*problem.hurst)*kernel.local_cholesky[0, 0]*kernel.historical_kernel[:n].clone()
        b[0] = math.sqrt(2*problem.hurst)*kernel.local_cholesky[1, 0]
        sigma2 = float(2*problem.hurst*kernel.local_cholesky[1, 1]**2)
        diagonal = b.square().cumsum(0)+sigma2
        if not math.isfinite(sigma2) or sigma2 <= 0:
            raise FloatingPointError("strict Gaussian covariance lower bound lost")
        torch.testing.assert_close(diagonal, kernel.volterra_variance[1:problem.steps], rtol=2e-12, atol=1e-14)
        length = 1 << (2*n-2).bit_length()
        return cls(problem, b, torch.fft.rfft(b, n=length), length, sigma2, diagonal)

    @property
    def dimension(self) -> int:
        return self.problem.steps-1

    def _validate(self, x: torch.Tensor) -> None:
        if (x.ndim != 2 or x.shape[1] != self.dimension or len(x) < 1 or x.dtype != torch.float64
                or x.device.type != "cpu" or not torch.isfinite(x).all() or x.requires_grad):
            raise ValueError("finite non-autograd CPU float64 driver rows required")

    def lower(self, x: torch.Tensor, *, transpose: bool = False) -> torch.Tensor:
        self._validate(x)
        v = x.flip(1) if transpose else x
        out = torch.fft.irfft(torch.fft.rfft(v, n=self.fft_length, dim=1)*self.transform,
                              n=self.fft_length, dim=1)[:, :self.dimension]
        return out.flip(1) if transpose else out

    def covariance(self, x: torch.Tensor) -> torch.Tensor:
        return self.lower(self.lower(x, transpose=True))+self.independent_variance*x

    def project(self, original: torch.Tensor) -> torch.Tensor:
        self.problem._validate_latent(original, self.problem.local_dimension)
        if original.requires_grad:
            raise ValueError("non-autograd probability reference only")
        return self.lower(original[:, :2*self.dimension:2])+math.sqrt(self.independent_variance)*original[:, 1:2*self.dimension:2]

    def solve(self, rhs: torch.Tensor, *, rtol: float = 2e-13, atol: float = 2e-14,
              maximum_iterations: int = 2048) -> DriverSolve:
        self._validate(rhs)
        if (not math.isfinite(rtol) or not 0 < rtol < 1 or not math.isfinite(atol) or atol <= 0
                or isinstance(maximum_iterations, bool) or not isinstance(maximum_iterations, int) or maximum_iterations < 1):
            raise ValueError("valid fixed solver tolerances/cap required")
        threshold = atol+rtol*torch.linalg.vector_norm(rhs, dim=1)
        x = torch.zeros_like(rhs)
        r = rhs.clone()
        z = r/self.diagonal
        direction = z.clone()
        rz = (r*z).sum(1)
        covariance_rows = 0
        for iteration in range(maximum_iterations+1):
            recursive_norm = torch.linalg.vector_norm(r, dim=1)
            if (recursive_norm <= threshold).all() or iteration == maximum_iterations:
                residual = rhs-self.covariance(x)
                covariance_rows += len(rhs)
                norms = torch.linalg.vector_norm(residual, dim=1)
                if (norms <= threshold).all():
                    return DriverSolve(x, norms, norms/self.independent_variance, iteration, covariance_rows)
                if iteration == maximum_iterations:
                    raise RuntimeError("PCG actual residual tolerance not reached; no sample dropped")
                r, z = residual, residual/self.diagonal
                direction, rz = z.clone(), (r*z).sum(1)
                recursive_norm = norms
            active = recursive_norm > threshold
            action = self.covariance(direction)
            covariance_rows += len(rhs)
            denominator = (direction*action).sum(1)
            if (denominator[active] <= 0).any() or not torch.isfinite(denominator).all():
                raise FloatingPointError("PCG positive curvature lost")
            step = torch.zeros_like(rz)
            step[active] = rz[active]/denominator[active]
            x = x+step[:, None]*direction
            r = r-step[:, None]*action
            new_z = r/self.diagonal
            new_rz = (r*new_z).sum(1)
            beta = torch.zeros_like(rz)
            beta[active] = new_rz[active]/rz[active]
            direction = new_z+beta[:, None]*direction
            rz = new_rz
        raise AssertionError("unreachable PCG termination")


@dataclass(frozen=True)
class DriverMarginalEvaluation:
    log_contribution: torch.Tensor
    log_probability: torch.Tensor
    log_density_ratio: torch.Tensor
    conditional_mean: torch.Tensor
    conditional_variance: torch.Tensor
    log_contribution_error_upper: torch.Tensor
    solve_iterations: tuple[int, int]
    covariance_row_evaluations: int


@dataclass(frozen=True)
class FFTDriverMarginalReference:
    operator: FFTDriverCovariance
    spec: GaussianMixtureShiftSpec
    projected_means: torch.Tensor
    precision_means: torch.Tensor
    quadratic_means: torch.Tensor
    mean_solve_error: torch.Tensor
    preparation_iterations: int
    preparation_covariance_rows: int

    @classmethod
    def create(cls, problem: RBergomiBaselineProblem, spec: GaussianMixtureShiftSpec) -> FFTDriverMarginalReference:
        if spec.dimension != problem.local_dimension or spec.means.dtype != torch.float64 or spec.means.device.type != "cpu":
            raise ValueError("same original CPU float64 shift dictionary required")
        natural = spec.natural_component_weight()
        if natural is None or natural < .1-1e-12:
            raise ValueError("at least ten percent defensive natural mass required")
        op = FFTDriverCovariance.create(problem)
        means = op.project(spec.means)
        solve = op.solve(means)
        quadratic = (means*solve.solution).sum(1)
        error = torch.linalg.vector_norm(means, dim=1)*solve.error_norm_upper
        if (quadratic < -error-1e-10).any() or (quadratic > spec.means.square().sum(1)+error+1e-9).any():
            raise FloatingPointError("marginal Gaussian projection energy failed")
        return cls(op, spec, means, solve.solution, quadratic, solve.error_norm_upper, solve.iterations,
                   solve.covariance_row_evaluations)

    def evaluate(self, driver: torch.Tensor, *, maximum_log_error: float = 1e-4) -> DriverMarginalEvaluation:
        op, p = self.operator, self.operator.problem
        op._validate(driver)
        if not math.isfinite(maximum_log_error) or not 0 < maximum_log_error < 1:
            raise ValueError("fixed positive numerical log-error budget required")
        after = strict_lognormal_variance(p.eta*driver-.5*p.eta**2*op.diagonal, xi=p.xi)
        left = torch.cat((torch.full((len(driver), 1), p.xi, dtype=torch.float64), after), 1)
        roots, integrated = left.sqrt(), p.step_dt*left.sum(1)
        a = op.lower(roots[:, :-1])
        sy, sa = op.solve(driver), op.solve(a)
        mean = math.log(p.spot)-.5*integrated+p.rho*math.sqrt(p.step_dt)*(a*sy.solution).sum(1)
        variance = integrated-p.rho**2*p.step_dt*(a*sa.solution).sum(1)
        norm_a = torch.linalg.vector_norm(a, dim=1)
        em = abs(p.rho)*math.sqrt(p.step_dt)*norm_a*sy.error_norm_upper
        ev = p.rho**2*p.step_dt*norm_a*sa.error_norm_upper
        if (not torch.isfinite(variance).all() or not torch.isfinite(mean).all() or (variance-ev <= 0).any()
                or (variance+ev < (1-p.rho**2)*integrated*(1-1e-10)).any()):
            raise FloatingPointError("conditional variance positivity/lower-bound failed")
        task = p.task
        assert isinstance(task, TerminalThresholdTask)
        numerator = math.log(task.level)-mean
        log_g = torch.special.log_ndtr(numerator/variance.sqrt())
        corners = torch.stack([(numerator+sm*em)/(variance+sv*ev).sqrt() for sm in (-1, 1) for sv in (-1, 1)], 1)
        low = torch.special.log_ndtr(corners.min(1).values)
        high = torch.special.log_ndtr(corners.max(1).values)
        payoff_error = torch.maximum((log_g-low).abs(), (high-log_g).abs())
        component_logs = driver @ self.precision_means.T-.5*self.quadratic_means
        log_d = torch.logsumexp(component_logs+self.spec.weights.log(), 1)
        density_error = ((torch.linalg.vector_norm(driver, dim=1)[:, None]
                          +.5*torch.linalg.vector_norm(self.projected_means, dim=1)[None])*self.mean_solve_error).max(1).values
        error = payoff_error+density_error
        if (not torch.isfinite(error).all() or not torch.isfinite(log_g).all() or not torch.isfinite(log_d).all()
                or (error > maximum_log_error).any()):
            raise FloatingPointError("numerical contribution budget failed; no sample clipped or discarded")
        return DriverMarginalEvaluation(log_g-log_d, log_g, log_d, mean, variance, error,
                                        (sy.iterations, sa.iterations),
                                        sy.covariance_row_evaluations+sa.covariance_row_evaluations)
