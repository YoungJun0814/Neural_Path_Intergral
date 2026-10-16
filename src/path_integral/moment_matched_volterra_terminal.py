"""Explicit alternative hybrid grid law; does not replace the frozen BLP law."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

from .baselines.rbergomi_common import RBergomiBaselineProblem
from .cached_volterra_terminal import CachedVolterraTerminal


@dataclass(frozen=True)
class MomentMatchedVolterraTerminal(CachedVolterraTerminal):
    """Positive RMS historical weights, exact local cell and adapted price law.

    Matches the continuous driver's variance at monitoring points. It does not
    match its entire covariance matrix or certify digital weak convergence.
    Every target/proposal trained on the original law requires new validation.
    """

    @classmethod
    def create(cls, problem: RBergomiBaselineProblem) -> MomentMatchedVolterraTerminal:
        base = CachedVolterraTerminal.create(problem)
        h, H, n = problem.step_dt, problem.hurst, problem.steps
        coefficients = torch.zeros(n, dtype=torch.float64)
        if n > 1:
            lower = torch.arange(1, n, dtype=torch.float64)
            # Stable integral of x^(2H-1) on each nonlocal cell.
            energy = lower.pow(2*H)*torch.expm1(2*H*torch.log1p(1/lower))/(2*H)
            coefficients[1:] = torch.sqrt(h**(2*H-1)*energy)
        variance = torch.zeros(n+1, dtype=torch.float64)
        local_variance = float(base.kernel.local_cholesky[1].square().sum())
        variance[1:] = 2*H*(local_variance+h*coefficients.square().cumsum(0))
        analytic = (torch.arange(1, n+1, dtype=torch.float64)*h).pow(2*H)
        torch.testing.assert_close(variance[1:], analytic, rtol=2e-12, atol=1e-14)
        if not torch.isfinite(coefficients).all() or not torch.isfinite(variance).all():
            raise FloatingPointError("nonfinite moment-matched kernel")
        kernel = replace(base.kernel, historical_kernel=coefficients, volterra_variance=variance)
        return cls(problem, kernel, torch.fft.rfft(coefficients, n=base.fft_length), base.fft_length)


def historical_cell_squared_errors(hurst: float, lag: int) -> tuple[float, float]:
    """Unit-step kernel L2 errors (mean weight, RMS weight), excluding local cell."""
    if not math.isfinite(hurst) or not 0 < hurst < .5 or isinstance(lag, bool) or not isinstance(lag, int) or lag < 2:
        raise ValueError("H in (0,.5) and integer nonlocal lag>=2 required")
    lo, power = float(lag-1), hurst+.5
    mean = lo**power*math.expm1(power*math.log1p(1/lo))/power
    energy = lo**(2*hurst)*math.expm1(2*hurst*math.log1p(1/lo))/(2*hurst)
    rms = math.sqrt(energy)
    # Stable algebra: energy-mean^2 and 2*rms*(rms-mean).
    mse = max(0., energy-mean*mean)
    rms_error = 2*rms*mse/(rms+mean)
    return mse, rms_error
