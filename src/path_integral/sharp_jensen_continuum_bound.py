"""Exact time-average Gaussian variance in Jensen inverse-moment bounds.

This supplements, never mutates, the frozen conservative bound. Float64 output
is not a directed-rounding certificate; observed probability is not a lower bound.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from src.physics_engine import RBergomiSimulator

from .blp_driver_error_bound import original_blp_left_driver_mse_bound
from .continuum_probability_bound import conditional_cdf_log_constant, logadd
from .rbergomi_fft import blp_fft_kernel
from .tail_weighted_continuum_bound import log_cdf_weighted_envelope


@dataclass(frozen=True)
class AverageDriverMoments:
    continuous_mean_point_variance: float
    continuous_time_average_variance: float
    discrete_mean_point_variance: float
    discrete_time_average_variance: float


def average_driver_moments(*, hurst: float, maturity: float, steps: int) -> AverageDriverMoments:
    if (isinstance(hurst, bool) or not math.isfinite(hurst) or not 0 < hurst < .5
            or isinstance(maturity, bool) or not math.isfinite(maturity) or maturity <= 0
            or type(steps) is not int or steps < 1):
        raise ValueError("original BLP H,T,N domain required")
    maximum = maturity**(2*hurst)
    continuous = hurst*maximum/((hurst+.5)**2*(hurst+1))
    kernel = blp_fft_kernel(RBergomiSimulator(H=hurst, device="cpu"),
                            n_steps=steps, step_dt=maturity/steps, dtype=torch.float64)
    if steps == 1:
        discrete = 0.
    else:
        sums = torch.cat((torch.zeros(1, dtype=torch.float64), kernel.historical_kernel[1:steps-1].cumsum(0)))
        coefficients = kernel.local_cholesky[1, 0]+kernel.local_cholesky[0, 0]*sums
        discrete = float(2*hurst/steps**2*(coefficients.square().sum()
                        +(steps-1)*kernel.local_cholesky[1, 1]**2))
    return AverageDriverMoments(maximum/(2*hurst+1), continuous,
        float(kernel.volterra_variance[:steps].mean()), discrete)


def inverse_moment_log_upper(m: AverageDriverMoments, *, order: float, eta: float,
                              xi: float, maturity: float) -> float:
    if (any(not math.isfinite(x) or x <= 0 for x in (order, eta, xi, maturity))
            or any(not math.isfinite(x) or x < 0 for x in m.__dict__.values())):
        raise ValueError("positive inverse-moment parameters and Gaussian variances required")
    values = [order*pv+order**2*av for pv, av in (
        (m.continuous_mean_point_variance, m.continuous_time_average_variance),
        (m.discrete_mean_point_variance, m.discrete_time_average_variance))]
    return -order*math.log(xi*maturity)+.5*eta**2*max(values)


def sharp_jensen_bounds(*, eta: float, xi: float, rho: float, maturity: float,
                         hurst: float, steps: int, holder_r: float,
                         moments: AverageDriverMoments | None = None) -> dict[str, float | bool | int]:
    conditional_cdf_log_constant(eta=eta, xi=xi, rho=rho, maturity=maturity, hurst=hurst)
    if isinstance(holder_r, bool) or not math.isfinite(holder_r) or not 1 < holder_r <= 32:
        raise ValueError("r in (1,32] required")
    m = average_driver_moments(hurst=hurst, maturity=maturity, steps=steps) if moments is None else moments
    def inv(s):
        return inverse_moment_log_upper(m, order=s, eta=eta, xi=xi, maturity=maturity)
    a = eta**2*maturity**(2*hurst)
    log_cosh = a/2+math.log1p(math.exp(-a))-math.log(2)
    log_cv = 2*math.log(xi)+a+logadd(2*math.log(eta), 4*math.log(eta)+2*hurst*math.log(maturity)+log_cosh)
    mse = original_blp_left_driver_mse_bound(hurst=hurst, maturity=maturity, steps=steps).integrated_mse_upper
    si = .5*(math.log(maturity)+log_cv+math.log(mse))
    sj = math.log(.5*eta)+.5*(math.log(xi)+math.log(mse))
    c = 1-rho*rho
    rho_sj = -math.inf if rho == 0 else math.log(abs(rho))+sj
    # Pointwise-in-theta derivative bounds use convexity of x**(-s), not a
    # pathwise minimum of endpoint I; hence no extra factor two is needed.
    abs_first = .5*(inv(1)-math.log(2*math.pi*c))+logadd(si-math.log(2), rho_sj)
    abs_second = .5*inv(2)-math.log(2)-.5*math.log(2*math.pi*math.e)+si
    absolute = logadd(abs_first, abs_second)
    r, q = holder_r, holder_r/(holder_r-1)
    k0 = log_cdf_weighted_envelope(power=2*q, monomial=0.)/q
    k1 = log_cdf_weighted_envelope(power=2*q, monomial=2*q)/q
    ci = logadd(inv(r)/r+k0-math.log(2*c), inv(2*r)/r+k1-math.log(2))
    cj = inv(r)/r+k0-math.log(c)
    exponent = 1-1/(2*q)
    root = math.log(exponent)+logadd(si+.5*ci, rho_sj+.5*cj)
    if not math.isfinite(absolute) or not math.isfinite(root):
        raise FloatingPointError("sharpened bound numerical range unresolved")
    return {"steps": steps, "holder_r": r, "root_exponent": exponent,
        "log_root_error_upper": root, "log_absolute_error_upper": absolute,
        "absolute_error_upper_clipped_at_probability_range": math.exp(min(0., absolute)),
        "root_error_upper_clipped_at_probability_range": math.exp(min(0., root)),
        "relative_bias_certificate_obtained": False, "directed_rounding_certificate": False}
