"""Conservative probability-root inequality, not an empirical certificate.

Analytic assumptions and derivation are in the 2026-10-28 theory note. Current
BLP constants can still give a vacuous interval; this is never auto-qualified.
"""

from __future__ import annotations

import math

from .blp_driver_error_bound import original_blp_left_driver_mse_bound
from .continuum_probability_bound import conditional_cdf_log_constant, logadd


def log_gaussian_monomial_max(power: float, gaussian_power: float) -> float:
    if (not math.isfinite(power) or power < 0 or not math.isfinite(gaussian_power)
            or gaussian_power <= 0):
        raise ValueError("nonnegative monomial and positive Gaussian power required")
    exponent = 0. if power == 0 else .5*power*(math.log(power/gaussian_power)-1)
    return -.5*gaussian_power*math.log(2*math.pi)+exponent


def log_cdf_weighted_envelope(*, power: float, monomial: float) -> float:
    """Upper bound sup_s |s|**m phi(s)**p / Phi(s), p>1.

    For x>=0, integrate phi over [x,x+1/(x+1)] to get
    Phi(-x)>=exp(-1.5)*phi(x)/(x+1). This avoids a fitted Mills ratio.
    """
    if not math.isfinite(power) or power <= 1:
        raise ValueError("CDF weighted envelope needs power>1")
    positive = math.log(2)+log_gaussian_monomial_max(monomial, power)
    negative = 1.5+logadd(log_gaussian_monomial_max(monomial+1, power-1),
                          log_gaussian_monomial_max(monomial, power-1))
    return max(positive, negative)


def probability_root_budget(*, eta: float, xi: float, rho: float, maturity: float,
                            hurst: float, steps: int, holder_r: float) -> dict[str, float | bool | int]:
    conditional_cdf_log_constant(eta=eta, xi=xi, rho=rho, maturity=maturity, hurst=hurst)
    if isinstance(holder_r, bool) or not math.isfinite(holder_r) or not 1 < holder_r <= 32:
        raise ValueError("fixed Holder exponent r in (1,32] required")
    r, q = holder_r, holder_r/(holder_r-1)
    a = eta**2*maturity**(2*hurst)
    log_b_r = -r*math.log(xi*maturity)+.5*a*(r+r*r)
    log_b_2r = -2*r*math.log(xi*maturity)+.5*a*(2*r+4*r*r)
    inverse_1 = (math.log(2)+log_b_r)/r
    inverse_2 = (math.log(2)+log_b_2r)/r
    k0 = log_cdf_weighted_envelope(power=2*q, monomial=0.)/q
    k1 = log_cdf_weighted_envelope(power=2*q, monomial=2*q)/q
    log_ci = logadd(inverse_1+k0-math.log(2*(1-rho*rho)), inverse_2+k1-math.log(2))
    log_cj = inverse_1+k0-math.log(1-rho*rho)
    log_cosh = a/2+math.log1p(math.exp(-a))-math.log(2)
    log_cv = 2*math.log(xi)+a+logadd(2*math.log(eta),
            4*math.log(eta)+2*hurst*math.log(maturity)+log_cosh)
    mse = original_blp_left_driver_mse_bound(hurst=hurst, maturity=maturity, steps=steps).integrated_mse_upper
    log_sigma_i = .5*(math.log(maturity)+log_cv+math.log(mse))
    log_sigma_j = math.log(.5*eta)+.5*(math.log(xi)+math.log(mse))
    log_d = logadd(log_sigma_i+.5*log_ci,
                  -math.inf if rho == 0 else math.log(abs(rho))+log_sigma_j+.5*log_cj)
    exponent = 1-1/(2*q)
    log_root_budget = math.log(exponent)+log_d
    if not math.isfinite(log_root_budget):
        raise FloatingPointError("probability root budget unresolved numeric range")
    return {"steps": steps, "holder_r": r, "holder_q": q, "root_exponent": exponent,
        "log_root_error_upper": log_root_budget,
        "root_error_upper_clipped_at_probability_range": math.exp(min(0., log_root_budget)),
        "directed_rounding_certificate": False,
        "relative_bias_certificate_obtained": False}


def observed_probability_root_diagnostic(observed_log_mu: float, *, exponent: float,
                                         log_root_budget: float) -> dict[str, float | bool]:
    if (not math.isfinite(observed_log_mu) or observed_log_mu > 0 or not .5 < exponent < 1
            or not math.isfinite(log_root_budget)):
        raise ValueError("finite observed scale and valid root bound required")
    center = math.exp(exponent*observed_log_mu)
    budget = math.exp(min(0., log_root_budget))
    lower = max(0., center-budget)**(1/exponent)
    upper = min(1., center+budget)**(1/exponent)
    return {"plug_in_lower": lower, "plug_in_upper": upper,
        "observed_grid_mean_is_not_certified": True,
        "relative_bias_certificate_obtained": False}
