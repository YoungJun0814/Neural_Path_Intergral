"""Executable conservative absolute bound, never an empirical relative certificate."""

from __future__ import annotations

import math

from .blp_driver_error_bound import original_blp_left_driver_mse_bound


def logadd(a: float, b: float) -> float:
    m = max(a, b)
    return m+math.log1p(math.exp(min(a, b)-m))


def conditional_cdf_log_constant(*, eta: float, xi: float, rho: float,
                                  maturity: float, hurst: float) -> float:
    if (any(isinstance(x, bool) or not math.isfinite(x) for x in (eta, xi, rho, maturity, hurst))
            or eta <= 0 or xi <= 0 or maturity <= 0 or not 0 < hurst < .5 or not abs(rho) < 1):
        raise ValueError("positive eta/xi/T, H in (0,.5), |rho|<1 required")
    m = maturity**(2*hurst)
    a = eta**2*m
    log_cosh = a/2+math.log1p(math.exp(-a))-math.log(2)
    log_cv = 2*math.log(xi)+a+logadd(2*math.log(eta), 4*math.log(eta)+math.log(m)+log_cosh)
    log_b1 = -math.log(xi*maturity)+a
    log_b2 = -2*math.log(xi*maturity)+3*a
    log_i = .5*(math.log(maturity)+log_cv)
    log_j = -math.inf if rho == 0 else math.log(.5*abs(rho)*eta)+.5*math.log(xi)
    first = .5*(log_b1-math.log(math.pi*(1-rho**2)))+logadd(math.log(.5)+log_i, log_j)
    second = .5*log_b2-math.log(2)-.5*math.log(math.pi*math.e)+log_i
    result = logadd(first, second)
    if not math.isfinite(result):
        raise FloatingPointError("continuum bound log range unresolved")
    return result


def continuum_bound_diagnostic(*, eta: float, xi: float, rho: float, maturity: float,
                                hurst: float, steps: int, observed_log_mu: float,
                                diagnostic_relative_budget: float = .1) -> dict[str, float | bool | int]:
    if (not math.isfinite(observed_log_mu) or observed_log_mu > 0
            or not 0 < diagnostic_relative_budget < 1):
        raise ValueError("finite observed probability scale and relative budget required")
    bound = original_blp_left_driver_mse_bound(hurst=hurst, maturity=maturity, steps=steps)
    log_c = conditional_cdf_log_constant(eta=eta, xi=xi, rho=rho, maturity=maturity, hurst=hurst)
    log_raw = log_c+.5*math.log(bound.integrated_mse_upper)
    # B(N) = B(1)*N**(-H). This reports the required N scale, not a resource allocation
    # or a certificate relative to an unknown true probability.
    log_target = math.log(diagnostic_relative_budget)+observed_log_mu
    log_n_required = math.log(steps)+(log_raw-log_target)/hurst
    return {"steps": steps, "integrated_driver_mse_upper": bound.integrated_mse_upper,
        "log_absolute_constant": log_c, "log_unclipped_absolute_upper": log_raw,
        "absolute_upper_clipped_at_probability_range": math.exp(min(0., log_raw)),
        "log_bound_over_observed_probability_scale": min(0., log_raw)-observed_log_mu,
        "diagnostic_log10_required_N_using_observed_scale": max(0., log_n_required/math.log(10)),
        "observed_mu_is_not_certified_true_probability_lower_bound": True,
        "relative_bias_certificate_obtained": False, "directed_rounding_interval_certificate": False}
