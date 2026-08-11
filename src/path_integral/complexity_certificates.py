"""Deterministic certificates for scoped V16 complexity statements."""

from __future__ import annotations

import math


def local_newton_error_bound(
    initial_error: float,
    *,
    strong_convexity: float,
    hessian_lipschitz: float,
    iterations: int,
) -> float:
    """Return the classical local-Newton quadratic error upper bound.

    The caller is responsible for verifying that every iterate stays in a convex
    neighborhood where ``H >= strong_convexity I`` and the Hessian has the stated
    Lipschitz constant.  This routine does not infer those assumptions from a
    pointwise Hessian.
    """

    values = (initial_error, strong_convexity, hessian_lipschitz)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Newton-bound inputs must be finite")
    if initial_error < 0.0 or strong_convexity <= 0.0 or hessian_lipschitz < 0.0:
        raise ValueError("Newton-bound scales have the wrong sign")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")
    error = initial_error
    factor = hessian_lipschitz / (2.0 * strong_convexity)
    for _ in range(iterations):
        error = factor * error * error
    return error


def relative_rmse_sample_count(relative_variance: float, tolerance: float) -> int:
    """Minimum ordinary-IS sample count certified by the variance identity."""

    if not math.isfinite(relative_variance) or relative_variance < 0.0:
        raise ValueError("relative_variance must be finite and nonnegative")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    return max(1, math.ceil(relative_variance / tolerance**2))


def amortization_break_even_queries(
    setup_work: float,
    *,
    cold_work_per_query: float,
    amortized_work_per_query: float,
) -> int | None:
    """Return the first integer query count at which setup plus amortized work wins."""

    values = (setup_work, cold_work_per_query, amortized_work_per_query)
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("work values must be finite and nonnegative")
    saving = cold_work_per_query - amortized_work_per_query
    if saving <= 0.0:
        return None
    return max(1, math.ceil(setup_work / saving))
