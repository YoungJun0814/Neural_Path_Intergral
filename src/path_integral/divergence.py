"""Log-domain diagnostics linking tilted path laws to estimator variance."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TiltedDivergenceDiagnostics:
    """Empirical diagnostics for nonnegative IS contributions.

    The quantities are plug-in path-sample estimates.  They satisfy the
    empirical identity ``relative_variance = exp(renyi2) - 1`` exactly up to
    floating-point precision, but are not unbiased estimators of the
    population divergences.
    """

    samples: int
    log_normalizer: torch.Tensor
    log_second_moment: torch.Tensor
    log_relative_second_moment: torch.Tensor
    relative_variance: torch.Tensor
    chi_square: torch.Tensor
    renyi2: torch.Tensor
    contribution_ess: torch.Tensor
    contribution_ess_fraction: torch.Tensor


def tilted_divergence_diagnostics(
    log_contributions: torch.Tensor,
) -> TiltedDivergenceDiagnostics:
    r"""Diagnose ``Y=g dM/dQ`` using stable log-domain reductions.

    For the population normalizer ``Z=E_Q[Y]``, the exact identities are

    ``Var_Q(Y)/Z^2 = chi2(Q_star || Q)`` and
    ``log(1 + Var_Q(Y)/Z^2) = D_2(Q_star || Q)``.

    This function computes their empirical analogues with population (``1/N``)
    moments along the one-dimensional path axis.
    """
    if log_contributions.ndim != 1 or log_contributions.numel() < 2:
        raise ValueError("log_contributions must contain at least two paths")
    if not log_contributions.is_floating_point():
        raise TypeError("log_contributions must be a floating-point tensor")
    if not torch.isfinite(log_contributions).all():
        raise ValueError("log_contributions must be finite")

    samples = log_contributions.numel()
    log_samples = math.log(samples)
    log_normalizer = torch.logsumexp(log_contributions, dim=0) - log_samples
    log_second_moment = torch.logsumexp(2.0 * log_contributions, dim=0) - log_samples
    raw_log_relative = log_second_moment - 2.0 * log_normalizer
    # Cauchy--Schwarz guarantees nonnegativity.  Clamp only round-off-sized
    # violations so constant contributions report exactly zero variance.
    log_relative = torch.clamp(raw_log_relative, min=0.0)
    relative_variance = torch.expm1(log_relative)

    log_sum = torch.logsumexp(log_contributions, dim=0)
    log_sum_squares = torch.logsumexp(2.0 * log_contributions, dim=0)
    contribution_ess = torch.exp(2.0 * log_sum - log_sum_squares)

    return TiltedDivergenceDiagnostics(
        samples=samples,
        log_normalizer=log_normalizer,
        log_second_moment=log_second_moment,
        log_relative_second_moment=log_relative,
        relative_variance=relative_variance,
        chi_square=relative_variance,
        renyi2=log_relative,
        contribution_ess=contribution_ess,
        contribution_ess_fraction=contribution_ess / samples,
    )
