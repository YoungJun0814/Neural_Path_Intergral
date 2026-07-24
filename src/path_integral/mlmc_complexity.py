"""Fail-closed MLMC complexity algebra with explicit evidence provenance.

This module evaluates the standard bias/variance/cost implication. It does not
prove any supplied rate. In particular, empirical slopes never authorize an
asymptotic complexity statement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, cast

RateEvidence = Literal[
    "proved_internal",
    "proved_external",
    "conditional",
    "empirical",
    "open",
]

_EVIDENCE = {
    "proved_internal",
    "proved_external",
    "conditional",
    "empirical",
    "open",
}


@dataclass(frozen=True)
class MLMCComplexityCertificate:
    """An algebraic complexity result and the provenance that limits its use."""

    weak_bias_exponent: float
    correction_variance_exponent: float
    sample_cost_exponent: float
    sample_cost_log_power: float
    weak_bias_evidence: RateEvidence
    correction_variance_evidence: RateEvidence
    sample_cost_evidence: RateEvidence
    compatibility_threshold: float
    compatibility_condition_satisfied: bool
    regime: str
    epsilon_polynomial_exponent: float | None
    epsilon_log_power: float | None
    finest_sample_polynomial_exponent: float
    finest_sample_log_power: float
    evidence_class: str
    conditional_complexity_authorized: bool
    unconditional_complexity_authorized: bool
    empirical_only: bool


def _validate_evidence(name: str, value: str) -> RateEvidence:
    if value not in _EVIDENCE:
        options = ", ".join(sorted(_EVIDENCE))
        raise ValueError(f"{name} must be one of: {options}")
    return cast(RateEvidence, value)


def _evidence_class(statuses: tuple[RateEvidence, ...]) -> str:
    if "open" in statuses:
        return "open"
    if "empirical" in statuses:
        return "empirical_only"
    if "conditional" in statuses:
        return "conditional"
    return "proved"


def mlmc_complexity_certificate(
    *,
    weak_bias_exponent: float,
    correction_variance_exponent: float,
    sample_cost_exponent: float,
    sample_cost_log_power: float = 0.0,
    weak_bias_evidence: RateEvidence,
    correction_variance_evidence: RateEvidence,
    sample_cost_evidence: RateEvidence,
    comparison_tolerance: float = 1e-12,
) -> MLMCComplexityCertificate:
    r"""Evaluate the MLMC rate implication without promoting unproved premises.

    The premises have the form

    ``bias = O(h**alpha)``, ``variance = O(h**beta)``, and
    ``cost = O(h**(-gamma) log(1/h)**kappa)``.

    The usual compatibility condition is
    ``alpha >= 0.5 * min(beta, gamma)``. For ``kappa >= 0``, the allocation
    calculation gives:

    - ``beta > gamma``: polynomial power 2;
    - ``beta = gamma``: polynomial power 2 and log power ``2 + kappa``;
    - ``beta < gamma``: polynomial power
      ``2 + (gamma-beta)/alpha`` and log power ``kappa``.

    A mandatory finest-level sample is also included. At the boundary
    ``alpha=gamma/2`` in the ``beta>gamma`` regime, its cost contributes the
    otherwise easy-to-miss log power ``kappa``.
    """

    numeric = (
        weak_bias_exponent,
        correction_variance_exponent,
        sample_cost_exponent,
        sample_cost_log_power,
        comparison_tolerance,
    )
    if any(not math.isfinite(value) for value in numeric):
        raise ValueError("MLMC exponents, log power, and tolerance must be finite")
    if (
        weak_bias_exponent <= 0.0
        or correction_variance_exponent <= 0.0
        or sample_cost_exponent <= 0.0
    ):
        raise ValueError("MLMC alpha, beta, and gamma must be strictly positive")
    if sample_cost_log_power < 0.0:
        raise ValueError("sample cost log power must be nonnegative")
    if comparison_tolerance < 0.0:
        raise ValueError("comparison tolerance must be nonnegative")

    bias_status = _validate_evidence("weak_bias_evidence", weak_bias_evidence)
    variance_status = _validate_evidence(
        "correction_variance_evidence", correction_variance_evidence
    )
    cost_status = _validate_evidence("sample_cost_evidence", sample_cost_evidence)
    statuses = (bias_status, variance_status, cost_status)
    evidence_class = _evidence_class(statuses)

    alpha = weak_bias_exponent
    beta = correction_variance_exponent
    gamma = sample_cost_exponent
    kappa = sample_cost_log_power
    compatibility_threshold = 0.5 * min(beta, gamma)
    compatible = alpha + comparison_tolerance >= compatibility_threshold
    finest_polynomial = gamma / alpha
    finest_log = kappa

    difference = beta - gamma
    if difference > comparison_tolerance:
        regime = "beta_greater_gamma"
        allocation_polynomial = 2.0
        allocation_log = 0.0
    elif difference < -comparison_tolerance:
        regime = "beta_less_gamma"
        allocation_polynomial = 2.0 + (gamma - beta) / alpha
        allocation_log = kappa
    else:
        regime = "beta_equal_gamma"
        allocation_polynomial = 2.0
        allocation_log = 2.0 + kappa

    if not compatible:
        polynomial: float | None = None
        log_power: float | None = None
    elif finest_polynomial > allocation_polynomial + comparison_tolerance:
        polynomial = finest_polynomial
        log_power = finest_log
    elif allocation_polynomial > finest_polynomial + comparison_tolerance:
        polynomial = allocation_polynomial
        log_power = allocation_log
    else:
        polynomial = max(allocation_polynomial, finest_polynomial)
        log_power = max(allocation_log, finest_log)

    conditional_authorized = compatible and _evidence_class(statuses) in {
        "proved",
        "conditional",
    }
    unconditional_authorized = compatible and evidence_class == "proved"

    return MLMCComplexityCertificate(
        weak_bias_exponent=alpha,
        correction_variance_exponent=beta,
        sample_cost_exponent=gamma,
        sample_cost_log_power=kappa,
        weak_bias_evidence=bias_status,
        correction_variance_evidence=variance_status,
        sample_cost_evidence=cost_status,
        compatibility_threshold=compatibility_threshold,
        compatibility_condition_satisfied=compatible,
        regime=regime,
        epsilon_polynomial_exponent=polynomial,
        epsilon_log_power=log_power,
        finest_sample_polynomial_exponent=finest_polynomial,
        finest_sample_log_power=finest_log,
        evidence_class=evidence_class,
        conditional_complexity_authorized=conditional_authorized,
        unconditional_complexity_authorized=unconditional_authorized,
        empirical_only=evidence_class == "empirical_only",
    )


def conservative_terminal_rbergomi_complexity(
    hurst: float,
    *,
    epsilon_margin: float,
) -> MLMCComplexityCertificate:
    """Return the V8 terminal rBergomi contract at ``r=H-epsilon``.

    The weak-bias and correction-variance rates remain conditional. This helper
    intentionally has no boolean promotion switch: a future proved-rate promotion
    requires a new, hash-bound review artifact and ledger revision. The FFT cost
    upper bound is an implementation-level premise.
    """

    if not math.isfinite(hurst) or not 0.0 < hurst < 0.5:
        raise ValueError("Hurst parameter must lie in (0, 0.5)")
    if (
        not math.isfinite(epsilon_margin)
        or epsilon_margin <= 0.0
        or epsilon_margin >= hurst
    ):
        raise ValueError("epsilon margin must lie in (0, H)")
    rate = hurst - epsilon_margin
    return mlmc_complexity_certificate(
        weak_bias_exponent=rate,
        correction_variance_exponent=2.0 * rate,
        sample_cost_exponent=1.0,
        sample_cost_log_power=1.0,
        weak_bias_evidence="conditional",
        correction_variance_evidence="conditional",
        sample_cost_evidence="proved_internal",
    )
