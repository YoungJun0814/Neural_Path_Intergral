"""Bounded conditional-risk G development; fixed q, exact proposal densities."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from scipy.special import logsumexp

from .finite_rank_gaussian_transport import (
    DefensiveFiniteRankGaussianMixture,
    FiniteRankGaussianComponent,
)
from .gaussian_span_marginalization import GaussianMixtureShiftSpec
from .stratified_paired_risk import estimate, sufficient
from .weighted_bank_mixture import fit_weighted_bank_mixture

EXCLUDED = (61, 63)
ACTIVE = tuple(i for i in range(64) if i not in EXCLUDED)


def risk_training_logs(log_m2: torch.Tensor, log_dq: torch.Tensor,
                       log_ds: torch.Tensor) -> torch.Tensor:
    """Ordinary IS fitting masses for f=m2/Dq; not m1 squared or raw g."""
    _vectors(log_m2, log_dq, log_ds)
    return log_m2-log_dq-log_ds


def selection_logs(log_m2: torch.Tensor, log_dq: torch.Tensor,
                   log_ds: torch.Tensor, log_dg: torch.Tensor) -> torch.Tensor:
    """IS estimate of E_p[f^2/DG] on an independent S bank."""
    _vectors(log_m2, log_dq, log_ds, log_dg)
    return 2*(log_m2-log_dq)-log_ds-log_dg


def _vectors(*values: torch.Tensor) -> None:
    if (not values or values[0].ndim != 1 or len(values[0]) < 2
            or any(t.shape != values[0].shape or t.dtype != torch.float64 or t.device.type != "cpu"
                   or t.requires_grad or not torch.isfinite(t).all() for t in values)):
        raise ValueError("matching detached finite CPU float64 log vectors required")


def moment_error(row: dict[str, Any]) -> dict[str, Any]:
    """Retain a log absolute proxy even when a tiny moment fails relative resolution."""
    m = row["moments"][1]
    value = m["log_estimate"]
    orders = m["order_log_estimates"]

    def difference(a: float, b: float) -> float:
        return -math.inf if a == b else max(a, b)+math.log(-math.expm1(min(a, b)-max(a, b)))

    error = float(logsumexp([value+math.log(1e-12), m["log_tail_upper"],
        difference(orders[2], orders[1]), difference(orders[1], orders[0])]))
    m1 = row["moments"][0]["log_estimate"]
    structural = (not row["critical"]["near_multiple_root"]
        and all(x["inside_numerical_envelope"] for x in row["moments"])
        and 2*m1-1e-8 <= value <= m1+1e-8)
    if not all(math.isfinite(x) for x in (value, error, m1)):
        raise FloatingPointError("finite moment and error proxy required")
    return {"log_m2": value, "log_error_proxy": error,
        "structural_numerics_pass": bool(structural),
        "individual_relative_resolved": bool(row["relative_numerically_resolved"]),
        "not_rigorous_bias_bound": True}


def aggregate_error(log_numerator: torch.Tensor, log_error: torch.Tensor,
                    log_denominator: torch.Tensor, *, squared: bool = False) -> float:
    _vectors(log_numerator, log_error, log_denominator)
    if squared:
        errors = torch.logaddexp(math.log(2)+log_numerator+log_error, 2*log_error)
        values = 2*log_numerator
    else:
        errors, values = log_error, log_numerator
    relative_log = float(torch.logsumexp(errors-log_denominator, 0)-torch.logsumexp(values-log_denominator, 0))
    return math.exp(min(690., relative_log))


def training_gate(log_weights: torch.Tensor, numerical_proxy: float,
                  structural_pass: bool, *, blocks: int = 8) -> dict[str, Any]:
    _vectors(log_weights)
    if type(blocks) is not int or blocks < 2 or len(log_weights) % blocks:
        raise ValueError("complete equal-sized independent training blocks required")
    if not math.isfinite(numerical_proxy) or numerical_proxy < 0:
        raise ValueError("finite nonnegative aggregate numerical proxy required")
    w = log_weights.softmax(0)
    block = w.reshape(blocks, -1).sum(1)
    ess, maximum, max_block = float(1/w.square().sum()), float(w.max()), float(block.max())
    passed = ess >= 32 and maximum <= .1 and max_block <= .5 and structural_pass and numerical_proxy <= 1e-6
    return {"count": len(w), "ess": ess, "maximum_weight_fraction": maximum,
        "block_fractions": block.tolist(), "maximum_block_fraction": max_block,
        "aggregate_numerical_error_proxy": numerical_proxy, "structural_pass": structural_pass,
        "pass": bool(passed), "not_tail_coverage_certificate": True}


def fit_marginal(samples: torch.Tensor, log_weights: torch.Tensor) -> tuple[DefensiveFiniteRankGaussianMixture, dict[str, Any]]:
    _vectors(log_weights)
    if (samples.shape != (len(log_weights), 64) or samples.dtype != torch.float64
            or samples.device.type != "cpu" or samples.requires_grad or not torch.isfinite(samples).all()):
        raise ValueError("finite detached full N32 samples required")
    fit = fit_weighted_bank_mixture(samples[:, ACTIVE], log_weights.softmax(0), clusters=2,
        covariance_rank=4, defensive_mass=.1, covariance_shrinkage=.5,
        minimum_variance=.25, maximum_variance=4., minimum_cluster_mass=.01, feature_rank=4)
    components = []
    for c in fit.proposal.components:
        mean = torch.zeros(64, dtype=torch.float64)
        directions = torch.zeros((64, c.rank), dtype=torch.float64)
        mean[list(ACTIVE)] = c.mean
        directions[list(ACTIVE)] = c.directions
        components.append(FiniteRankGaussianComponent(mean, directions, c.variance_eigenvalues.clone()))
    full = DefensiveFiniteRankGaussianMixture(tuple(components), fit.proposal.weights.clone())
    require_factorization(full)
    return full, {"cluster_masses_before_pruning_renormalization": list(fit.cluster_masses),
        "retained_clusters": len(components)-1, "weighted_bank_ess": fit.weighted_bank_ess,
        "active_coordinates": list(ACTIVE), "excluded_coordinates": list(EXCLUDED),
        "normalized_weights_used_for_fitting_only": True}


def require_factorization(proposal: DefensiveFiniteRankGaussianMixture) -> None:
    if (proposal.dimension != 64 or proposal.defensive_mass < .1-1e-12
            or any(not torch.equal(c.mean[list(EXCLUDED)], torch.zeros(2, dtype=torch.float64))
                   or bool((c.directions[list(EXCLUDED)] != 0).any()) for c in proposal.components)):
        raise ValueError("exact standard independent residual61/unused63 and natural floor required")


def blend(q: GaussianMixtureShiftSpec, fitted: DefensiveFiniteRankGaussianMixture | None,
          alpha: float) -> DefensiveFiniteRankGaussianMixture:
    if not math.isfinite(alpha) or alpha not in (0., .25, .5):
        raise ValueError("frozen alpha grid required")
    components = tuple(FiniteRankGaussianComponent(m.clone(), torch.empty((64, 0), dtype=torch.float64),
        torch.empty(0, dtype=torch.float64)) for m in q.means)
    base = DefensiveFiniteRankGaussianMixture(components, q.weights.clone())
    require_factorization(base)
    if alpha == 0:
        return base
    if fitted is None:
        raise ValueError("fitted marginal proposal required for nonzero alpha")
    require_factorization(fitted)
    result = DefensiveFiniteRankGaussianMixture(base.components+fitted.components,
        torch.cat(((1-alpha)*base.weights, alpha*fitted.weights)))
    require_factorization(result)
    return result


def select_alpha(objectives: list[float]) -> float:
    if len(objectives) != 3 or not all(math.isfinite(x) for x in objectives):
        raise ValueError("three finite ordinary log second moments required")
    minimum = min(objectives)
    return (0., .25, .5)[next(i for i, x in enumerate(objectives) if x <= minimum+1e-12)]


def precision(log_values: torch.Tensor, *, blocks: int = 8,
              numerical_proxy: float = 0.) -> dict[str, Any]:
    _vectors(log_values)
    if (type(blocks) is not int or blocks < 2 or len(log_values) % blocks
            or len(log_values)//blocks < 2 or not math.isfinite(numerical_proxy) or numerical_proxy < 0):
        raise ValueError("complete fixed blocks and finite nonnegative numerical proxy required")
    raw = estimate([sufficient(log_values[:, None])], torch.ones(1, dtype=torch.float64))
    means = log_values.reshape(blocks, -1).logsumexp(1)-math.log(len(log_values)//blocks)
    blocked = estimate([sufficient(means[:, None])], torch.ones(1, dtype=torch.float64))
    scaled = (means-means.max()).exp()
    loo = float(((scaled.sum()-scaled)/(blocks-1)/(scaled.sum()/blocks)-1).abs().max())
    passed = (raw["relative_se"][0] <= .2 and blocked["relative_se"][0] <= .2
        and raw["maximum_fraction"][0] <= .1 and loo <= .15 and numerical_proxy <= 1e-6)
    return {"log_mean": raw["log_mean"][0], "raw_rse": raw["relative_se"][0],
        "block_rse": blocked["relative_se"][0], "maximum_fraction": raw["maximum_fraction"][0],
        "loo_relative_shift": loo, "numerical_error_proxy": numerical_proxy,
        "qualified": bool(passed), "not_finite_sample_tail_certificate": True}


def consistent(a: dict[str, Any], b: dict[str, Any]) -> bool:
    scale = max(a["log_mean"], b["log_mean"])
    x, y = math.exp(a["log_mean"]-scale), math.exp(b["log_mean"]-scale)
    se = math.hypot(x*max(a["raw_rse"], a["block_rse"]), y*max(b["raw_rse"], b["block_rse"]))
    return abs(x-y) <= 3.291*se+1e-12*max(x, y)


def json_compatible(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: json_compatible(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_compatible(v) for v in value]
    return value
