"""Price-only matched guide and bounded observed-contribution diagnostics.

No empirical normalization is applied to probability estimators. Ratios in the
diagnostics describe observed weighted features, not unbiased tail certificates.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import torch

from .baselines.rbergomi_common import RBergomiBaselineProblem
from .gaussian_span_marginalization import GaussianMixtureShiftSpec
from .paired_volterra_gradient_reference import PairedVolterraProbability
from .path_functionals import TerminalThresholdTask
from .physical_window_guide import PhysicalWindowGuide

FEATURES = ("log_integrated_variance", "correlated_driver_integral", "log_maximum_left_variance",
            "maximum_left_variance_time", "sampled_component_noise_rms", "maximum_vol_feature_projection")


def price_only_window_guide(problem: RBergomiBaselineProblem, *, amplitudes: list[float],
                            price_amplitudes: list[float], defensive_mass: float = .1) -> PhysicalWindowGuide:
    base = PhysicalWindowGuide.create(problem, kind="point", amplitudes=amplitudes,
                                     price_amplitudes=price_amplitudes, defensive_mass=defensive_mass)
    n = problem.steps
    widths = tuple(min(n//8, n-c) for c in base.centres)
    price = torch.zeros_like(base.price_directions)
    sign = -1. if problem.rho > 0 else 1.
    for row, (c, width) in enumerate(zip(base.centres, widths, strict=True)):
        price[row, 2*torch.arange(c, c+width)] = sign/math.sqrt(width)
    price = price/torch.linalg.vector_norm(price, dim=1, keepdim=True)
    if not torch.equal((base.vol_directions*price).sum(1), torch.zeros(31, dtype=torch.float64)):
        raise ValueError("matched causal orthogonality failed")
    means = torch.stack([torch.zeros(2*n, dtype=torch.float64)]+[
        a*base.vol_directions[i]+b*price[i] for i in range(31) for a in amplitudes for b in price_amplitudes])
    return replace(base, kind="point-vol-window-price", price_window_steps=widths,
                   price_directions=price, spec=GaussianMixtureShiftSpec(means, base.spec.weights.clone()))


def paired_contributions_with_features(pair: PairedVolterraProbability, guide: PhysicalWindowGuide,
                                      z: torch.Tensor, epsilon: torch.Tensor, labels: torch.Tensor
                                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """Two physical payoffs, one full fine density; no extra nonlinear feature calls."""
    pair.fine.problem._validate_latent(z, guide.dimension)
    if labels.shape != (len(z),) or labels.dtype != torch.int64 or labels.device.type != "cpu":
        raise ValueError("CPU integer mixture labels required")
    if (labels < 0).any() or (labels >= guide.spec.components).any():
        raise ValueError("mixture label out of range")
    p = pair.fine.problem
    if guide.problem != p or not isinstance(p.task, TerminalThresholdTask):
        raise ValueError("same finite-grid terminal problem required")
    terminal, integrated, variance = pair.fine.quantities(z)
    fine = torch.special.log_ndtr((math.log(p.task.level)-terminal)/torch.sqrt((1-p.rho**2)*integrated))
    coarse = pair.coarse.log_probability(pair.adapter.coarse(z, epsilon))
    contributions = torch.stack((fine, coarse), 1)-guide.log_q_over_p(z)[:, None]
    local = z.reshape(-1, p.steps, 2) @ pair.fine.kernel.local_cholesky.T
    j = (variance[:, :-1].sqrt()*local[:, :, 0]).sum(1)
    maximum, indices = variance[:, :-1].max(1)
    features = torch.stack((integrated.log(), j, maximum.log(), indices*p.step_dt,
        (z-guide.spec.means[labels]).square().mean(1).sqrt(),
        (z @ guide.vol_directions.T).abs().max(1).values), 1)
    if (not torch.isfinite(contributions).all() or not torch.isfinite(features).all()
            or (contributions > math.log(1/float(guide.spec.weights[0]))+1e-9).any()):
        raise FloatingPointError("contribution/features contract failed")
    return contributions, features


def contribution_diagnostics(log_values: torch.Tensor, labels: torch.Tensor, features: torch.Tensor,
                             *, components: int, top_count: int = 16) -> dict[str, Any]:
    if (log_values.ndim != 1 or len(log_values) < 1 or labels.shape != log_values.shape
            or labels.dtype != torch.int64 or features.shape != (len(log_values), len(FEATURES))
            or log_values.dtype != torch.float64 or features.dtype != torch.float64
            or log_values.device.type != "cpu" or features.device.type != "cpu" or labels.device.type != "cpu"
            or not torch.isfinite(log_values).all() or not torch.isfinite(features).all()
            or isinstance(components, bool) or not isinstance(components, int) or components < 1
            or isinstance(top_count, bool) or not isinstance(top_count, int) or top_count < 1
            or (labels < 0).any() or (labels >= components).any()):
        raise ValueError("invalid observed contribution ledger")
    log_total = float(torch.logsumexp(log_values, 0))
    weights = torch.exp(log_values-log_total)
    label_sums: list[float | None] = [None]*components
    for label in labels.unique().tolist():
        label_sums[label] = float(torch.logsumexp(log_values[labels == label], 0))
    top = torch.argsort(log_values, descending=True, stable=True)[:top_count]
    return {"count": len(log_values), "label_counts": torch.bincount(labels, minlength=components).tolist(),
            "label_log_sums": label_sums, "log_sum": log_total,
            "observed_weighted_feature_means": (weights[:, None]*features).sum(0).tolist(),
            "top": [{"log_contribution": float(log_values[i]), "label": int(labels[i]), "features": features[i].tolist()} for i in top],
            "top_count": top_count, "feature_names": list(FEATURES),
            "scope": "descriptive_observed_contributions_not_unbiased_feature_means_or_tail_coverage"}


def merge_contribution_diagnostics(parts: list[dict[str, Any]]) -> dict[str, Any]:
    if not parts:
        raise ValueError("missing diagnostic parts")
    first = parts[0]
    n = len(first["label_counts"])
    if any(p["feature_names"] != list(FEATURES) or p["top_count"] != first["top_count"]
           or len(p["label_counts"]) != n or len(p["label_log_sums"]) != n for p in parts):
        raise ValueError("changed diagnostic schema")
    totals = torch.tensor([p["log_sum"] for p in parts], dtype=torch.float64)
    log_total = float(torch.logsumexp(totals, 0))
    masses = (totals-log_total).exp()
    feature_means = torch.tensor([p["observed_weighted_feature_means"] for p in parts], dtype=torch.float64)
    label_sums = []
    for k in range(n):
        values = [p["label_log_sums"][k] for p in parts if p["label_log_sums"][k] is not None]
        label_sums.append(float(torch.logsumexp(torch.tensor(values, dtype=torch.float64), 0)) if values else None)
    top = sorted((r for p in parts for r in p["top"]), key=lambda r: r["log_contribution"], reverse=True)[:first["top_count"]]
    return {**first, "count": sum(p["count"] for p in parts),
            "label_counts": [sum(p["label_counts"][k] for p in parts) for k in range(n)],
            "label_log_sums": label_sums, "log_sum": log_total,
            "observed_weighted_feature_means": (masses[:, None]*feature_means).sum(0).tolist(),
            "top": top}
