"""Development-only Gaussian mixtures fitted to weighted conditional SMC banks.

SMC particles are correlated; their normalized weights are used only to fit a
proposal. Final estimates must use fresh IID draws and the *exact mixture*
density, never the SMC bank as ordinary importance-sampling observations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from src.path_integral.finite_rank_gaussian_transport import (
    DefensiveFiniteRankGaussianMixture,
    FiniteRankGaussianComponent,
)


@dataclass(frozen=True)
class WeightedBankMixtureFit:
    proposal: DefensiveFiniteRankGaussianMixture
    cluster_masses: tuple[float, ...]
    weighted_bank_ess: float
    bank_count: int
    centers: torch.Tensor
    feature_directions: torch.Tensor
    feature_mean: torch.Tensor
    feature_scales: torch.Tensor


def _validate_bank(samples: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if (samples.ndim != 2 or samples.shape[0] < 2 or samples.device.type != "cpu"
            or samples.dtype != torch.float64 or not torch.isfinite(samples).all()
            or weights.shape != (samples.shape[0],) or weights.device.type != "cpu"
            or weights.dtype != torch.float64 or not torch.isfinite(weights).all()
            or bool((weights < 0).any()) or float(weights.sum()) <= 0.0):
        raise ValueError("weighted bank must be finite CPU float64 with positive mass")
    return weights / weights.sum()


def _weighted_quantile(values: torch.Tensor, weights: torch.Tensor, fraction: float) -> float:
    order = torch.argsort(values)
    cumulative = torch.cumsum(weights[order], dim=0)
    index = int(torch.searchsorted(cumulative, torch.tensor(fraction, dtype=torch.float64)))
    return float(values[order[min(index, values.numel() - 1)]])


def _cluster_features(
    samples: torch.Tensor, weights: torch.Tensor, *, feature_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = weights @ samples
    centered = samples - mean
    covariance = centered.T @ (weights[:, None] * centered)
    eigvals, eigvecs = torch.linalg.eigh(covariance)
    order = torch.argsort(eigvals, descending=True)[:feature_rank]
    directions = eigvecs[:, order]
    projected = centered @ directions
    scales = torch.sqrt(torch.clamp(eigvals[order], min=0.1))
    return projected / scales, directions, mean, scales


def _weighted_kmeans(
    features: torch.Tensor, weights: torch.Tensor, *, clusters: int,
    iterations: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    if clusters == 1:
        center = (weights @ features)[None, :]
        return torch.zeros(features.shape[0], dtype=torch.long), center
    first = features[:, 0]
    centers = torch.stack([
        features[torch.argmin(torch.abs(first - _weighted_quantile(
            first, weights, (index + 0.5) / clusters,
        )))].clone()
        for index in range(clusters)
    ])
    assignments = torch.zeros(features.shape[0], dtype=torch.long)
    for _ in range(iterations):
        distances = torch.cdist(features, centers).square()
        assignments = torch.argmin(distances, dim=1)
        updated = []
        for index in range(clusters):
            selected = assignments == index
            mass = float(weights[selected].sum())
            updated.append(
                (weights[selected] @ features[selected]) / mass
                if mass > 0.0 else centers[index]
            )
        centers = torch.stack(updated)
    assignments = torch.argmin(torch.cdist(features, centers).square(), dim=1)
    return assignments, centers


def fit_weighted_bank_mixture(
    samples: torch.Tensor, weights: torch.Tensor, *, clusters: int,
    covariance_rank: int, defensive_mass: float = 0.1,
    covariance_shrinkage: float = 0.5, minimum_variance: float = 0.25,
    maximum_variance: float = 4.0, minimum_cluster_mass: float = 0.01,
    feature_rank: int = 4,
) -> WeightedBankMixtureFit:
    """Fit clustered low-rank Gaussians and keep an exact defensive component."""

    normalized = _validate_bank(samples, weights)
    n, d = samples.shape
    if (clusters < 1 or clusters > n or covariance_rank < 0 or covariance_rank > d
            or not 0.0 < defensive_mass < 1.0
            or not 0.0 <= covariance_shrinkage <= 1.0
            or not 0.0 < minimum_variance <= 1.0 <= maximum_variance
            or not 0.0 <= minimum_cluster_mass < 1.0 / clusters
            or feature_rank < 1):
        raise ValueError("invalid weighted mixture configuration")
    feature_rank = min(feature_rank, d)
    features, feature_directions, feature_mean, feature_scales = _cluster_features(
        samples, normalized, feature_rank=feature_rank,
    )
    assignments, centers = _weighted_kmeans(features, normalized, clusters=clusters)
    component_masses = []
    components = []
    kept_centers = []
    for index in range(clusters):
        selected = assignments == index
        mass = float(normalized[selected].sum())
        if mass < minimum_cluster_mass or int(selected.sum()) < 2:
            continue
        cluster_weights = normalized[selected] / mass
        cluster_samples = samples[selected]
        mean = cluster_weights @ cluster_samples
        centered = cluster_samples - mean
        if covariance_rank:
            covariance = centered.T @ (cluster_weights[:, None] * centered)
            covariance = (1.0 - covariance_shrinkage) * torch.eye(
                d, dtype=torch.float64,
            ) + covariance_shrinkage * covariance
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
            order = torch.argsort(torch.abs(eigenvalues - 1.0), descending=True)[:covariance_rank]
            directions = eigenvectors[:, order]
            variances = torch.clamp(
                eigenvalues[order], minimum_variance, maximum_variance,
            )
        else:
            directions = torch.empty((d, 0), dtype=torch.float64)
            variances = torch.empty(0, dtype=torch.float64)
        components.append(FiniteRankGaussianComponent(mean, directions, variances))
        component_masses.append(mass)
        kept_centers.append(centers[index])
    if not components:
        raise RuntimeError("all weighted clusters were empty or too small")
    masses = torch.tensor(component_masses, dtype=torch.float64)
    masses /= masses.sum()
    proposal = DefensiveFiniteRankGaussianMixture(
        (FiniteRankGaussianComponent.natural(d), *components),
        torch.cat((
            torch.tensor((defensive_mass,), dtype=torch.float64),
            (1.0 - defensive_mass) * masses,
        )),
    )
    bank_ess = 1.0 / float(torch.sum(normalized.square()))
    if not math.isfinite(bank_ess):
        raise FloatingPointError("invalid weighted bank ESS")
    return WeightedBankMixtureFit(
        proposal=proposal,
        cluster_masses=tuple(component_masses),
        weighted_bank_ess=bank_ess,
        bank_count=n,
        centers=torch.stack(kept_centers),
        feature_directions=feature_directions,
        feature_mean=feature_mean,
        feature_scales=feature_scales,
    )


def assign_weighted_bank_clusters(
    samples: torch.Tensor, fit: WeightedBankMixtureFit,
) -> torch.Tensor:
    """Assign new paths to training-only centers; no refitting on held-out data."""

    if samples.ndim != 2 or samples.shape[1] != fit.feature_mean.numel():
        raise ValueError("cluster-assignment dimension mismatch")
    transformed = (
        (samples - fit.feature_mean) @ fit.feature_directions
    ) / fit.feature_scales
    return torch.argmin(torch.cdist(transformed, fit.centers).square(), dim=1)


def assign_saved_cluster_geometry(
    samples: torch.Tensor, geometry: dict[str, Any],
) -> torch.Tensor:
    """Apply fixed training-only centers from a serialized development artifact."""

    mean = torch.tensor(geometry["feature_mean"], dtype=torch.float64)
    directions = torch.tensor(geometry["feature_directions"], dtype=torch.float64)
    scales = torch.tensor(geometry["feature_scales"], dtype=torch.float64)
    centers = torch.tensor(geometry["centers"], dtype=torch.float64)
    if (samples.ndim != 2 or samples.dtype != torch.float64
            or samples.shape[1] != mean.numel()
            or directions.shape != (mean.numel(), scales.numel())
            or centers.ndim != 2 or centers.shape[1] != scales.numel()
            or bool((scales <= 0).any())
            or not all(torch.isfinite(x).all() for x in (mean, directions, scales, centers))):
        raise ValueError("invalid saved cluster geometry")
    transformed = ((samples - mean) @ directions) / scales
    return torch.argmin(torch.cdist(transformed, centers).square(), dim=1)


def proposal_from_parameters(parameters: dict[str, Any]) -> DefensiveFiniteRankGaussianMixture:
    """Reconstruct an exact proposal from a saved, auditable parameter record."""

    raw_components = parameters["components"]
    if not isinstance(raw_components, list) or not raw_components:
        raise ValueError("proposal parameter record has no components")
    components = []
    for raw in raw_components:
        mean = torch.tensor(raw["mean"], dtype=torch.float64)
        eigenvalues = torch.tensor(raw["eigenvalues"], dtype=torch.float64)
        directions = torch.tensor(raw["directions"], dtype=torch.float64).reshape(
            mean.numel(), eigenvalues.numel(),
        )
        components.append(FiniteRankGaussianComponent(mean, directions, eigenvalues))
    return DefensiveFiniteRankGaussianMixture(
        tuple(components), torch.tensor(parameters["weights"], dtype=torch.float64),
    )
