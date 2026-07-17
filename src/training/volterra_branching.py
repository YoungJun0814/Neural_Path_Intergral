"""Coarse-measurable boundary-variance allocation for Volterra branching."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from src.path_integral.path_functionals import DownsideExcursionTask
from src.path_integral.rbergomi_branching import RBergomiCoarseTrunks


class CoarseBoundaryVarianceModel(nn.Module):
    """Small allocation score model; it never changes the sampling density."""

    network: nn.Sequential

    def __init__(self, feature_dimension: int, hidden_dimension: int = 24) -> None:
        super().__init__()
        if feature_dimension <= 0 or hidden_dimension <= 0:
            raise ValueError("feature and hidden dimensions must be positive")
        self.network = nn.Sequential(
            nn.Linear(feature_dimension, hidden_dimension),
            nn.SiLU(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.SiLU(),
            nn.Linear(hidden_dimension, 1),
        )

    def forward(self, standardized_features: torch.Tensor) -> torch.Tensor:
        if standardized_features.ndim != 2:
            raise ValueError("standardized_features must have shape (parents, features)")
        return self.network(standardized_features).squeeze(-1)


@dataclass(frozen=True)
class BoundaryVarianceTrainingRecord:
    epoch: int
    loss: float
    accuracy: float
    positive_recall: float


@dataclass(frozen=True)
class BoundaryVarianceFit:
    model: CoarseBoundaryVarianceModel
    feature_mean: torch.Tensor
    feature_scale: torch.Tensor
    target_variance_threshold: float
    positive_fraction: float
    history: tuple[BoundaryVarianceTrainingRecord, ...]


def coarse_variance_features(
    trunks: RBergomiCoarseTrunks,
    task: DownsideExcursionTask,
    *,
    feature_points: int = 9,
) -> torch.Tensor:
    """Create fixed-width features using only the completed coarse path."""
    coarse_steps = trunks.spot.shape[1] - 1
    if feature_points < 2 or feature_points > coarse_steps + 1:
        raise ValueError("feature_points must lie in [2, coarse_steps + 1]")
    indices = (
        torch.linspace(
            0,
            coarse_steps,
            feature_points,
            device=trunks.spot.device,
            dtype=trunks.spot.dtype,
        )
        .round()
        .long()
    )
    running_minimum, occupation, hit = task.prefix_state(trunks.spot, 2.0 * trunks.fine_dt)
    initial_variance = trunks.model_parameters[2]
    sampled_log_spot = torch.log(trunks.spot[:, indices] / trunks.initial_spot)
    sampled_log_variance = torch.log(trunks.variance[:, indices] / initial_variance)
    sampled_volterra = trunks.volterra[:, indices]
    sampled_occupation = occupation[:, indices] / trunks.maturity
    signed_hit_margin = (running_minimum[:, -1] - task.hit_barrier) / task.hit_scale
    signed_occupation_margin = (occupation[:, -1] - task.minimum_occupation) / task.occupation_scale
    log_returns = torch.diff(torch.log(trunks.spot), dim=1)
    time_of_minimum = torch.argmin(trunks.spot, dim=1).to(trunks.spot.dtype) / coarse_steps
    scalar = torch.stack(
        (
            signed_hit_margin,
            signed_occupation_margin,
            torch.abs(signed_hit_margin),
            torch.abs(signed_occupation_margin),
            hit[:, -1].to(trunks.spot.dtype),
            (occupation[:, -1] >= task.minimum_occupation).to(trunks.spot.dtype),
            time_of_minimum,
            torch.sum(log_returns.square(), dim=1),
            torch.log(trunks.spot[:, -1] / trunks.initial_spot),
            torch.log(trunks.variance[:, -1] / initial_variance),
            trunks.volterra[:, -1],
        ),
        dim=1,
    )
    features = torch.cat(
        (
            sampled_log_spot,
            sampled_log_variance,
            sampled_volterra,
            sampled_occupation,
            scalar,
        ),
        dim=1,
    )
    if not torch.isfinite(features).all():
        raise FloatingPointError("coarse branching features became nonfinite")
    return features


def fit_coarse_boundary_variance_model(
    features: torch.Tensor,
    conditional_variances: torch.Tensor,
    *,
    seed: int,
    high_variance_fraction: float = 0.10,
    hidden_dimension: int = 24,
    epochs: int = 120,
    batch_size: int = 512,
    learning_rate: float = 2e-3,
    weight_decay: float = 1e-4,
) -> BoundaryVarianceFit:
    """Fit a classifier for the top conditional-variance coarse trunks."""
    if features.ndim != 2 or conditional_variances.ndim != 1:
        raise ValueError("features and conditional_variances have invalid ranks")
    if features.shape[0] != conditional_variances.shape[0] or features.shape[0] < 2:
        raise ValueError("features and targets must share a nontrivial batch")
    if features.device != conditional_variances.device:
        raise ValueError("features and targets must share a device")
    if not features.is_floating_point() or not conditional_variances.is_floating_point():
        raise TypeError("features and targets must be floating point")
    if not torch.isfinite(features).all() or not torch.isfinite(conditional_variances).all():
        raise ValueError("features and targets must be finite")
    if bool((conditional_variances < 0.0).any()):
        raise ValueError("conditional variances must be nonnegative")
    if not 0.0 < high_variance_fraction < 0.5:
        raise ValueError("high_variance_fraction must lie in (0, 0.5)")
    if hidden_dimension <= 0 or epochs <= 0 or batch_size <= 0:
        raise ValueError("model and training sizes must be positive")
    if learning_rate <= 0.0 or weight_decay < 0.0:
        raise ValueError("optimizer parameters are outside their valid ranges")
    feature_mean = features.mean(dim=0)
    feature_scale = features.std(dim=0, unbiased=True)
    feature_scale = torch.clamp(feature_scale, min=1e-8)
    standardized = (features - feature_mean) / feature_scale
    threshold = torch.quantile(conditional_variances, 1.0 - high_variance_fraction)
    labels = conditional_variances >= threshold
    positives = int(labels.sum())
    negatives = labels.numel() - positives
    if positives == 0 or negatives == 0:
        raise RuntimeError("conditional-variance labels are degenerate")
    positive_weight = torch.tensor(
        negatives / positives, device=features.device, dtype=features.dtype
    )
    torch.manual_seed(seed)
    model = CoarseBoundaryVarianceModel(features.shape[1], hidden_dimension=hidden_dimension).to(
        device=features.device, dtype=features.dtype
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    history: list[BoundaryVarianceTrainingRecord] = []
    generator = torch.Generator(device=features.device).manual_seed(seed + 1)
    for epoch in range(epochs):
        permutation = torch.randperm(features.shape[0], device=features.device, generator=generator)
        for start in range(0, features.shape[0], batch_size):
            selected = permutation[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(standardized[selected])
            loss = loss_fn(logits, labels[selected].to(features.dtype))
            loss.backward()
            optimizer.step()
        if epoch == 0 or (epoch + 1) % max(epochs // 6, 1) == 0 or epoch + 1 == epochs:
            with torch.no_grad():
                logits = model(standardized)
                loss = loss_fn(logits, labels.to(features.dtype))
                prediction = logits >= 0.0
                accuracy = float((prediction == labels).double().mean())
                recall = float((prediction & labels).sum() / labels.sum())
            history.append(
                BoundaryVarianceTrainingRecord(
                    epoch=epoch,
                    loss=float(loss),
                    accuracy=accuracy,
                    positive_recall=recall,
                )
            )
    return BoundaryVarianceFit(
        model=model,
        feature_mean=feature_mean.detach(),
        feature_scale=feature_scale.detach(),
        target_variance_threshold=float(threshold),
        positive_fraction=float(labels.double().mean()),
        history=tuple(history),
    )


def boundary_variance_scores(fit: BoundaryVarianceFit, features: torch.Tensor) -> torch.Tensor:
    """Return a frozen coarse-measurable allocation score."""
    if features.ndim != 2 or features.shape[1] != fit.feature_mean.shape[0]:
        raise ValueError("features are incompatible with the fitted allocator")
    standardized = (features - fit.feature_mean) / fit.feature_scale
    with torch.no_grad():
        return fit.model(standardized)


def score_threshold_branch_counts(
    scores: torch.Tensor,
    *,
    threshold: float,
    high_branches: int,
) -> torch.Tensor:
    """Convert a frozen scalar threshold into positive integer branch counts."""
    if scores.ndim != 1 or not scores.is_floating_point() or not torch.isfinite(scores).all():
        raise ValueError("scores must be a finite floating-point vector")
    if not math.isfinite(threshold) or high_branches < 1:
        raise ValueError("threshold and high_branches are invalid")
    return torch.where(
        scores >= threshold,
        torch.full_like(scores, high_branches, dtype=torch.long),
        torch.ones_like(scores, dtype=torch.long),
    )
