"""Preserve the parent's identity-covariance mixture instead of compressing modes."""

from __future__ import annotations

import math

import torch

from src.path_integral.finite_rank_gaussian_transport import (
    DefensiveFiniteRankGaussianMixture,
    FiniteRankGaussianComponent,
)
from src.path_integral.r1_bottleneck_diagnostics import _check_bank


def refit_identity_mixture(
    parent: DefensiveFiniteRankGaussianMixture,
    samples: torch.Tensor, log_g: torch.Tensor, log_p_over_bank: torch.Tensor,
    *, steps: int = 80, learning_rate: float = 0.04, maximum_norm: float = 20,
) -> tuple[DefensiveFiniteRankGaussianMixture, dict[str, float]]:
    """Weighted KL fitting, same number of learned components, fixed delta.

    Natural components are aggregated; learned means and their relative weights
    are optimized. Covariance learning and component deletion are not performed.
    The best empirical iterate (including parent) is returned, not a final loss
    mislabelled as improvement. This is fitting, never an unbiased IS estimate.
    """
    _check_bank(samples, log_g, log_p_over_bank)
    if (samples.shape[1] != parent.dimension or steps < 1
            or not math.isfinite(learning_rate) or learning_rate <= 0
            or not math.isfinite(maximum_norm) or maximum_norm <= 0
            or any(c.rank != 0 for c in parent.components)):
        raise ValueError("refit currently supports identity-covariance parents only")
    indexes = [i for i, c in enumerate(parent.components) if not c.is_natural()]
    delta = parent.defensive_mass
    if not indexes or not 0 < delta < 1:
        raise ValueError("parent must contain natural and learned components")
    initial = torch.stack([parent.components[i].mean for i in indexes])
    if bool((torch.linalg.vector_norm(initial, dim=1) > maximum_norm).any()):
        raise ValueError("initial parent exceeds declared mean constraint")
    means = initial.detach().clone().requires_grad_(True)
    logits = torch.log(parent.weights[indexes] / (1 - delta)).detach().requires_grad_(True)
    target = torch.softmax(log_g + log_p_over_bank, dim=0).detach()

    def loss_value() -> torch.Tensor:
        shifted = samples @ means.T - 0.5 * means.square().sum(dim=1)
        learned = torch.logsumexp(shifted + torch.log_softmax(logits, dim=0), dim=1)
        ratio = torch.logaddexp(torch.full_like(learned, math.log(delta)),
                                math.log1p(-delta) + learned)
        return -(target * ratio).sum()

    initial_loss = float(loss_value().detach())
    best_loss, best_means, best_logits = initial_loss, means.detach().clone(), logits.detach().clone()
    optimizer = torch.optim.Adam((means, logits), lr=learning_rate)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_value()
        loss.backward()
        if means.grad is None or logits.grad is None or not torch.isfinite(
                means.grad).all() or not torch.isfinite(logits.grad).all():
            raise FloatingPointError("nonfinite mixture gradient")
        optimizer.step()
        with torch.no_grad():
            norms = torch.linalg.vector_norm(means, dim=1, keepdim=True)
            means.mul_(torch.clamp(maximum_norm / norms.clamp_min(1e-300), max=1))
            value = float(loss_value())
            if not math.isfinite(value):
                raise FloatingPointError("nonfinite mixture fit")
            if value < best_loss:
                best_loss, best_means, best_logits = value, means.clone(), logits.clone()
    components = (FiniteRankGaussianComponent.natural(parent.dimension),) + tuple(
        FiniteRankGaussianComponent(m, torch.empty((parent.dimension, 0), dtype=torch.float64),
                                    torch.empty(0, dtype=torch.float64)) for m in best_means
    )
    weights = torch.cat((torch.tensor([delta], dtype=torch.float64),
                         (1 - delta) * torch.softmax(best_logits, dim=0)))
    return DefensiveFiniteRankGaussianMixture(components, weights), {
        "initial_empirical_kl_loss": initial_loss, "returned_empirical_kl_loss": best_loss,
        "target_weight_ess": float(1 / target.square().sum()),
    }
