"""Exact finite-grid density primitives for mixtures of Brownian drift proposals."""

from __future__ import annotations

import math

import torch


def _validated_weights(
    weights: torch.Tensor,
    *,
    components: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(weights, torch.Tensor):
        raise TypeError("weights must be a torch.Tensor")
    if weights.ndim != 1 or weights.shape[0] != components:
        raise ValueError(f"weights must have shape ({components},)")
    if not weights.is_floating_point():
        raise TypeError("weights must be floating point")
    if not torch.isfinite(weights).all() or bool((weights <= 0.0).any()):
        raise ValueError("mixture weights must be finite and strictly positive")
    converted = weights.to(device=device, dtype=dtype)
    total = converted.sum()
    tolerance = 64.0 * torch.finfo(dtype).eps
    if not bool(torch.abs(total - 1.0) <= tolerance):
        raise ValueError("mixture weights must sum to one")
    return converted


def all_expert_log_q_over_p(
    controls: torch.Tensor,
    target_increments: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Evaluate every expert density on fixed target-coordinate paths.

    ``controls`` has shape ``(batch, experts, steps, drivers)`` and contains
    controls causally replayed on the target path. ``target_increments`` has
    shape ``(batch, steps, drivers)``. The result has shape ``(batch, experts)``
    and equals ``log(dQ_k/dP)`` for every path/expert pair.
    """
    if controls.ndim != 4:
        raise ValueError("controls must have shape (batch, experts, steps, drivers)")
    if target_increments.ndim != 3:
        raise ValueError("target increments must have shape (batch, steps, drivers)")
    if controls.shape[0] != target_increments.shape[0]:
        raise ValueError("controls and target increments must share the batch size")
    if controls.shape[2:] != target_increments.shape[1:]:
        raise ValueError("controls and target increments must share steps and drivers")
    if controls.shape[1] < 1:
        raise ValueError("at least one expert is required")
    if controls.device != target_increments.device or controls.dtype != target_increments.dtype:
        raise ValueError("controls and target increments must share device and dtype")
    if not controls.is_floating_point():
        raise TypeError("controls and target increments must be floating point")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if not torch.isfinite(controls).all() or not torch.isfinite(target_increments).all():
        raise ValueError("controls and target increments must be finite")
    target = target_increments[:, None, :, :]
    stochastic = torch.sum(controls * target, dim=(-2, -1))
    energy = dt * torch.sum(controls.square(), dim=(-2, -1))
    return stochastic - 0.5 * energy


def log_mixture_q_over_p(
    component_log_q_over_p: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Return ``log(dQ_mix/dP)`` using stable balance-mixture evaluation."""
    if component_log_q_over_p.ndim < 1:
        raise ValueError("component log densities must have an expert axis")
    if not component_log_q_over_p.is_floating_point():
        raise TypeError("component log densities must be floating point")
    if not torch.isfinite(component_log_q_over_p).all():
        raise ValueError("component log densities must be finite")
    components = component_log_q_over_p.shape[-1]
    resolved = _validated_weights(
        weights,
        components=components,
        device=component_log_q_over_p.device,
        dtype=component_log_q_over_p.dtype,
    )
    return torch.logsumexp(component_log_q_over_p + torch.log(resolved), dim=-1)


def selected_component_log_p_over_q(
    component_log_q_over_p: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Return label-preserving component-wise ``log(dP/dQ_K)`` weights."""
    if component_log_q_over_p.ndim != 2:
        raise ValueError("component log densities must have shape (batch, experts)")
    if labels.ndim != 1 or labels.shape[0] != component_log_q_over_p.shape[0]:
        raise ValueError("labels must have shape (batch,)")
    if labels.device != component_log_q_over_p.device:
        raise ValueError("labels and component log densities must share a device")
    if labels.dtype != torch.long:
        raise TypeError("labels must have dtype torch.long")
    if bool((labels < 0).any()) or bool((labels >= component_log_q_over_p.shape[1]).any()):
        raise ValueError("labels contain an invalid expert index")
    selected = torch.gather(component_log_q_over_p, 1, labels[:, None]).squeeze(1)
    return -selected


def sample_mixture_labels(
    weights: torch.Tensor,
    num_samples: int,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw iid categorical expert labels from strictly positive normalized weights."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if not isinstance(weights, torch.Tensor) or weights.ndim != 1:
        raise ValueError("weights must be a one-dimensional torch.Tensor")
    resolved = _validated_weights(
        weights,
        components=weights.shape[0],
        device=weights.device,
        dtype=weights.dtype,
    )
    return torch.multinomial(
        resolved, num_samples=num_samples, replacement=True, generator=generator
    )
