"""Utilities for fixed-rank mesh and Galerkin-rank action audits."""

from __future__ import annotations

import torch


def pad_channel_coefficients(
    coefficients: torch.Tensor,
    *,
    old_modes: int,
    new_modes: int,
    channels: int = 2,
) -> torch.Tensor:
    """Embed channel-blocked DCT coefficients into a larger nested basis."""

    if not 1 <= old_modes <= new_modes:
        raise ValueError("mode counts must satisfy 1 <= old_modes <= new_modes")
    if isinstance(channels, bool) or not isinstance(channels, int) or channels < 1:
        raise ValueError("channels must be a positive integer")
    if coefficients.shape != (channels * old_modes,):
        raise ValueError("coefficient vector has the wrong shape")
    if coefficients.device.type != "cpu" or coefficients.dtype != torch.float64:
        raise ValueError("coefficients must be CPU float64")
    if not torch.isfinite(coefficients).all():
        raise ValueError("coefficients must be finite")
    padded = torch.zeros(channels * new_modes, dtype=torch.float64)
    for channel in range(channels):
        padded[
            channel * new_modes : channel * new_modes + old_modes
        ] = coefficients[channel * old_modes : (channel + 1) * old_modes]
    return padded


def omitted_mode_gradient_norm(
    gradient: torch.Tensor,
    *,
    retained_modes: int,
    expanded_modes: int,
    channels: int = 2,
) -> float:
    """Return the gradient norm in modes omitted by a nested Galerkin solve."""

    if gradient.shape != (channels * expanded_modes,):
        raise ValueError("expanded gradient has the wrong shape")
    mask = torch.ones_like(gradient, dtype=torch.bool)
    for channel in range(channels):
        mask[channel * expanded_modes : channel * expanded_modes + retained_modes] = False
    return float(torch.linalg.vector_norm(gradient[mask]))


def pad_nested_coefficients(
    coefficients: torch.Tensor,
    *,
    new_modes: int,
) -> torch.Tensor:
    """Zero-pad coefficients for a one-block nested continuum basis."""

    if coefficients.ndim != 1 or not coefficients.numel() <= new_modes:
        raise ValueError("new_modes must be at least the current coefficient count")
    if coefficients.device.type != "cpu" or coefficients.dtype != torch.float64:
        raise ValueError("coefficients must be CPU float64")
    if not torch.isfinite(coefficients).all():
        raise ValueError("coefficients must be finite")
    padded = torch.zeros(new_modes, dtype=torch.float64)
    padded[: coefficients.numel()] = coefficients
    return padded


def omitted_tail_gradient_norm(
    gradient: torch.Tensor,
    *,
    retained_modes: int,
) -> float:
    """Return the gradient norm beyond a nested one-block truncation."""

    if gradient.ndim != 1 or not 0 <= retained_modes <= gradient.numel():
        raise ValueError("retained_modes is incompatible with the gradient")
    if gradient.device.type != "cpu" or gradient.dtype != torch.float64:
        raise ValueError("gradient must be CPU float64")
    if not torch.isfinite(gradient).all():
        raise ValueError("gradient must be finite")
    return float(torch.linalg.vector_norm(gradient[retained_modes:]))
