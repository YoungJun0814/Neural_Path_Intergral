"""Variance diagnostics use independent outer units, not inner draws as IID."""

from __future__ import annotations

import math
from typing import Any

import torch


def nested_variance_diagnostic(first: torch.Tensor, second: torch.Tensor) -> dict[str, Any]:
    """Two conditionally independent inner means per shared outer coordinate."""
    if (first.shape != second.shape or first.ndim != 1 or len(first) < 2
            or any(v.dtype != torch.float64 or v.device.type != "cpu" or not torch.isfinite(v).all()
                   for v in (first, second))):
        raise ValueError("invalid paired nested outer contributions")
    scale = float(torch.maximum(first.max(), second.max()))
    x, y = torch.exp(first - scale), torch.exp(second - scale)
    inner_mean_variance = float((x - y).square().mean()) / 2
    midpoint_variance = float(((x + y) / 2).var(unbiased=True))
    outer_variance = midpoint_variance - inner_mean_variance / 2
    mean = float((x + y).mean()) / 2
    if mean <= 0:
        raise FloatingPointError("zero scaled nested mean")
    return {"count": len(first), "log_scale": scale,
            "inner_mean_cv2": inner_mean_variance / mean**2,
            "outer_cv2_unclipped": outer_variance / mean**2,
            "negative_outer_estimate": outer_variance < 0,
            "log_combined_mean": scale + math.log(mean),
            "scope": "paired_inner_noise_decomposition_not_tail_oracle"}
