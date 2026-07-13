"""Strictly positive soft targets represented through stable potentials."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def terminal_left_tail_potential(
    terminal_value: torch.Tensor,
    barrier: float,
    temperature: float,
) -> torch.Tensor:
    r"""Return the stable potential for a soft terminal left-tail event.

    The associated functional is

    .. math::

        g_{\tau,K}(X_T)=\operatorname{sigmoid}
        ((K-X_T)/(\tau K)),

    and this function evaluates ``Phi=-log(g)`` as
    ``softplus((X_T-K)/(temperature*K))`` without first forming ``g``.
    ``temperature`` is dimensionless and ``barrier`` must therefore be
    strictly positive.
    """
    if not terminal_value.is_floating_point():
        raise TypeError("terminal_value must be a floating-point tensor")
    if not math.isfinite(barrier) or barrier <= 0.0:
        raise ValueError("barrier must be finite and positive")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not torch.isfinite(terminal_value).all():
        raise ValueError("terminal_value must be finite")
    scaled_margin = (terminal_value - barrier) / (temperature * barrier)
    return F.softplus(scaled_margin)
