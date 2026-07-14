"""Lean task-specific feedback controller for two-driver rBergomi proposals."""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn

RBergomiTaskMode = Literal["left", "right", "union"]


class ConstantTwoDriverControl(nn.Module):
    """Non-trainable constant control used by exact CEM baselines."""

    def __init__(self, first: float, second: float) -> None:
        super().__init__()
        if not math.isfinite(first) or not math.isfinite(second):
            raise ValueError("constant controls must be finite")
        self.register_buffer(
            "value", torch.tensor([first, second], dtype=torch.float64)
        )

    def forward(
        self,
        _time: float | torch.Tensor,
        spot: torch.Tensor,
        _variance: torch.Tensor,
        _volterra: torch.Tensor,
    ) -> torch.Tensor:
        return self.value.to(device=spot.device, dtype=spot.dtype).expand(spot.shape[0], -1)


class LeanRBergomiControl(nn.Module):
    """Small stateless controller with no inactive memory-branch overhead."""

    def __init__(
        self,
        *,
        spot: float,
        xi: float,
        maturity: float,
        lower_threshold: float,
        upper_threshold: float,
        mode: RBergomiTaskMode,
        hidden_dim: int = 24,
        control_bound: tuple[float, float] = (8.0, 8.0),
    ) -> None:
        super().__init__()
        values = (spot, xi, maturity, lower_threshold, upper_threshold)
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("spot, xi, maturity, and thresholds must be finite and positive")
        if lower_threshold >= upper_threshold:
            raise ValueError("lower_threshold must be smaller than upper_threshold")
        if mode not in ("left", "right", "union"):
            raise ValueError("mode must be 'left', 'right', or 'union'")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if len(control_bound) != 2 or not all(
            math.isfinite(value) and value > 0.0 for value in control_bound
        ):
            raise ValueError("control_bound must contain two finite positive values")
        self.spot = float(spot)
        self.xi = float(xi)
        self.maturity = float(maturity)
        self.lower_threshold = float(lower_threshold)
        self.upper_threshold = float(upper_threshold)
        self.mode: RBergomiTaskMode = mode
        self.hidden_dim = int(hidden_dim)
        mode_index = {"left": 0, "right": 1, "union": 2}[mode]
        mode_feature = torch.zeros(3, dtype=torch.float32)
        mode_feature[mode_index] = 1.0
        self.register_buffer("mode_feature", mode_feature)
        self.register_buffer(
            "control_bounds", torch.tensor(control_bound, dtype=torch.float32)
        )
        self.network = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        time: float | torch.Tensor,
        spot: torch.Tensor,
        variance: torch.Tensor,
        volterra: torch.Tensor,
    ) -> torch.Tensor:
        if spot.shape != variance.shape or spot.shape != volterra.shape:
            raise ValueError("spot, variance, and volterra must have identical shapes")
        if spot.ndim != 1 or not spot.is_floating_point():
            raise ValueError("state tensors must be one-dimensional floating-point batches")
        if not torch.isfinite(spot).all() or not torch.isfinite(variance).all():
            raise ValueError("state tensors must be finite")
        tiny = torch.finfo(spot.dtype).tiny
        safe_spot = torch.clamp(spot, min=tiny)
        safe_variance = torch.clamp(variance, min=tiny)
        time_tensor = (
            time.to(device=spot.device, dtype=spot.dtype)
            if torch.is_tensor(time)
            else torch.as_tensor(float(time), device=spot.device, dtype=spot.dtype)
        ).expand_as(spot)
        mode = self.mode_feature.to(device=spot.device, dtype=spot.dtype)
        mode = mode.expand(spot.shape[0], -1)
        features = torch.cat(
            (
                (time_tensor / self.maturity)[:, None],
                torch.log(safe_spot / self.spot)[:, None],
                torch.log(safe_variance / self.xi)[:, None],
                volterra[:, None],
                torch.log(safe_spot / self.lower_threshold)[:, None],
                torch.log(self.upper_threshold / safe_spot)[:, None],
                mode,
            ),
            dim=-1,
        )
        raw = self.network(features)
        bounds = self.control_bounds.to(device=raw.device, dtype=raw.dtype)
        return bounds * torch.tanh(raw)

    def frozen_copy(self) -> LeanRBergomiControl:
        reference = next(self.parameters())
        result = LeanRBergomiControl(
            spot=self.spot,
            xi=self.xi,
            maturity=self.maturity,
            lower_threshold=self.lower_threshold,
            upper_threshold=self.upper_threshold,
            mode=self.mode,
            hidden_dim=self.hidden_dim,
            control_bound=tuple(
                float(value) for value in self.control_bounds.detach().cpu()
            ),
        ).to(device=reference.device, dtype=reference.dtype)
        result.load_state_dict(self.state_dict())
        result.eval()
        for parameter in result.parameters():
            parameter.requires_grad_(False)
        return result
