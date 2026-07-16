"""Lean task-specific feedback controller for two-driver rBergomi proposals."""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn

RBergomiTaskMode = Literal["left", "right", "union"]


class ConstantTwoDriverControl(nn.Module):
    """Non-trainable constant control used by exact CEM baselines."""

    value: torch.Tensor

    def __init__(self, first: float, second: float) -> None:
        super().__init__()
        if not math.isfinite(first) or not math.isfinite(second):
            raise ValueError("constant controls must be finite")
        self.register_buffer("value", torch.tensor([first, second], dtype=torch.float64))

    def forward(
        self,
        _time: float | torch.Tensor,
        spot: torch.Tensor,
        _variance: torch.Tensor,
        _volterra: torch.Tensor,
    ) -> torch.Tensor:
        return self.value.to(device=spot.device, dtype=spot.dtype).expand(spot.shape[0], -1)


class TimePiecewiseTwoDriverControl(nn.Module):
    """Non-trainable deterministic two-driver drift on equal time segments."""

    values: torch.Tensor

    def __init__(
        self,
        values: tuple[tuple[float, float], ...],
        *,
        maturity: float,
    ) -> None:
        super().__init__()
        if not math.isfinite(maturity) or maturity <= 0.0:
            raise ValueError("maturity must be finite and positive")
        if not values or any(len(value) != 2 for value in values):
            raise ValueError("values must contain nonempty two-driver segments")
        if not all(math.isfinite(entry) for value in values for entry in value):
            raise ValueError("piecewise controls must be finite")
        self.maturity = float(maturity)
        self.segments = len(values)
        self.register_buffer("values", torch.tensor(values, dtype=torch.float64))

    def forward(
        self,
        time: float | torch.Tensor,
        spot: torch.Tensor,
        _variance: torch.Tensor,
        _volterra: torch.Tensor,
    ) -> torch.Tensor:
        resolved_time = (
            time.to(device=spot.device, dtype=spot.dtype)
            if torch.is_tensor(time)
            else torch.as_tensor(float(time), device=spot.device, dtype=spot.dtype)
        ).expand_as(spot)
        segment = torch.floor(resolved_time / self.maturity * self.segments).long()
        segment = torch.clamp(segment, min=0, max=self.segments - 1)
        values = self.values.to(device=spot.device, dtype=spot.dtype)
        return values[segment]

    def as_tuple(self) -> tuple[tuple[float, float], ...]:
        return tuple(
            (float(value[0]), float(value[1])) for value in self.values.detach().cpu()
        )


class LeanRBergomiControl(nn.Module):
    """Small stateless controller with no inactive memory-branch overhead."""

    mode_feature: torch.Tensor
    control_bounds: torch.Tensor
    network: nn.Sequential

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
        self.register_buffer("control_bounds", torch.tensor(control_bound, dtype=torch.float32))
        output_layer = nn.Linear(hidden_dim, 2)
        self.network = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            output_layer,
        )
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

    def _features(
        self,
        time: float | torch.Tensor,
        spot: torch.Tensor,
        variance: torch.Tensor,
        volterra: torch.Tensor,
    ) -> torch.Tensor:
        if spot.shape != variance.shape or spot.shape != volterra.shape:
            raise ValueError("spot, variance, and volterra must have identical shapes")
        if spot.ndim != 1 or not all(
            value.is_floating_point() for value in (spot, variance, volterra)
        ):
            raise ValueError("state tensors must be one-dimensional floating-point batches")
        if not all(
            value.device == spot.device and value.dtype == spot.dtype
            for value in (variance, volterra)
        ):
            raise ValueError("state tensors must have identical devices and dtypes")
        if not all(torch.isfinite(value).all() for value in (spot, variance, volterra)):
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
        return features

    def forward(
        self,
        time: float | torch.Tensor,
        spot: torch.Tensor,
        variance: torch.Tensor,
        volterra: torch.Tensor,
    ) -> torch.Tensor:
        raw = self.network(self._features(time, spot, variance, volterra))
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
            control_bound=(
                float(self.control_bounds[0]),
                float(self.control_bounds[1]),
            ),
        ).to(device=reference.device, dtype=reference.dtype)
        result.load_state_dict(self.state_dict())
        result.eval()
        for parameter in result.parameters():
            parameter.requires_grad_(False)
        return result


class CEMAnchoredResidualControl(LeanRBergomiControl):
    """Zero-initialized feedback residual around a fixed two-driver CEM drift."""

    base_control: torch.Tensor
    residual_bounds: torch.Tensor

    def __init__(
        self,
        *,
        spot: float,
        xi: float,
        maturity: float,
        lower_threshold: float,
        upper_threshold: float,
        mode: RBergomiTaskMode,
        base_control: tuple[float, float],
        residual_bound: tuple[float, float] = (2.0, 2.0),
        hidden_dim: int = 24,
        control_bound: tuple[float, float] = (8.0, 8.0),
    ) -> None:
        super().__init__(
            spot=spot,
            xi=xi,
            maturity=maturity,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
            mode=mode,
            hidden_dim=hidden_dim,
            control_bound=control_bound,
        )
        if len(base_control) != 2 or not all(math.isfinite(value) for value in base_control):
            raise ValueError("base_control must contain two finite values")
        if len(residual_bound) != 2 or not all(
            math.isfinite(value) and value > 0.0 for value in residual_bound
        ):
            raise ValueError("residual_bound must contain two finite positive values")
        global_bounds = self.control_bounds.detach().cpu()
        if any(
            abs(value) > float(global_bounds[index]) for index, value in enumerate(base_control)
        ):
            raise ValueError("base_control must lie within the global control bounds")
        self.register_buffer("base_control", torch.tensor(base_control, dtype=torch.float64))
        self.register_buffer("residual_bounds", torch.tensor(residual_bound, dtype=torch.float32))

    def forward(
        self,
        time: float | torch.Tensor,
        spot: torch.Tensor,
        variance: torch.Tensor,
        volterra: torch.Tensor,
    ) -> torch.Tensor:
        raw = self.network(self._features(time, spot, variance, volterra))
        residual_bounds = self.residual_bounds.to(device=raw.device, dtype=raw.dtype)
        base = self.base_control.to(device=raw.device, dtype=raw.dtype)
        total = base + residual_bounds * torch.tanh(raw)
        global_bounds = self.control_bounds.to(device=raw.device, dtype=raw.dtype)
        return torch.maximum(torch.minimum(total, global_bounds), -global_bounds)

    def frozen_copy(self) -> CEMAnchoredResidualControl:
        reference = next(self.parameters())
        result = CEMAnchoredResidualControl(
            spot=self.spot,
            xi=self.xi,
            maturity=self.maturity,
            lower_threshold=self.lower_threshold,
            upper_threshold=self.upper_threshold,
            mode=self.mode,
            hidden_dim=self.hidden_dim,
            control_bound=(
                float(self.control_bounds[0]),
                float(self.control_bounds[1]),
            ),
            base_control=(
                float(self.base_control[0]),
                float(self.base_control[1]),
            ),
            residual_bound=(
                float(self.residual_bounds[0]),
                float(self.residual_bounds[1]),
            ),
        ).to(device=reference.device, dtype=reference.dtype)
        result.load_state_dict(self.state_dict())
        result.eval()
        for parameter in result.parameters():
            parameter.requires_grad_(False)
        return result
