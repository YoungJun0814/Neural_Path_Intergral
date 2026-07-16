"""Spectral Doob--Volterra controller for finite-grid excursion sampling."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.path_integral.memory import SOEKernelBank


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class SpectralDoobVolterraControl(nn.Module):
    """Bounded piecewise-anchored control with causal SOE path features.

    The sum-of-exponentials bank is only a controller feature.  The target path
    remains the unmodified BLP finite-grid rBergomi law in the simulator.
    """

    uses_running_minimum = True
    control_bounds: torch.Tensor
    anchor_values: torch.Tensor
    residual_bounds: torch.Tensor

    def __init__(
        self,
        *,
        H: float,
        spot: float,
        xi: float,
        maturity: float,
        hit_barrier: float,
        stress_level: float,
        minimum_occupation: float,
        minimum_dt: float,
        anchor_values: tuple[tuple[float, float], ...],
        soe_terms: int = 8,
        hidden_dim: int = 64,
        control_bound: tuple[float, float] = (8.0, 8.0),
        residual_bound: tuple[float, float] = (2.0, 2.0),
        desirability_floor: float = 1e-5,
        initial_desirability: float = 0.05,
    ) -> None:
        super().__init__()
        positive = (
            spot,
            xi,
            maturity,
            hit_barrier,
            stress_level,
            minimum_occupation,
            minimum_dt,
            desirability_floor,
            initial_desirability,
        )
        if not 0.0 < H < 0.5:
            raise ValueError("H must lie in (0, 0.5)")
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("SDV scale parameters must be finite and positive")
        if hit_barrier >= stress_level or minimum_occupation > maturity:
            raise ValueError("SDV path-functional thresholds are inconsistent")
        if hidden_dim <= 0 or soe_terms < 2:
            raise ValueError("hidden_dim and soe_terms are too small")
        if not anchor_values or any(len(pair) != 2 for pair in anchor_values):
            raise ValueError("anchor_values must contain two-driver segments")
        if not all(math.isfinite(value) for pair in anchor_values for value in pair):
            raise ValueError("anchor values must be finite")
        if len(control_bound) != 2 or len(residual_bound) != 2:
            raise ValueError("control and residual bounds must have two coordinates")
        if not all(math.isfinite(value) and value > 0.0 for value in control_bound):
            raise ValueError("control bounds must be finite and positive")
        if not all(math.isfinite(value) and value > 0.0 for value in residual_bound):
            raise ValueError("residual bounds must be finite and positive")
        if not desirability_floor < initial_desirability < 1.0:
            raise ValueError("initial desirability must lie between its floor and one")
        if any(
            abs(value) >= control_bound[index]
            for pair in anchor_values
            for index, value in enumerate(pair)
        ):
            raise ValueError("anchor controls must lie strictly inside global bounds")

        self.H = float(H)
        self.spot = float(spot)
        self.xi = float(xi)
        self.maturity = float(maturity)
        self.hit_barrier = float(hit_barrier)
        self.stress_level = float(stress_level)
        self.minimum_occupation = float(minimum_occupation)
        self.minimum_dt = float(minimum_dt)
        self.hidden_dim = int(hidden_dim)
        self.desirability_floor = float(desirability_floor)
        self.initial_desirability = float(initial_desirability)
        self.soe_bank = SOEKernelBank(
            H=H,
            minimum_lag=minimum_dt,
            maximum_lag=maturity,
            terms=soe_terms,
        )
        input_dim = 7 + soe_terms
        self.desirability_network = _mlp(input_dim, hidden_dim, 1)
        self.residual_network = _mlp(input_dim, hidden_dim, 2)
        self.register_buffer("anchor_values", torch.tensor(anchor_values, dtype=torch.float64))
        self.register_buffer("control_bounds", torch.tensor(control_bound, dtype=torch.float64))
        self.register_buffer("residual_bounds", torch.tensor(residual_bound, dtype=torch.float64))

        h_output = self.desirability_network[-1]
        residual_output = self.residual_network[-1]
        if not isinstance(h_output, nn.Linear) or not isinstance(residual_output, nn.Linear):
            raise TypeError("SDV heads must end in linear layers")
        adjusted = (initial_desirability - desirability_floor) / (1.0 - desirability_floor)
        nn.init.zeros_(h_output.weight)
        nn.init.constant_(h_output.bias, math.log(adjusted / (1.0 - adjusted)))
        nn.init.zeros_(residual_output.weight)
        nn.init.zeros_(residual_output.bias)

        self._soe_state: torch.Tensor | None = None
        self._occupation: torch.Tensor | None = None
        self._last_time: torch.Tensor | None = None
        self._last_desirability: torch.Tensor | None = None

    @property
    def segments(self) -> int:
        return int(self.anchor_values.shape[0])

    @property
    def last_desirability(self) -> torch.Tensor:
        if self._last_desirability is None:
            raise RuntimeError("SDV has not evaluated a desirability value")
        return self._last_desirability

    def reset_for_simulation(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self._soe_state = self.soe_bank.initial_state(
            batch_size, device=device, dtype=dtype
        )
        self._occupation = torch.zeros(batch_size, device=device, dtype=dtype)
        self._last_time = None
        self._last_desirability = None

    def _resolved_time(
        self, time: float | torch.Tensor, spot: torch.Tensor
    ) -> torch.Tensor:
        return (
            time.to(device=spot.device, dtype=spot.dtype)
            if torch.is_tensor(time)
            else torch.as_tensor(float(time), device=spot.device, dtype=spot.dtype)
        ).expand_as(spot)

    def _features(
        self,
        time: float | torch.Tensor,
        spot: torch.Tensor,
        variance: torch.Tensor,
        volterra: torch.Tensor,
        running_minimum: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._soe_state is None or self._occupation is None:
            raise RuntimeError("SDV memory must be reset before evaluation")
        if not (
            spot.shape
            == variance.shape
            == volterra.shape
            == running_minimum.shape
            == self._occupation.shape
        ):
            raise ValueError("SDV state tensors must have identical shapes")
        resolved_time = self._resolved_time(time, spot)
        if self._last_time is not None:
            elapsed = resolved_time - self._last_time
            if bool(torch.any(elapsed < -1e-12)):
                raise ValueError("SDV time must be nondecreasing")
            self._occupation = self._occupation + torch.clamp(elapsed, min=0.0) * (
                spot <= self.stress_level
            ).to(spot.dtype)
        self._last_time = resolved_time.detach().clone()

        tiny = torch.finfo(spot.dtype).tiny
        scalar = torch.stack(
            (
                resolved_time / self.maturity,
                torch.log(torch.clamp(spot, min=tiny) / self.spot),
                torch.log(torch.clamp(variance, min=1e-12) / self.xi),
                volterra,
                torch.log(torch.clamp(running_minimum, min=tiny) / self.spot),
                self._occupation / self.maturity,
                (running_minimum <= self.hit_barrier).to(spot.dtype),
            ),
            dim=-1,
        )
        spectral = self.soe_bank.weighted_features(self._soe_state)
        return torch.cat((scalar, spectral), dim=-1), resolved_time

    def anchor_at(self, resolved_time: torch.Tensor) -> torch.Tensor:
        segment = torch.floor(resolved_time / self.maturity * self.segments).long()
        segment = torch.clamp(segment, min=0, max=self.segments - 1)
        return self.anchor_values.to(
            device=resolved_time.device, dtype=resolved_time.dtype
        )[segment]

    def forward(
        self,
        time: float | torch.Tensor,
        spot: torch.Tensor,
        variance: torch.Tensor,
        volterra: torch.Tensor,
        running_minimum: torch.Tensor,
    ) -> torch.Tensor:
        features, resolved_time = self._features(
            time, spot, variance, volterra, running_minimum
        )
        raw_h = self.desirability_network(features).squeeze(-1)
        self._last_desirability = self.desirability_floor + (
            1.0 - self.desirability_floor
        ) * torch.sigmoid(raw_h)

        anchor = self.anchor_at(resolved_time)
        bounds = self.control_bounds.to(device=features.device, dtype=features.dtype)
        residual_bounds = self.residual_bounds.to(
            device=features.device, dtype=features.dtype
        )
        residual = residual_bounds * torch.tanh(self.residual_network(features))
        anchor_coordinate = torch.atanh(anchor / bounds)
        return bounds * torch.tanh(anchor_coordinate + residual)

    def observe_target_increment(
        self, target_driver_one_increment: torch.Tensor, dt: float
    ) -> None:
        if self._soe_state is None:
            raise RuntimeError("SDV memory must be reset before observing increments")
        self._soe_state = self.soe_bank.update(
            self._soe_state, target_driver_one_increment, dt
        )

    def frozen_copy(self) -> SpectralDoobVolterraControl:
        reference = next(self.parameters())
        result = SpectralDoobVolterraControl(
            H=self.H,
            spot=self.spot,
            xi=self.xi,
            maturity=self.maturity,
            hit_barrier=self.hit_barrier,
            stress_level=self.stress_level,
            minimum_occupation=self.minimum_occupation,
            minimum_dt=self.minimum_dt,
            anchor_values=tuple(
                (float(pair[0]), float(pair[1])) for pair in self.anchor_values
            ),
            soe_terms=self.soe_bank.terms,
            hidden_dim=self.hidden_dim,
            control_bound=(float(self.control_bounds[0]), float(self.control_bounds[1])),
            residual_bound=(
                float(self.residual_bounds[0]),
                float(self.residual_bounds[1]),
            ),
            desirability_floor=self.desirability_floor,
            initial_desirability=self.initial_desirability,
        ).to(device=reference.device, dtype=reference.dtype)
        result.load_state_dict(self.state_dict())
        result.eval()
        for parameter in result.parameters():
            parameter.requires_grad_(False)
        return result
