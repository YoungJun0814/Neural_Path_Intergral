"""Task-specific Volterra--Föllmer Operator (VFO) controller."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from src.path_integral.memory.soe_bank import SOEKernelBank

VFOStage = Literal["instant", "structural", "residual", "joint"]


@dataclass(frozen=True)
class VFOBranchDiagnostics:
    instantaneous_rms: float
    structural_rms: float
    residual_rms: float
    total_rms: float
    structural_gate: float
    residual_gate: float
    residual_energy_fraction: float


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class VolterraFollmerOperator(nn.Module):
    """Additive-gated two-driver control with structural and residual memory.

    Mutable memory is reset by the rBergomi simulator at the beginning of every
    path batch.  ``observe_target_increment`` is called only after the matching
    control has been evaluated, preserving left adaptedness.
    """

    uses_running_minimum = True

    def __init__(
        self,
        *,
        H: float,
        rho: float,
        eta: float,
        xi: float,
        maturity: float,
        barrier: float,
        minimum_dt: float,
        soe_terms: int = 8,
        hidden_dim: int = 32,
        residual_dim: int = 16,
        control_bound: tuple[float, float] = (8.0, 8.0),
    ) -> None:
        super().__init__()
        if not 0.0 < H < 0.5 or not -1.0 <= rho <= 1.0:
            raise ValueError("H or rho is outside its valid range")
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (eta, xi, maturity, barrier, minimum_dt)
        ):
            raise ValueError("eta, xi, maturity, barrier, and minimum_dt must be positive")
        if hidden_dim <= 0 or residual_dim <= 0:
            raise ValueError("hidden dimensions must be positive")
        if len(control_bound) != 2 or not all(value > 0.0 for value in control_bound):
            raise ValueError("control_bound must contain two positive values")

        self.H = float(H)
        self.rho = float(rho)
        self.eta = float(eta)
        self.xi = float(xi)
        self.maturity = float(maturity)
        self.barrier = float(barrier)
        self.minimum_dt = float(minimum_dt)
        self.hidden_dim = int(hidden_dim)
        self.residual_dim = int(residual_dim)
        self.soe_bank = SOEKernelBank(
            H=H,
            minimum_lag=minimum_dt,
            maximum_lag=maturity,
            terms=soe_terms,
        )
        self.instantaneous = _mlp(5, hidden_dim, 2)
        self.structural = _mlp(soe_terms + 4, hidden_dim, 2)
        self.residual_cell = nn.GRUCell(6, residual_dim)
        self.residual_head = _mlp(residual_dim, hidden_dim, 2)
        # B0 starts at the null proposal while hidden features remain
        # nondegenerate and can train immediately.
        nn.init.zeros_(self.instantaneous[-1].weight)
        nn.init.zeros_(self.instantaneous[-1].bias)
        self.structural_gate_parameter = nn.Parameter(torch.zeros(()))
        self.residual_gate_parameter = nn.Parameter(torch.zeros(()))
        self.register_buffer(
            "control_bounds", torch.tensor(control_bound, dtype=torch.float32)
        )

        self._soe_state: torch.Tensor | None = None
        self._residual_state: torch.Tensor | None = None
        self._last_branch_outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._branch_energy_sums: torch.Tensor | None = None
        self._branch_element_count = 0
        self._simulation_steps = 0
        self.set_stage("instant")

    @property
    def structural_gate(self) -> torch.Tensor:
        return torch.tanh(self.structural_gate_parameter)

    @property
    def residual_gate(self) -> torch.Tensor:
        return torch.tanh(self.residual_gate_parameter)

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
        self._residual_state = torch.zeros(
            batch_size, self.residual_dim, device=device, dtype=dtype
        )
        self._last_branch_outputs = None
        self._branch_energy_sums = torch.zeros(4, device=device, dtype=torch.float64)
        self._branch_element_count = 0
        self._simulation_steps = 0

    def _state_features(
        self,
        time: float | torch.Tensor,
        spot: torch.Tensor,
        variance: torch.Tensor,
        volterra: torch.Tensor,
        running_minimum: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._soe_state is None or self._residual_state is None:
            raise RuntimeError("VFO memory must be reset before control evaluation")
        if spot.shape != variance.shape or spot.shape != volterra.shape:
            raise ValueError("spot, variance, and volterra must have identical shapes")
        if self._soe_state.shape[0] != spot.shape[0]:
            raise ValueError("VFO batch size changed without a reset")
        safe_spot = torch.clamp(spot, min=torch.finfo(spot.dtype).tiny)
        safe_variance = torch.clamp(variance, min=1e-12)
        resolved_minimum = spot if running_minimum is None else running_minimum
        if resolved_minimum.shape != spot.shape:
            raise ValueError("running_minimum must match the state shape")
        safe_minimum = torch.clamp(resolved_minimum, min=torch.finfo(spot.dtype).tiny)
        time_tensor = (
            time.to(device=spot.device, dtype=spot.dtype)
            if torch.is_tensor(time)
            else torch.as_tensor(float(time), device=spot.device, dtype=spot.dtype)
        ).expand_as(spot)
        normalized_time = time_tensor / self.maturity
        log_moneyness = torch.log(safe_spot / self.barrier)
        log_variance = torch.log(safe_variance / self.xi)
        barrier_progress = torch.log(safe_minimum / self.barrier)
        instantaneous = torch.stack(
            (
                normalized_time,
                log_moneyness,
                log_variance,
                volterra,
                barrier_progress,
            ),
            dim=-1,
        )
        structural = torch.cat(
            (
                self.soe_bank.weighted_features(self._soe_state),
                normalized_time[:, None],
                log_moneyness[:, None],
                volterra[:, None],
                barrier_progress[:, None],
            ),
            dim=-1,
        )
        residual_input = torch.stack(
            (
                normalized_time,
                log_moneyness,
                log_variance,
                volterra,
                spot / self.barrier - 1.0,
                barrier_progress,
            ),
            dim=-1,
        )
        return instantaneous, structural, residual_input

    def forward(
        self,
        time: float | torch.Tensor,
        spot: torch.Tensor,
        variance: torch.Tensor,
        volterra: torch.Tensor,
        running_minimum: torch.Tensor | None = None,
    ) -> torch.Tensor:
        instantaneous_features, structural_features, residual_input = self._state_features(
            time, spot, variance, volterra, running_minimum
        )
        assert self._residual_state is not None
        self._residual_state = self.residual_cell(residual_input, self._residual_state)
        instantaneous_output = self.instantaneous(instantaneous_features)
        structural_output = self.structural(structural_features)
        residual_output = self.residual_head(self._residual_state)
        raw = (
            instantaneous_output
            + self.structural_gate * structural_output
            + self.residual_gate * residual_output
        )
        bounds = self.control_bounds.to(device=raw.device, dtype=raw.dtype)
        total = bounds * torch.tanh(raw)
        self._last_branch_outputs = (
            instantaneous_output,
            self.structural_gate * structural_output,
            self.residual_gate * residual_output,
        )
        assert self._branch_energy_sums is not None
        detached_outputs = (*self._last_branch_outputs, raw)
        self._branch_energy_sums = self._branch_energy_sums + torch.stack(
            [value.detach().double().square().sum() for value in detached_outputs]
        )
        self._branch_element_count += raw.numel()
        return total

    def observe_target_increment(
        self,
        target_driver_one_increment: torch.Tensor,
        dt: float,
    ) -> None:
        if self._soe_state is None:
            raise RuntimeError("VFO memory must be reset before observing increments")
        self._soe_state = self.soe_bank.update(
            self._soe_state, target_driver_one_increment, dt
        )
        self._simulation_steps += 1

    def branch_diagnostics(self) -> VFOBranchDiagnostics:
        if self._last_branch_outputs is None:
            raise RuntimeError("no VFO control has been evaluated")
        if self._branch_energy_sums is None or self._branch_element_count <= 0:
            raise RuntimeError("VFO branch energy was not accumulated")
        mean_energies = self._branch_energy_sums / self._branch_element_count
        return VFOBranchDiagnostics(
            instantaneous_rms=float(torch.sqrt(mean_energies[0])),
            structural_rms=float(torch.sqrt(mean_energies[1])),
            residual_rms=float(torch.sqrt(mean_energies[2])),
            total_rms=float(torch.sqrt(mean_energies[3])),
            structural_gate=float(self.structural_gate.detach()),
            residual_gate=float(self.residual_gate.detach()),
            residual_energy_fraction=float(
                mean_energies[2] / mean_energies[3].clamp_min(1e-12)
            ),
        )

    def set_stage(self, stage: VFOStage) -> None:
        if stage not in ("instant", "structural", "residual", "joint"):
            raise ValueError("unknown VFO training stage")
        modules = {
            "instant": self.instantaneous,
            "structural": self.structural,
            "residual_cell": self.residual_cell,
            "residual_head": self.residual_head,
        }
        trainable = {
            "instant": {"instant"},
            "structural": {"instant", "structural"},
            "residual": {"residual_cell", "residual_head"},
            "joint": set(modules),
        }[stage]
        for name, module in modules.items():
            for parameter in module.parameters():
                parameter.requires_grad_(name in trainable)
        self.structural_gate_parameter.requires_grad_(stage in ("structural", "joint"))
        self.residual_gate_parameter.requires_grad_(stage in ("residual", "joint"))
        if stage == "instant":
            with torch.no_grad():
                self.structural_gate_parameter.zero_()
                self.residual_gate_parameter.zero_()
        elif stage == "structural":
            with torch.no_grad():
                self.residual_gate_parameter.zero_()

    def frozen_copy(self) -> VolterraFollmerOperator:
        reference = next(self.parameters())
        result = VolterraFollmerOperator(
            H=self.H,
            rho=self.rho,
            eta=self.eta,
            xi=self.xi,
            maturity=self.maturity,
            barrier=self.barrier,
            minimum_dt=self.minimum_dt,
            soe_terms=self.soe_bank.terms,
            hidden_dim=self.hidden_dim,
            residual_dim=self.residual_dim,
            control_bound=tuple(
                float(value) for value in self.control_bounds.detach().cpu()
            ),
        ).to(device=reference.device, dtype=reference.dtype)
        result.load_state_dict(self.state_dict())
        result.eval()
        for parameter in result.parameters():
            parameter.requires_grad_(False)
        return result
