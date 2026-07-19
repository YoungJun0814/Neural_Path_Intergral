"""Production sampler bridge from rBergomi corrections to the generic MLMC engine."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from src.path_integral.mlmc import LevelBatch
from src.path_integral.rbergomi_dcs_mlmc import (
    RBergomiPathTask,
    evaluate_rbergomi_dcs_adjacent,
    evaluate_rbergomi_dcs_level,
)
from src.path_integral.rbergomi_mixture import (
    RBergomiControl,
    simulate_rbergomi_mixture,
)
from src.path_integral.rbergomi_multilevel import simulate_coupled_rbergomi_mixture
from src.physics_engine import RBergomiSimulator

CorrectionMethod = Literal["raw_defensive", "dcs_mgi"]
SimulationEngine = Literal["reference", "fft"]


@dataclass(frozen=True)
class RBergomiMLMCSamplerConfig:
    """Frozen non-random inputs shared by raw and DCS estimators."""

    spot: float
    maturity: float
    coarsest_steps: int
    method: CorrectionMethod
    engine: SimulationEngine = "fft"
    dtype: torch.dtype = torch.float64

    def __post_init__(self) -> None:
        if not math.isfinite(self.spot) or self.spot <= 0.0:
            raise ValueError("spot must be finite and positive")
        if not math.isfinite(self.maturity) or self.maturity <= 0.0:
            raise ValueError("maturity must be finite and positive")
        if self.coarsest_steps < 2 or self.coarsest_steps % 2:
            raise ValueError("coarsest_steps must be an even integer at least two")
        if self.method not in ("raw_defensive", "dcs_mgi"):
            raise ValueError("unsupported correction method")
        if self.engine not in ("reference", "fft"):
            raise ValueError("unsupported simulation engine")
        if self.dtype != torch.float64:
            raise ValueError("G11 research-evidence sampling requires torch.float64")


class RBergomiMLMCSampler:
    """Generate exact level-zero or adjacent terms from fully declared seeds."""

    def __init__(
        self,
        simulator: RBergomiSimulator,
        controls: Sequence[RBergomiControl],
        weights: torch.Tensor,
        task: RBergomiPathTask,
        config: RBergomiMLMCSamplerConfig,
    ) -> None:
        if not controls:
            raise ValueError("at least one deterministic control is required")
        if any(
            not bool(getattr(control, "is_deterministic_time_control", False))
            for control in controls
        ):
            raise ValueError("G11 DCS sampler rejects path-dependent controls")
        if (
            weights.ndim != 1
            or weights.shape[0] != len(controls)
            or not weights.is_floating_point()
            or not torch.isfinite(weights).all()
            or bool((weights <= 0.0).any())
            or abs(float(torch.sum(weights)) - 1.0) > 1e-12
        ):
            raise ValueError("weights must be a positive normalized floating vector")
        self.simulator = simulator
        self.controls = tuple(controls)
        self.weights = weights.to(device=simulator.device, dtype=config.dtype)
        self.task = task
        self.config = config

    @staticmethod
    def _validate_natural_component(all_expert_controls: torch.Tensor) -> None:
        if float(torch.max(torch.abs(all_expert_controls[:, 0]))) > 1e-13:
            raise ValueError("expert zero must be the zero-control natural component")

    def __call__(
        self,
        level: int,
        role: Literal["pilot", "final"],
        count: int,
        seeds: Mapping[str, int],
    ) -> LevelBatch:
        del role
        if level < 0 or count < 1:
            raise ValueError("level must be nonnegative and count must be positive")
        if set(seeds) != {"proposal", "labels"}:
            raise ValueError("rBergomi sampler requires proposal and label streams")
        torch.manual_seed(seeds["proposal"])
        label_generator = torch.Generator(device="cpu").manual_seed(seeds["labels"])
        fine_steps = self.config.coarsest_steps * 2**level
        start = time.perf_counter()
        if level == 0:
            sample = simulate_rbergomi_mixture(
                self.simulator,
                self.controls,
                self.weights,
                spot=self.config.spot,
                maturity=self.config.maturity,
                dt=self.config.maturity / fine_steps,
                num_paths=count,
                dtype=self.config.dtype,
                label_generator=label_generator,
                engine=self.config.engine,
            )
            self._validate_natural_component(sample.all_expert_controls)
            evaluation = evaluate_rbergomi_dcs_level(
                sample, task=self.task, rho=self.simulator.rho
            )
            values = (
                evaluation.raw_contribution
                if self.config.method == "raw_defensive"
                else evaluation.marginalized_contribution
            )
        else:
            sample = simulate_coupled_rbergomi_mixture(
                self.simulator,
                self.controls,
                self.weights,
                spot=self.config.spot,
                maturity=self.config.maturity,
                fine_steps=fine_steps,
                num_paths=count,
                dtype=self.config.dtype,
                label_generator=label_generator,
                engine=self.config.engine,
            )
            self._validate_natural_component(sample.all_expert_controls)
            evaluation = evaluate_rbergomi_dcs_adjacent(
                sample, task=self.task, rho=self.simulator.rho
            )
            values = (
                evaluation.raw_correction
                if self.config.method == "raw_defensive"
                else evaluation.marginalized_correction
            )
        elapsed = time.perf_counter() - start
        if not torch.isfinite(values).all():
            raise FloatingPointError("rBergomi MLMC term became nonfinite")
        operation_proxy = count * fine_steps * max(1.0, math.log2(fine_steps))
        return LevelBatch(
            values.detach().clone(),
            work_units=float(operation_proxy),
            wall_seconds=elapsed,
        )
