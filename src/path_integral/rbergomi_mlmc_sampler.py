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

CorrectionMethod = Literal["raw", "raw_defensive", "dcs_mgi"]
SimulationEngine = Literal["reference", "fft"]
SamplingRole = Literal["pilot", "final"]


@dataclass(frozen=True)
class RBergomiMLMCSamplerConfig:
    """Frozen non-random inputs shared by raw and DCS estimators."""

    spot: float
    maturity: float
    coarsest_steps: int
    method: CorrectionMethod
    engine: SimulationEngine = "fft"
    dtype: torch.dtype = torch.float64
    require_natural_component: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.spot) or self.spot <= 0.0:
            raise ValueError("spot must be finite and positive")
        if not math.isfinite(self.maturity) or self.maturity <= 0.0:
            raise ValueError("maturity must be finite and positive")
        if self.coarsest_steps < 2 or self.coarsest_steps % 2:
            raise ValueError("coarsest_steps must be an even integer at least two")
        if self.method not in ("raw", "raw_defensive", "dcs_mgi"):
            raise ValueError("unsupported correction method")
        if self.method == "dcs_mgi" and not self.require_natural_component:
            raise ValueError("DCS evidence requires a defensive natural component")
        if self.engine not in ("reference", "fft"):
            raise ValueError("unsupported simulation engine")
        if self.dtype != torch.float64:
            raise ValueError("G11 research-evidence sampling requires torch.float64")


@dataclass(frozen=True)
class RBergomiRawDCSPairBatch:
    """Raw and Rao--Blackwellized contributions from one identical path batch."""

    raw_values: torch.Tensor
    dcs_values: torch.Tensor
    work_units: float
    wall_seconds: float

    def __post_init__(self) -> None:
        if (
            self.raw_values.ndim != 1
            or self.dcs_values.shape != self.raw_values.shape
            or self.raw_values.numel() < 1
            or self.raw_values.dtype != torch.float64
            or self.dcs_values.dtype != torch.float64
            or not torch.isfinite(self.raw_values).all()
            or not torch.isfinite(self.dcs_values).all()
        ):
            raise ValueError("paired raw/DCS values must be matching finite float64 vectors")
        if (
            not math.isfinite(self.work_units)
            or self.work_units <= 0.0
            or not math.isfinite(self.wall_seconds)
            or self.wall_seconds < 0.0
        ):
            raise ValueError("paired raw/DCS work measurements are invalid")


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
        role: SamplingRole,
        count: int,
        seeds: Mapping[str, int],
    ) -> LevelBatch:
        if role not in {"pilot", "final"}:
            raise ValueError("rBergomi sampling role must be pilot or final")
        if level < 0 or count < 1:
            raise ValueError("level must be nonnegative and count must be positive")
        if set(seeds) != {"proposal", "labels"}:
            raise ValueError("rBergomi sampler requires proposal and label streams")
        torch.manual_seed(seeds["proposal"])
        label_generator = torch.Generator(device="cpu").manual_seed(seeds["labels"])
        fine_steps = self.config.coarsest_steps * 2**level
        start = time.perf_counter()
        if level == 0:
            level_sample = simulate_rbergomi_mixture(
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
            if self.config.require_natural_component:
                self._validate_natural_component(level_sample.all_expert_controls)
            if self.config.method in {"raw", "raw_defensive"}:
                hard_event = self.task.hard_event(
                    level_sample.paths.spot,
                    level_sample.paths.step_dt,
                )
                values = hard_event.to(self.config.dtype) * torch.exp(
                    level_sample.mixture_log_likelihood
                )
            else:
                level_evaluation = evaluate_rbergomi_dcs_level(
                    level_sample, task=self.task, rho=self.simulator.rho
                )
                values = level_evaluation.marginalized_contribution
        else:
            coupled_sample = simulate_coupled_rbergomi_mixture(
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
            if self.config.require_natural_component:
                self._validate_natural_component(coupled_sample.all_expert_controls)
            if self.config.method in {"raw", "raw_defensive"}:
                fine_event = self.task.hard_event(
                    coupled_sample.paths.fine.spot,
                    coupled_sample.paths.fine.step_dt,
                )
                coarse_event = self.task.hard_event(
                    coupled_sample.paths.coarse.spot,
                    coupled_sample.paths.coarse.step_dt,
                )
                values = (
                    fine_event.to(self.config.dtype)
                    - coarse_event.to(self.config.dtype)
                ) * torch.exp(coupled_sample.mixture_log_likelihood)
            else:
                adjacent_evaluation = evaluate_rbergomi_dcs_adjacent(
                    coupled_sample, task=self.task, rho=self.simulator.rho
                )
                values = adjacent_evaluation.marginalized_correction
        elapsed = time.perf_counter() - start
        if not torch.isfinite(values).all():
            raise FloatingPointError("rBergomi MLMC term became nonfinite")
        operation_proxy = count * fine_steps * max(1.0, math.log2(fine_steps))
        return LevelBatch(
            values.detach().clone(),
            work_units=float(operation_proxy),
            wall_seconds=elapsed,
        )

    def sample_raw_dcs_pair(
        self,
        level: int,
        role: SamplingRole,
        count: int,
        seeds: Mapping[str, int],
    ) -> RBergomiRawDCSPairBatch:
        """Evaluate raw and exact conditional contributions on the same paths.

        This is a mechanism-diagnostic path.  It intentionally computes both
        contributions and must not be used to report the isolated runtime of
        either production estimator.
        """

        if not self.config.require_natural_component:
            raise ValueError("paired DCS diagnostics require a defensive natural component")
        if role not in {"pilot", "final"}:
            raise ValueError("rBergomi sampling role must be pilot or final")
        if level < 0 or count < 1:
            raise ValueError("level must be nonnegative and count must be positive")
        if set(seeds) != {"proposal", "labels"}:
            raise ValueError("rBergomi sampler requires proposal and label streams")
        torch.manual_seed(seeds["proposal"])
        label_generator = torch.Generator(device="cpu").manual_seed(seeds["labels"])
        fine_steps = self.config.coarsest_steps * 2**level
        start = time.perf_counter()
        if level == 0:
            level_sample = simulate_rbergomi_mixture(
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
            self._validate_natural_component(level_sample.all_expert_controls)
            evaluation = evaluate_rbergomi_dcs_level(
                level_sample, task=self.task, rho=self.simulator.rho
            )
            hard_event = self.task.hard_event(
                level_sample.paths.spot,
                level_sample.paths.step_dt,
            )
            raw_values = hard_event.to(self.config.dtype) * torch.exp(
                level_sample.mixture_log_likelihood
            )
            dcs_values = evaluation.marginalized_contribution
        else:
            coupled_sample = simulate_coupled_rbergomi_mixture(
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
            self._validate_natural_component(coupled_sample.all_expert_controls)
            evaluation_adjacent = evaluate_rbergomi_dcs_adjacent(
                coupled_sample, task=self.task, rho=self.simulator.rho
            )
            fine_event = self.task.hard_event(
                coupled_sample.paths.fine.spot,
                coupled_sample.paths.fine.step_dt,
            )
            coarse_event = self.task.hard_event(
                coupled_sample.paths.coarse.spot,
                coupled_sample.paths.coarse.step_dt,
            )
            raw_values = (
                fine_event.to(self.config.dtype)
                - coarse_event.to(self.config.dtype)
            ) * torch.exp(coupled_sample.mixture_log_likelihood)
            dcs_values = evaluation_adjacent.marginalized_correction
        elapsed = time.perf_counter() - start
        operation_proxy = count * fine_steps * max(1.0, math.log2(fine_steps))
        return RBergomiRawDCSPairBatch(
            raw_values=raw_values.detach().clone(),
            dcs_values=dcs_values.detach().clone(),
            work_units=float(operation_proxy),
            wall_seconds=elapsed,
        )
