"""rBergomi profile map connecting robust selection to hybrid final execution."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import cast

import torch

from src.path_integral.mlmc import LevelBatch
from src.path_integral.rbergomi_dcs_mlmc import RBergomiPathTask
from src.path_integral.rbergomi_mixture import RBergomiControl
from src.path_integral.rbergomi_mlmc_sampler import (
    CorrectionMethod,
    RBergomiMLMCSampler,
    RBergomiMLMCSamplerConfig,
    RBergomiRawDCSPairBatch,
    SamplingRole,
    SimulationEngine,
)
from src.physics_engine import RBergomiSimulator

_PROFILE_PATTERN = re.compile(r"^(single|correction)_(\d+)$")


def rbergomi_hybrid_profile_ids(finest_level: int) -> tuple[str, ...]:
    if finest_level < 0:
        raise ValueError("finest_level must be nonnegative")
    return tuple(f"single_{level}" for level in range(finest_level + 1)) + tuple(
        f"correction_{level}" for level in range(1, finest_level + 1)
    )


def rbergomi_hybrid_candidate_profiles(
    finest_level: int,
) -> dict[str, tuple[str, ...]]:
    """Return every valid start-level telescope to one fixed finest grid."""

    if finest_level < 0:
        raise ValueError("finest_level must be nonnegative")
    return {
        f"start_{start}": (f"single_{start}",)
        + tuple(f"correction_{level}" for level in range(start + 1, finest_level + 1))
        for start in range(finest_level + 1)
    }


class RBergomiHybridTermSampler:
    """Sample single-level and adjacent DCS terms named by absolute hierarchy level."""

    def __init__(
        self,
        simulator: RBergomiSimulator,
        controls: Sequence[RBergomiControl],
        weights: torch.Tensor,
        task: RBergomiPathTask,
        *,
        spot: float,
        maturity: float,
        coarsest_steps: int,
        finest_level: int,
        engine: SimulationEngine = "fft",
        correction_method: CorrectionMethod = "dcs_mgi",
    ) -> None:
        if finest_level < 0:
            raise ValueError("finest_level must be nonnegative")
        if coarsest_steps < 2 or coarsest_steps % 2:
            raise ValueError("coarsest_steps must be an even integer at least two")
        self.simulator = simulator
        self.controls = tuple(controls)
        self.weights = weights
        self.task = task
        self.spot = float(spot)
        self.maturity = float(maturity)
        self.coarsest_steps = coarsest_steps
        self.finest_level = finest_level
        self.engine = engine
        self.correction_method = correction_method
        self._correction_sampler = self._sampler(coarsest_steps)
        self._single_samplers = {
            level: self._sampler(coarsest_steps * 2**level) for level in range(finest_level + 1)
        }

    def _sampler(self, coarsest_steps: int) -> RBergomiMLMCSampler:
        return RBergomiMLMCSampler(
            self.simulator,
            self.controls,
            self.weights,
            self.task,
            RBergomiMLMCSamplerConfig(
                spot=self.spot,
                maturity=self.maturity,
                coarsest_steps=coarsest_steps,
                method=self.correction_method,
                engine=self.engine,
                dtype=torch.float64,
                require_natural_component=True,
            ),
        )

    def _parse(self, profile_id: str) -> tuple[str, int]:
        match = _PROFILE_PATTERN.fullmatch(profile_id)
        if match is None:
            raise ValueError("invalid rBergomi hybrid profile identifier")
        kind, raw_level = match.groups()
        level = int(raw_level)
        if level > self.finest_level or (kind == "correction" and level < 1):
            raise ValueError("hybrid profile level is outside the fixed hierarchy")
        return kind, level

    def __call__(
        self,
        profile_id: str,
        role: str,
        count: int,
        seeds: Mapping[str, int],
    ) -> LevelBatch:
        if role not in {"pilot", "final"}:
            raise ValueError("hybrid sampling role must be pilot or final")
        sampling_role = cast(SamplingRole, role)
        kind, level = self._parse(profile_id)
        if kind == "single":
            return self._single_samplers[level](0, sampling_role, count, seeds)
        return self._correction_sampler(level, sampling_role, count, seeds)

    def sample_raw_dcs_pair(
        self,
        profile_id: str,
        role: str,
        count: int,
        seeds: Mapping[str, int],
    ) -> RBergomiRawDCSPairBatch:
        """Return matched raw/DCS values for a mechanism diagnostic."""

        if role not in {"pilot", "final"}:
            raise ValueError("hybrid sampling role must be pilot or final")
        sampling_role = cast(SamplingRole, role)
        kind, level = self._parse(profile_id)
        if kind == "single":
            return self._single_samplers[level].sample_raw_dcs_pair(
                0,
                sampling_role,
                count,
                seeds,
            )
        return self._correction_sampler.sample_raw_dcs_pair(
            level,
            sampling_role,
            count,
            seeds,
        )

    def cost_per_sample(self, profile_id: str) -> float:
        _kind, level = self._parse(profile_id)
        steps = self.coarsest_steps * 2**level
        return float(steps * max(1.0, math.log2(steps)))

    @property
    def declared_natural_component_weight(self) -> float:
        """Weight of expert zero, whose zero-control identity is checked at sampling."""

        return float(self.weights[0])

    @property
    def defensive_absolute_bound(self) -> float:
        return 1.0 / self.declared_natural_component_weight
