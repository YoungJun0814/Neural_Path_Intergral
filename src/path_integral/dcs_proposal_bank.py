"""Replayable, fully costed construction of rank-one defensive DCS proposals."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import TypeAlias

from src.path_integral.baseline_framework import BaselineCostLedger
from src.path_integral.path_functionals import (
    DiscreteBarrierHitTask,
    TerminalThresholdTask,
)
from src.path_integral.provenance import process_peak_resident_memory_bytes
from src.physics_engine import RBergomiSimulator
from src.training.rbergomi_piecewise_cem import fit_rbergomi_piecewise_cem

DCSBankTask: TypeAlias = TerminalThresholdTask | DiscreteBarrierHitTask
PiecewiseValues = tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class DCSBankCell:
    cell_id: str
    task: DCSBankTask
    hurst: float

    def __post_init__(self) -> None:
        if not self.cell_id:
            raise ValueError("DCS bank cell id cannot be empty")
        if not math.isfinite(self.hurst) or not 0.0 < self.hurst < 0.5:
            raise ValueError("DCS bank Hurst exponent must lie in (0, 0.5)")


@dataclass(frozen=True)
class DCSBankTrainingConfig:
    segments: int
    replicates: int
    paths_per_iteration: int
    maximum_iterations: int
    elite_quantile: float
    smoothing: float
    minimum_elite_paths: int
    control_bound: float
    target_level_repetitions: int
    minimum_price_driver_magnitude: float
    initial_control: PiecewiseValues
    mixture_scales: tuple[float, ...]
    mixture_weights: tuple[float, ...]

    def __post_init__(self) -> None:
        integers = (
            self.segments,
            self.replicates,
            self.paths_per_iteration,
            self.maximum_iterations,
            self.minimum_elite_paths,
            self.target_level_repetitions,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in integers):
            raise ValueError("DCS bank integer settings must be positive")
        if len(self.initial_control) != self.segments:
            raise ValueError("initial-control length must equal segments")
        if any(len(pair) != 2 for pair in self.initial_control):
            raise ValueError("each initial control must have two drivers")
        if any(not math.isfinite(value) for pair in self.initial_control for value in pair):
            raise ValueError("initial controls must be finite")
        if not 0.0 < self.elite_quantile < 1.0 or not 0.0 < self.smoothing <= 1.0:
            raise ValueError("invalid DCS bank CEM quantile or smoothing")
        if self.minimum_elite_paths > self.paths_per_iteration:
            raise ValueError("minimum elites exceed paths per iteration")
        if self.control_bound <= 0.0 or self.minimum_price_driver_magnitude <= 0.0:
            raise ValueError("DCS bank control bounds must be positive")
        if len(self.mixture_scales) != len(self.mixture_weights) or len(self.mixture_scales) < 2:
            raise ValueError("DCS mixture scales and weights are inconsistent")
        if self.mixture_scales[0] != 0.0 or any(value <= 0.0 for value in self.mixture_scales[1:]):
            raise ValueError("DCS mixture requires a natural scale followed by positive scales")
        if any(value <= 0.0 for value in self.mixture_weights) or not math.isclose(
            sum(self.mixture_weights), 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("DCS mixture weights must be positive and sum to one")


@dataclass(frozen=True)
class DCSBankTrainingReplicate:
    seed: int
    iterations: int
    converged: bool
    final_control: PiecewiseValues
    final_level: float
    final_hard_event_fraction: float
    final_probability_estimate: float


@dataclass(frozen=True)
class DCSProposalBankEntry:
    cell_id: str
    hurst: float
    schedules: tuple[PiecewiseValues, ...]
    weights: tuple[float, ...]
    replicates: tuple[DCSBankTrainingReplicate, ...]
    training_cost: BaselineCostLedger
    training_budget_work_units: float


@dataclass(frozen=True)
class DCSProposalBank:
    entries: tuple[DCSProposalBankEntry, ...]
    total_training_cost: BaselineCostLedger
    bank_sha256: str


def _average_controls(replicates: tuple[DCSBankTrainingReplicate, ...]) -> PiecewiseValues:
    count = len(replicates)
    segments = len(replicates[0].final_control)
    return tuple(
        (
            sum(item.final_control[segment][0] for item in replicates) / count,
            sum(item.final_control[segment][1] for item in replicates) / count,
        )
        for segment in range(segments)
    )


def _scaled_schedules(
    profile: PiecewiseValues, scales: tuple[float, ...]
) -> tuple[PiecewiseValues, ...]:
    return tuple(
        tuple((scale * first, scale * second) for first, second in profile)
        for scale in scales
    )


def train_dcs_proposal_bank(
    cells: tuple[DCSBankCell, ...],
    *,
    spot: float,
    maturity: float,
    steps: int,
    eta: float,
    xi: float,
    rho: float,
    base_seed: int,
    config: DCSBankTrainingConfig,
) -> DCSProposalBank:
    """Train every cell independently, average replicates, and charge all work.

    No validation outcome selects a replicate, scale, or mixture weight.  The
    replicate average and scale/weight roster are frozen inputs.  Thus the bank is
    replayable without an unrecorded hyperparameter-search branch.
    """

    if not cells or len({cell.cell_id for cell in cells}) != len(cells):
        raise ValueError("DCS bank cells must be nonempty and unique")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("DCS bank steps must be a positive integer")
    if any(not math.isfinite(value) for value in (spot, maturity, eta, xi, rho)):
        raise ValueError("DCS bank model parameters must be finite")
    if spot <= 0.0 or maturity <= 0.0 or eta <= 0.0 or xi <= 0.0 or not -1.0 < rho < 1.0:
        raise ValueError("invalid DCS bank model parameters")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("DCS bank base seed must be nonnegative")

    entries: list[DCSProposalBankEntry] = []
    total: BaselineCostLedger | None = None
    work_per_path = 6 * steps + 2 * config.segments
    maximum_cell_work = (
        config.replicates
        * config.maximum_iterations
        * config.paths_per_iteration
        * work_per_path
    )
    seed = base_seed
    for cell in cells:
        simulator = RBergomiSimulator(
            H=cell.hurst,
            eta=eta,
            xi=xi,
            rho=rho,
            device="cpu",
        )
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        fits: list[DCSBankTrainingReplicate] = []
        completed_iterations = 0
        for _ in range(config.replicates):
            fit = fit_rbergomi_piecewise_cem(
                simulator,
                cell.task,
                spot=spot,
                maturity=maturity,
                dt=maturity / steps,
                initial_control=config.initial_control,
                num_paths=config.paths_per_iteration,
                seed=seed,
                max_iterations=config.maximum_iterations,
                elite_quantile=config.elite_quantile,
                smoothing=config.smoothing,
                min_elite_paths=config.minimum_elite_paths,
                control_bound=config.control_bound,
                target_level_repetitions=config.target_level_repetitions,
                price_driver_sign="negative",
                minimum_price_driver_magnitude=config.minimum_price_driver_magnitude,
            )
            if not fit.history:
                raise RuntimeError("DCS bank CEM returned an empty history")
            last = fit.history[-1]
            fits.append(
                DCSBankTrainingReplicate(
                    seed=seed,
                    iterations=len(fit.history),
                    converged=fit.converged,
                    final_control=fit.control,
                    final_level=last.level,
                    final_hard_event_fraction=last.hard_event_fraction,
                    final_probability_estimate=last.hard_probability_estimate,
                )
            )
            completed_iterations += len(fit.history)
            seed += 1
        training_samples = completed_iterations * config.paths_per_iteration
        work = training_samples * work_per_path
        cost = BaselineCostLedger(
            training_samples=training_samples,
            optimizer_steps=completed_iterations,
            hyperparameter_trials=1,
            algorithmic_work_units=float(work),
            wall_seconds=time.perf_counter() - wall_started,
            cpu_seconds=time.process_time() - cpu_started,
            peak_memory_bytes=process_peak_resident_memory_bytes(),
            measurement_mode="standardized_hardware_wall",
        )
        profile = _average_controls(tuple(fits))
        entry = DCSProposalBankEntry(
            cell_id=cell.cell_id,
            hurst=cell.hurst,
            schedules=_scaled_schedules(profile, config.mixture_scales),
            weights=config.mixture_weights,
            replicates=tuple(fits),
            training_cost=cost,
            training_budget_work_units=float(maximum_cell_work),
        )
        entries.append(entry)
        total = cost if total is None else total.plus(cost)

    payload = json.dumps(
        [
            {
                "cell_id": entry.cell_id,
                "hurst": entry.hurst,
                "schedules": entry.schedules,
                "weights": entry.weights,
            }
            for entry in entries
        ],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return DCSProposalBank(
        entries=tuple(entries),
        total_training_cost=total if total is not None else BaselineCostLedger(),
        bank_sha256=hashlib.sha256(payload).hexdigest(),
    )
