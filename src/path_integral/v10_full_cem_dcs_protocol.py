"""V10 Protocol: Full-Dimensional Defensive CEM + Exact DCS Integration Benchmark Engine."""

from __future__ import annotations

import math

import torch

from src.path_integral.dcs_benchmark import PairedDCSBenchmarkBatch, evaluate_paired_dcs_benchmark
from src.path_integral.path_functionals import (
    DiscreteBarrierHitTask,
    DownsideExcursionTask,
    TerminalThresholdTask,
)
from src.path_integral.rbergomi_mixture import RBergomiControl
from src.path_integral.v10_proposal_bank import V10ProposalBankEntry

V10Task = TerminalThresholdTask | DiscreteBarrierHitTask | DownsideExcursionTask


class V10DefensiveCEMControl:
    """Deterministic time-only control representing full-dimensional CEM shift."""

    is_deterministic_time_control = True

    def __init__(self, schedule: torch.Tensor, step_dt: float) -> None:
        if schedule.ndim != 2 or schedule.shape[1] != 2:
            raise ValueError("schedule must have shape (steps, 2)")
        self.schedule = schedule
        self.step_dt = step_dt

    def deterministic_schedule(self, times: torch.Tensor) -> torch.Tensor:
        return self.schedule

    def __call__(
        self, t: float, spot: torch.Tensor, variance: torch.Tensor, volterra: torch.Tensor
    ) -> torch.Tensor:
        step = min(int(round(t / self.step_dt)), self.schedule.shape[0] - 1)
        return self.schedule[step].unsqueeze(0).expand(spot.shape[0], -1)


def create_v10_controls_and_weights(
    entry: V10ProposalBankEntry, steps: int, step_dt: float
) -> tuple[list[RBergomiControl], torch.Tensor]:
    """Build natural and CEM expert controls from a V10 bank entry."""
    dim = entry.dimension
    if dim != 3 * steps:
        raise ValueError(f"entry dimension {dim} disagrees with 3 * steps ({3 * steps})")

    # Natural component: zero drift
    zero_schedule = torch.zeros((steps, 2), dtype=torch.float64)
    natural_control = V10DefensiveCEMControl(zero_schedule, step_dt)

    # CEM component: shift / sqrt(dt)
    learned = torch.tensor(entry.learned_mean, dtype=torch.float64)
    local_part = learned[: 2 * steps].reshape(steps, 2)
    price_part = learned[2 * steps :]

    # Driver 0: local Volterra cell shift
    # Driver 1: price driver shift
    cem_schedule = torch.zeros((steps, 2), dtype=torch.float64)
    cem_schedule[:, 0] = local_part[:, 0] / math.sqrt(step_dt)
    cem_schedule[:, 1] = price_part / math.sqrt(step_dt)

    cem_control = V10DefensiveCEMControl(cem_schedule, step_dt)

    controls: list[RBergomiControl] = [natural_control, cem_control]
    defensive_weight = entry.defensive_weight
    weights = torch.tensor([defensive_weight, 1.0 - defensive_weight], dtype=torch.float64)

    return controls, weights


def evaluate_v10_paired_dcs_benchmark(
    *,
    entry: V10ProposalBankEntry,
    task: V10Task,
    spot: float,
    maturity: float,
    steps: int,
    eta: float,
    xi: float,
    rho: float,
    sample_count: int,
    path_seed: int,
    label_seed: int,
) -> PairedDCSBenchmarkBatch:
    """Evaluate paired Raw vs DCS benchmark on full-dimensional Defensive CEM proposal."""
    step_dt = maturity / steps
    controls, weights = create_v10_controls_and_weights(entry, steps, step_dt)

    return evaluate_paired_dcs_benchmark(
        task=task,
        task_id=entry.cell_id,
        spot=spot,
        maturity=maturity,
        steps=steps,
        hurst=entry.hurst,
        eta=eta,
        xi=xi,
        rho=rho,
        controls=controls,
        weights=weights,
        sample_count=sample_count,
        path_seed=path_seed,
        label_seed=label_seed,
    )
