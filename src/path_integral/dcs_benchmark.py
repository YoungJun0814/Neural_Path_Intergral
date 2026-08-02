"""Paired raw/DCS mechanism benchmark on one exact rBergomi mixture law."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from src.path_integral.baseline_framework import BaselineCostLedger
from src.path_integral.path_functionals import (
    DiscreteBarrierHitTask,
    DownsideExcursionTask,
    TerminalThresholdTask,
)
from src.path_integral.provenance import process_peak_resident_memory_bytes
from src.path_integral.rbergomi_dcs_mlmc import evaluate_rbergomi_dcs_level
from src.path_integral.rbergomi_mixture import RBergomiControl, simulate_rbergomi_mixture
from src.physics_engine import RBergomiSimulator

DCSBenchmarkTask = TerminalThresholdTask | DiscreteBarrierHitTask | DownsideExcursionTask


@dataclass(frozen=True)
class PairedDCSBenchmarkBatch:
    """Raw and marginalized contributions from exactly the same proposal paths."""

    raw_contribution: torch.Tensor
    dcs_contribution: torch.Tensor
    likelihood_normalization: torch.Tensor
    log_likelihood: torch.Tensor
    component_counts: tuple[int, ...]
    raw_cost: BaselineCostLedger
    dcs_cost: BaselineCostLedger
    maximum_path_reconstruction_error: float
    maximum_component_density_error: float
    maximum_mixture_density_error: float
    maximum_full_likelihood_error: float

    def __post_init__(self) -> None:
        vectors = (
            self.raw_contribution,
            self.dcs_contribution,
            self.likelihood_normalization,
            self.log_likelihood,
        )
        count = self.raw_contribution.numel()
        if count < 2 or any(value.shape != (count,) for value in vectors):
            raise ValueError("paired DCS vectors must be matching nontrivial vectors")
        if any(
            value.device.type != "cpu"
            or value.dtype != torch.float64
            or not torch.isfinite(value).all()
            for value in vectors
        ):
            raise ValueError("paired DCS vectors must be finite CPU float64")
        if bool((self.raw_contribution < 0.0).any()) or bool((self.dcs_contribution < 0.0).any()):
            raise ValueError("probability contributions must be nonnegative")
        if bool((self.likelihood_normalization <= 0.0).any()):
            raise ValueError("likelihood normalization must be strictly positive")
        if sum(self.component_counts) != count or any(value < 0 for value in self.component_counts):
            raise ValueError("component occupancy does not conserve the path count")
        errors = (
            self.maximum_path_reconstruction_error,
            self.maximum_component_density_error,
            self.maximum_mixture_density_error,
            self.maximum_full_likelihood_error,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in errors):
            raise ValueError("DCS exactness errors must be finite and nonnegative")


def evaluate_paired_dcs_benchmark(
    *,
    task: DCSBenchmarkTask,
    task_id: str,
    spot: float,
    maturity: float,
    steps: int,
    hurst: float,
    eta: float,
    xi: float,
    rho: float,
    controls: Sequence[RBergomiControl],
    weights: torch.Tensor,
    sample_count: int,
    path_seed: int,
    label_seed: int,
) -> PairedDCSBenchmarkBatch:
    """Draw a frozen mixture once and evaluate raw and DCS ordinary means.

    The ``task_id`` is required to make accidental cross-cell reuse visible at
    the call site.  Seeds for Gaussian paths and mixture labels are distinct.
    """

    if not task_id.strip():
        raise ValueError("task_id must be nonempty")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 2:
        raise ValueError("sample_count must be an integer of at least two")
    seeds = (path_seed, label_seed)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in seeds):
        raise ValueError("DCS seeds must be nonnegative integers")
    if path_seed == label_seed:
        raise ValueError("path and mixture-label seeds must be disjoint")
    if not controls:
        raise ValueError("DCS benchmark requires at least one proposal component")
    if (
        weights.shape != (len(controls),)
        or weights.device.type != "cpu"
        or weights.dtype != torch.float64
        or not torch.isfinite(weights).all()
        or bool((weights <= 0.0).any())
        or not math.isclose(float(torch.sum(weights)), 1.0, rel_tol=1e-12, abs_tol=1e-12)
    ):
        raise ValueError("mixture weights must be positive normalized CPU float64")
    simulator = RBergomiSimulator(H=hurst, eta=eta, xi=xi, rho=rho, device="cpu")

    simulation_wall_started = time.perf_counter()
    simulation_cpu_started = time.process_time()
    torch.manual_seed(path_seed)
    sample = simulate_rbergomi_mixture(
        simulator,
        controls,
        weights,
        spot=spot,
        maturity=maturity,
        dt=maturity / steps,
        num_paths=sample_count,
        dtype=torch.float64,
        label_generator=torch.Generator(device="cpu").manual_seed(label_seed),
        engine="fft",
    )
    simulation_cpu = time.process_time() - simulation_cpu_started
    simulation_wall = time.perf_counter() - simulation_wall_started

    raw_wall_started = time.perf_counter()
    raw_cpu_started = time.process_time()
    event = task.hard_event_from_log_spot(sample.paths.log_spot, sample.paths.step_dt)
    normalization = torch.exp(sample.mixture_log_likelihood)
    raw = event.to(torch.float64) * normalization
    raw_cpu = time.process_time() - raw_cpu_started
    raw_wall = time.perf_counter() - raw_wall_started

    dcs_wall_started = time.perf_counter()
    dcs_cpu_started = time.process_time()
    evaluation = evaluate_rbergomi_dcs_level(sample, task=task, rho=rho)
    dcs = evaluation.marginalized_contribution
    dcs_cpu = time.process_time() - dcs_cpu_started
    dcs_wall = time.perf_counter() - dcs_wall_started
    if not torch.allclose(raw, evaluation.raw_contribution, rtol=1e-12, atol=1e-14):
        raise AssertionError("independent raw contribution disagrees with DCS evaluator")
    raw = evaluation.raw_contribution
    if (
        not torch.isfinite(sample.paths.log_spot).all()
        or not torch.isfinite(sample.paths.variance).all()
    ):
        raise FloatingPointError("DCS benchmark path state became nonfinite")

    components = len(controls)
    occupancy = tuple(int(torch.sum(sample.labels == index)) for index in range(components))
    peak = process_peak_resident_memory_bytes()
    simulation_work = sample_count * (3 * steps + 2 * steps * components)
    density_work = sample_count * components
    raw_work = simulation_work + density_work + sample_count
    dcs_work = simulation_work + density_work + sample_count * (steps + 1)
    raw_cost = BaselineCostLedger(
        final_samples=sample_count,
        likelihood_evaluations=sample_count * components,
        algorithmic_work_units=float(raw_work),
        wall_seconds=simulation_wall + raw_wall,
        cpu_seconds=simulation_cpu + raw_cpu,
        peak_memory_bytes=peak,
        measurement_mode="standardized_hardware_wall",
    )
    dcs_cost = BaselineCostLedger(
        final_samples=sample_count,
        likelihood_evaluations=sample_count * components,
        cdf_calls=sample_count,
        algorithmic_work_units=float(dcs_work),
        wall_seconds=simulation_wall + dcs_wall,
        cpu_seconds=simulation_cpu + dcs_cpu,
        peak_memory_bytes=peak,
        measurement_mode="standardized_hardware_wall",
    )
    return PairedDCSBenchmarkBatch(
        raw_contribution=raw,
        dcs_contribution=dcs,
        likelihood_normalization=normalization,
        log_likelihood=sample.mixture_log_likelihood,
        component_counts=occupancy,
        raw_cost=raw_cost,
        dcs_cost=dcs_cost,
        maximum_path_reconstruction_error=evaluation.maximum_path_reconstruction_error,
        maximum_component_density_error=evaluation.maximum_legacy_component_density_error,
        maximum_mixture_density_error=evaluation.maximum_legacy_mixture_density_error,
        maximum_full_likelihood_error=evaluation.maximum_legacy_full_likelihood_error,
    )
