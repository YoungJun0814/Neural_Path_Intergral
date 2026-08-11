"""Mesh audits for conditional Cameron--Martin rBergomi transport.

The adjacent BLP construction gives both finite-grid marginals exactly.  Results in
this module are empirical discretization diagnostics; they do not constitute a
continuous-time convergence theorem.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.rbergomi_residual_mlmc import RBergomiAdjacentResidualProblem


@dataclass(frozen=True)
class ConditionalMeshPairBatch:
    fine_probability: torch.Tensor
    coarse_probability: torch.Tensor
    correction: torch.Tensor
    fine_log_probability: torch.Tensor
    coarse_log_probability: torch.Tensor


@dataclass(frozen=True)
class ScalarMonteCarloSummary:
    mean: float
    variance: float
    standard_error: float
    sample_count: int


@dataclass(frozen=True)
class ConditionalMeshPairSummary:
    fine_steps: int
    coarse_steps: int
    fine: ScalarMonteCarloSummary
    coarse: ScalarMonteCarloSummary
    correction: ScalarMonteCarloSummary
    correlation: float


@dataclass(frozen=True)
class ConditionalMeshStudy:
    levels: tuple[ScalarMonteCarloSummary, ...]
    steps: tuple[int, ...]
    adjacent: tuple[ConditionalMeshPairSummary, ...]
    observed_correction_rate: float | None
    final_correction_signal_to_noise: float


def evaluate_adjacent_conditional_mesh_pair(
    problem: RBergomiAdjacentResidualProblem,
    local_standard_normal: torch.Tensor,
) -> ConditionalMeshPairBatch:
    """Integrate the complete independent price driver on an exact BLP pair."""

    if not isinstance(problem.task, TerminalThresholdTask):
        raise TypeError("V15 full price-driver conditionalization supports terminal tasks")
    if local_standard_normal.ndim != 2 or local_standard_normal.shape[1] != problem.local_dimension:
        raise ValueError("adjacent local standard normals have the wrong shape")
    if (
        local_standard_normal.device.type != "cpu"
        or local_standard_normal.dtype != torch.float64
        or not torch.isfinite(local_standard_normal).all()
    ):
        raise ValueError("adjacent local standard normals must be finite CPU float64")
    zeros = torch.zeros(
        (local_standard_normal.shape[0], problem.fine_steps),
        dtype=torch.float64,
    )
    paths = problem.simulate(torch.cat((local_standard_normal, zeros), dim=1))
    log_strike = math.log(problem.task.level)

    def conditional(
        log_spot: torch.Tensor,
        variance: torch.Tensor,
        step_dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        integrated_variance = step_dt * torch.sum(variance[:, :-1], dim=1)
        conditional_variance = (1.0 - problem.rho**2) * integrated_variance
        if torch.any(conditional_variance <= 0.0):
            raise FloatingPointError("conditional price variance must be positive")
        standardized = (log_strike - log_spot[:, -1]) / torch.sqrt(conditional_variance)
        log_probability = torch.special.log_ndtr(standardized)
        return torch.exp(log_probability), log_probability

    fine, fine_log = conditional(paths.fine.log_spot, paths.fine.variance, paths.fine.step_dt)
    coarse, coarse_log = conditional(
        paths.coarse.log_spot,
        paths.coarse.variance,
        paths.coarse.step_dt,
    )
    return ConditionalMeshPairBatch(
        fine_probability=fine,
        coarse_probability=coarse,
        correction=fine - coarse,
        fine_log_probability=fine_log,
        coarse_log_probability=coarse_log,
    )


def summarize_samples(values: torch.Tensor) -> ScalarMonteCarloSummary:
    if values.ndim != 1 or values.numel() < 2 or not torch.isfinite(values).all():
        raise ValueError("summary values must be a finite vector with at least two entries")
    variance = float(torch.var(values, unbiased=True))
    count = int(values.numel())
    return ScalarMonteCarloSummary(
        mean=float(torch.mean(values)),
        variance=variance,
        standard_error=math.sqrt(variance / count),
        sample_count=count,
    )


def run_conditional_mesh_study(
    template: RBergomiBaselineProblem,
    *,
    steps: tuple[int, ...],
    sample_count: int,
    seed: int,
    batch_size: int | None = None,
) -> ConditionalMeshStudy:
    if not isinstance(template.task, TerminalThresholdTask):
        raise TypeError("mesh study requires a terminal threshold task")
    if len(steps) < 2 or any(
        fine != 2 * coarse for coarse, fine in zip(steps[:-1], steps[1:], strict=True)
    ):
        raise ValueError("mesh steps must be a consecutive doubling hierarchy")
    if steps[0] < 1 or sample_count < 2:
        raise ValueError("mesh study requires positive steps and at least two samples")
    resolved_batch = sample_count if batch_size is None else batch_size
    if (
        isinstance(resolved_batch, bool)
        or not isinstance(resolved_batch, int)
        or resolved_batch < 2
    ):
        raise ValueError("mesh batch size must be an integer of at least two")
    adjacent: list[ConditionalMeshPairSummary] = []
    levels: list[ScalarMonteCarloSummary] = []
    for index, fine_steps in enumerate(steps[1:]):
        problem = RBergomiAdjacentResidualProblem(
            task_id=f"{template.task_id}-mesh-{fine_steps}",
            task=template.task,
            spot=template.spot,
            maturity=template.maturity,
            fine_steps=fine_steps,
            hurst=template.hurst,
            eta=template.eta,
            xi=template.xi,
            rho=template.rho,
        )
        generator = torch.Generator().manual_seed(seed + 104_729 * index)
        moments = torch.zeros(7, dtype=torch.float64)
        completed = 0
        while completed < sample_count:
            count = min(resolved_batch, sample_count - completed)
            local = torch.randn(
                (count, problem.local_dimension),
                dtype=torch.float64,
                generator=generator,
            )
            batch = evaluate_adjacent_conditional_mesh_pair(problem, local)
            fine = batch.fine_probability
            coarse = batch.coarse_probability
            correction = batch.correction
            moments += torch.stack(
                (
                    torch.sum(fine),
                    torch.sum(fine.square()),
                    torch.sum(coarse),
                    torch.sum(coarse.square()),
                    torch.sum(correction),
                    torch.sum(correction.square()),
                    torch.sum(fine * coarse),
                )
            )
            completed += count

        def moment_summary(total: torch.Tensor, square_total: torch.Tensor) -> ScalarMonteCarloSummary:
            mean = float(total / sample_count)
            variance = float((square_total - total.square() / sample_count) / (sample_count - 1))
            variance = max(0.0, variance)
            return ScalarMonteCarloSummary(
                mean=mean,
                variance=variance,
                standard_error=math.sqrt(variance / sample_count),
                sample_count=sample_count,
            )

        fine_summary = moment_summary(moments[0], moments[1])
        coarse_summary = moment_summary(moments[2], moments[3])
        correction_summary = moment_summary(moments[4], moments[5])
        covariance = float(
            (moments[6] - moments[0] * moments[2] / sample_count) / (sample_count - 1)
        )
        denominator = math.sqrt(fine_summary.variance * coarse_summary.variance)
        correlation = covariance / denominator
        adjacent.append(
            ConditionalMeshPairSummary(
                fine_steps=fine_steps,
                coarse_steps=fine_steps // 2,
                fine=fine_summary,
                coarse=coarse_summary,
                correction=correction_summary,
                correlation=correlation,
            )
        )
        if index == 0:
            levels.append(coarse_summary)
        levels.append(fine_summary)
    absolute_means = torch.tensor(
        [abs(item.correction.mean) for item in adjacent],
        dtype=torch.float64,
    )
    observed_rate = None
    positive = absolute_means > 0.0
    if int(torch.count_nonzero(positive)) >= 2:
        x = torch.log2(torch.tensor([item.fine_steps for item in adjacent], dtype=torch.float64))
        x = x[positive]
        y = torch.log2(absolute_means[positive])
        slope = torch.dot(x - torch.mean(x), y - torch.mean(y)) / torch.sum(
            (x - torch.mean(x)).square()
        )
        observed_rate = -float(slope)
    final = adjacent[-1].correction
    signal_to_noise = abs(final.mean) / max(final.standard_error, torch.finfo(torch.float64).tiny)
    return ConditionalMeshStudy(
        levels=tuple(levels),
        steps=steps,
        adjacent=tuple(adjacent),
        observed_correction_rate=observed_rate,
        final_correction_signal_to_noise=signal_to_noise,
    )


def evaluate_dct_cameron_martin_drift(
    coefficients: torch.Tensor,
    *,
    steps: int,
    maturity: float,
) -> torch.Tensor:
    """Return the piecewise drift represented by stable DCT coefficients."""

    if coefficients.ndim != 2 or coefficients.shape[0] < 1:
        raise ValueError("coefficients must have shape (drivers, modes)")
    if coefficients.device.type != "cpu" or coefficients.dtype != torch.float64:
        raise ValueError("coefficients must be CPU float64")
    drivers, modes = coefficients.shape
    if modes > steps or maturity <= 0.0:
        raise ValueError("invalid drift grid")
    from src.path_integral.cameron_martin_basis import build_blp_cameron_martin_basis

    basis = build_blp_cameron_martin_basis(
        steps=steps,
        modes_per_driver=modes,
        drivers=drivers,
    )
    flattened = coefficients.reshape(-1)
    standardized_shift = basis.expand(flattened).reshape(steps, drivers)
    return standardized_shift / math.sqrt(maturity / steps)
