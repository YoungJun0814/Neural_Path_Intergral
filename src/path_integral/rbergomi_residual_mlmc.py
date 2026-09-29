"""Exact signed residual transports for adjacent finite-grid rBergomi corrections."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch

from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.mlmc import LevelBatch
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.rbergomi_dcs_mlmc import scalar_task_threshold
from src.path_integral.rbergomi_fft import (
    AdjacentRBergomiFFTInnovations,
    simulate_coupled_rbergomi_adjacent_fft,
)
from src.path_integral.rbergomi_residual_transport import (
    evaluate_rbergomi_residual_transport,
)
from src.path_integral.rbergomi_smoothing import affine_rbergomi_log_spot
from src.path_integral.residual_transport import (
    FrozenResidualTransport,
    ResidualTransportTrainingConfig,
    evaluate_residual_likelihood,
    evaluate_signed_residual_contribution,
    fit_weighted_residual_transport,
    project_orthogonal,
    sample_residual_mixture,
)
from src.path_integral.stable_gaussian import signed_log_normal_cdf_difference


@dataclass(frozen=True)
class RBergomiAdjacentResidualProblem:
    task_id: str
    task: TerminalThresholdTask
    spot: float
    maturity: float
    fine_steps: int
    hurst: float
    eta: float
    xi: float
    rho: float

    def __post_init__(self) -> None:
        if self.fine_steps < 2 or self.fine_steps % 2:
            raise ValueError("fine steps must be an even integer at least two")
        # Reuse the single-grid contract for all model-parameter validation.
        self.fine_problem()

    @property
    def coarse_steps(self) -> int:
        return self.fine_steps // 2

    @property
    def local_dimension(self) -> int:
        return 5 * self.coarse_steps

    @property
    def latent_dimension(self) -> int:
        return self.local_dimension + self.fine_steps

    def fine_problem(self) -> RBergomiBaselineProblem:
        return RBergomiBaselineProblem(
            task_id=self.task_id,
            task=self.task,
            spot=self.spot,
            maturity=self.maturity,
            steps=self.fine_steps,
            hurst=self.hurst,
            eta=self.eta,
            xi=self.xi,
            rho=self.rho,
        )

    def innovations(self, latent: torch.Tensor) -> AdjacentRBergomiFFTInnovations:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dimension:
            raise ValueError("adjacent latent array has the wrong shape")
        first_end = 3 * self.coarse_steps
        second_end = self.local_dimension
        return AdjacentRBergomiFFTInnovations(
            first_cell_standard_normal=latent[:, :first_end].reshape(
                -1, self.coarse_steps, 3
            ),
            second_cell_standard_normal=latent[:, first_end:second_end].reshape(
                -1, self.coarse_steps, 2
            ),
            price_standard_normal=latent[:, second_end:],
        )

    def simulate(self, latent: torch.Tensor):
        return simulate_coupled_rbergomi_adjacent_fft(
            self.fine_problem().simulator(),
            S0=self.spot,
            T=self.maturity,
            fine_steps=self.fine_steps,
            num_paths=latent.shape[0],
            innovations=self.innovations(latent),
            dtype=torch.float64,
        )


def adjacent_positive_price_direction(
    problem: RBergomiAdjacentResidualProblem,
    price_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    weights = (
        torch.ones(problem.fine_steps, dtype=torch.float64)
        if price_weights is None
        else price_weights
    )
    if weights.shape != (problem.fine_steps,) or bool((weights <= 0.0).any()):
        raise ValueError("adjacent price weights must be strictly positive")
    direction = torch.zeros(problem.latent_dimension, dtype=torch.float64)
    direction[problem.local_dimension :] = weights / torch.linalg.vector_norm(weights)
    return direction


@dataclass(frozen=True)
class AdjacentConditionalResidualBatch:
    fine_threshold: torch.Tensor
    coarse_threshold: torch.Tensor
    sign: torch.Tensor
    log_absolute_correction: torch.Tensor
    conditional_correction: torch.Tensor
    maximum_coordinate_mismatch: float
    maximum_path_reconstruction_error: float
    hard_correction_mismatch_count: int


def evaluate_adjacent_conditional_residual(
    problem: RBergomiAdjacentResidualProblem,
    residual: torch.Tensor,
    direction: torch.Tensor,
) -> AdjacentConditionalResidualBatch:
    if direction.shape != (problem.latent_dimension,):
        raise ValueError("adjacent direction has the wrong shape")
    if float(torch.amax(torch.abs(direction[: problem.local_dimension]))) > 1e-12:
        raise ValueError("adjacent integrated direction may enter only the price block")
    if bool((direction[problem.local_dimension :] <= 0.0).any()):
        raise ValueError("adjacent integrated price direction must be positive")
    if abs(float(torch.linalg.vector_norm(direction)) - 1.0) > 1e-10:
        raise ValueError("adjacent direction must be unit norm")
    projection = float(torch.amax(torch.abs(residual @ direction)))
    if projection > 1e-10 * max(1.0, float(torch.amax(torch.abs(residual)))):
        raise ValueError("adjacent residual is not orthogonal")
    paths = problem.simulate(residual)
    target = paths.target_fine_brownian_increments
    price_direction = direction[problem.local_dimension :]
    fine = affine_rbergomi_log_spot(
        spot=paths.fine.spot,
        log_spot=paths.fine.log_spot,
        variance=paths.fine.variance,
        proposal_fine_brownian_increments=target,
        fine_step_dt=paths.fine.step_dt,
        rho=problem.rho,
        direction=price_direction,
    )
    coarse = affine_rbergomi_log_spot(
        spot=paths.coarse.spot,
        log_spot=paths.coarse.log_spot,
        variance=paths.coarse.variance,
        proposal_fine_brownian_increments=target,
        fine_step_dt=paths.fine.step_dt,
        rho=problem.rho,
        direction=price_direction,
        coarse_from_fine_pairs=True,
    )
    fine_threshold = scalar_task_threshold(
        fine.intercept, fine.slope, step_dt=paths.fine.step_dt, task=problem.task
    )
    coarse_threshold = scalar_task_threshold(
        coarse.intercept,
        coarse.slope,
        step_dt=paths.coarse.step_dt,
        task=problem.task,
    )
    coordinate_mismatch = float(torch.amax(torch.abs(fine.coordinate - coarse.coordinate)))
    if coordinate_mismatch > 2e-12:
        raise AssertionError("fine and coarse integrated coordinates differ")
    sign, log_absolute = signed_log_normal_cdf_difference(
        fine_threshold, coarse_threshold
    )
    correction = sign * torch.exp(log_absolute)
    hard_fine = problem.task.hard_event_from_log_spot(
        paths.fine.log_spot, paths.fine.step_dt
    )
    hard_coarse = problem.task.hard_event_from_log_spot(
        paths.coarse.log_spot, paths.coarse.step_dt
    )
    hard_correction = hard_fine.to(torch.float64) - hard_coarse.to(torch.float64)
    threshold_correction = (fine.coordinate <= fine_threshold).to(torch.float64) - (
        coarse.coordinate <= coarse_threshold
    ).to(torch.float64)
    mismatch = int(torch.count_nonzero(hard_correction != threshold_correction))
    if mismatch:
        raise AssertionError("adjacent scalar thresholds are not pathwise exact")
    return AdjacentConditionalResidualBatch(
        fine_threshold=fine_threshold,
        coarse_threshold=coarse_threshold,
        sign=sign,
        log_absolute_correction=log_absolute,
        conditional_correction=correction,
        maximum_coordinate_mismatch=coordinate_mismatch,
        maximum_path_reconstruction_error=max(
            fine.maximum_path_reconstruction_error,
            coarse.maximum_path_reconstruction_error,
        ),
        hard_correction_mismatch_count=mismatch,
    )


@dataclass(frozen=True)
class AdjacentResidualEvaluationBatch:
    raw_correction: torch.Tensor
    residual_correction: torch.Tensor
    likelihood_normalization: torch.Tensor
    maximum_coordinate_mismatch: float
    maximum_path_reconstruction_error: float
    hard_correction_mismatch_count: int


def train_adjacent_residual_transport(
    problem: RBergomiAdjacentResidualProblem,
    *,
    sample_count: int,
    training_seed: int,
    config: ResidualTransportTrainingConfig | None = None,
) -> FrozenResidualTransport:
    direction = adjacent_positive_price_direction(problem)
    target = torch.randn(
        (sample_count, problem.latent_dimension),
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(training_seed),
    )
    residual = project_orthogonal(target, direction)
    conditional = evaluate_adjacent_conditional_residual(problem, residual, direction)
    # The signed target is not a probability density.  The variance-oriented
    # proposal is trained against |G| p; the sign is restored only at evaluation.
    fitted = fit_weighted_residual_transport(
        task_id=problem.task_id,
        target_residuals=residual,
        log_target_weights=conditional.log_absolute_correction,
        direction=direction,
        training_seed=training_seed + 1,
        config=config,
    )
    return fitted.proposal


def evaluate_adjacent_residual_transport(
    problem: RBergomiAdjacentResidualProblem,
    proposal: FrozenResidualTransport,
    *,
    sample_count: int,
    gaussian_seed: int,
    label_seed: int,
    coordinate_seed: int,
) -> AdjacentResidualEvaluationBatch:
    spec = proposal.spec()
    sample = sample_residual_mixture(
        spec,
        sample_count,
        gaussian_generator=torch.Generator().manual_seed(gaussian_seed),
        label_generator=torch.Generator().manual_seed(label_seed),
    )
    conditional = evaluate_adjacent_conditional_residual(
        problem, sample.residual, spec.direction
    )
    signed = evaluate_signed_residual_contribution(
        sample.residual,
        spec,
        sign=conditional.sign,
        log_absolute_conditional_value=conditional.log_absolute_correction,
    )
    coordinate = torch.randn(
        sample_count,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(coordinate_seed),
    )
    fine = coordinate <= conditional.fine_threshold
    coarse = coordinate <= conditional.coarse_threshold
    likelihood = evaluate_residual_likelihood(sample.residual, spec).likelihood
    raw = (fine.to(torch.float64) - coarse.to(torch.float64)) * likelihood
    return AdjacentResidualEvaluationBatch(
        raw_correction=raw,
        residual_correction=signed.contribution,
        likelihood_normalization=likelihood,
        maximum_coordinate_mismatch=conditional.maximum_coordinate_mismatch,
        maximum_path_reconstruction_error=conditional.maximum_path_reconstruction_error,
        hard_correction_mismatch_count=conditional.hard_correction_mismatch_count,
    )


class RBergomiResidualMLMCSampler:
    """Adapt exact residual level terms to the repository MLMC engine.

    Pass ``streams=("gaussian", "labels", "coordinate")`` to ``prepare_mlmc``
    and ``run_prepared_mlmc``.  Level zero uses the single-grid nonnegative
    estimator; each positive level uses one signed adjacent correction.
    """

    def __init__(
        self,
        *,
        level_zero_problem: RBergomiBaselineProblem,
        level_zero_proposal: FrozenResidualTransport,
        adjacent_levels: Mapping[
            int, tuple[RBergomiAdjacentResidualProblem, FrozenResidualTransport]
        ],
    ) -> None:
        if not adjacent_levels or set(adjacent_levels) != set(
            range(1, max(adjacent_levels) + 1)
        ):
            raise ValueError("adjacent residual levels must be contiguous from one")
        self.level_zero_problem = level_zero_problem
        self.level_zero_proposal = level_zero_proposal
        self.adjacent_levels = dict(adjacent_levels)

    def __call__(
        self,
        level: int,
        role: Literal["pilot", "final"],
        count: int,
        seeds: Mapping[str, int],
    ) -> LevelBatch:
        if role not in {"pilot", "final"} or count < 1:
            raise ValueError("invalid residual MLMC sampling request")
        if set(seeds) != {"gaussian", "labels", "coordinate"}:
            raise ValueError("residual MLMC requires Gaussian, label, and coordinate streams")
        started = time.perf_counter()
        if level == 0:
            evaluated = evaluate_rbergomi_residual_transport(
                self.level_zero_problem,
                self.level_zero_proposal,
                sample_count=count,
                gaussian_seed=seeds["gaussian"],
                label_seed=seeds["labels"],
                coordinate_seed=seeds["coordinate"],
                reconstruction_paths=min(16, count),
            )
            values = evaluated.ecrpt_contribution
            work = evaluated.ecrpt_cost.algorithmic_work_units
        else:
            if level not in self.adjacent_levels:
                raise ValueError("residual MLMC level is unavailable")
            problem, proposal = self.adjacent_levels[level]
            evaluated_adjacent = evaluate_adjacent_residual_transport(
                problem,
                proposal,
                sample_count=count,
                gaussian_seed=seeds["gaussian"],
                label_seed=seeds["labels"],
                coordinate_seed=seeds["coordinate"],
            )
            values = evaluated_adjacent.residual_correction
            work = count * (
                problem.latent_dimension * (proposal.components + 1)
                + problem.fine_steps
                + 1
            )
        return LevelBatch(
            values=values.detach().clone(),
            work_units=float(work),
            wall_seconds=time.perf_counter() - started,
        )
