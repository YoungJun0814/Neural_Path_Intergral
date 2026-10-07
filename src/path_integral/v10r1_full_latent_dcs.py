"""Corrected full-latent defensive CEM plus exact scalar marginalization.

Unlike the exploratory V10 adapter, this module never converts the ``3 * N``
BLP latent mean into a ``2 * N`` Brownian control.  It samples and evaluates the
frozen Gaussian mixture in the exact latent coordinates consumed by
``RBergomiBaselineProblem.simulate_latent``.  Only one positive direction in
the independent price block is integrated out; all ``2 * N`` local coordinates
and the remaining ``N - 1`` price coordinates stay in the residual likelihood.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from src.path_integral.baseline_framework import (
    BaselineCostLedger,
    FrozenBaselineProposal,
    evaluate_baseline_log_q_over_p,
)
from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.gaussian_smoothing import scaled_normal_cdf
from src.path_integral.gaussian_span_marginalization import (
    GaussianMixtureShiftSpec,
    MarginalLikelihoodEvaluation,
    build_orthonormal_control_span,
    evaluate_marginal_likelihood,
    sample_gaussian_mixture,
)
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.provenance import process_peak_resident_memory_bytes
from src.path_integral.rbergomi_fft import blp_fft_kernel
from src.path_integral.rbergomi_smoothing import affine_rbergomi_log_spot


@dataclass(frozen=True)
class FullLatentDCSBatch:
    """Paired raw and Rao--Blackwellized contributions on one exact proposal."""

    raw_contribution: torch.Tensor
    dcs_contribution: torch.Tensor
    likelihood_normalization: torch.Tensor
    log_likelihood: torch.Tensor
    labels: torch.Tensor
    component_counts: tuple[int, ...]
    integration_direction: torch.Tensor
    raw_cost: BaselineCostLedger
    dcs_cost: BaselineCostLedger
    maximum_local_latent_reconstruction_error: float
    maximum_price_latent_reconstruction_error: float
    maximum_path_reconstruction_error: float
    maximum_coordinate_error: float
    maximum_component_density_error: float
    maximum_mixture_density_error: float
    maximum_full_likelihood_error: float
    maximum_full_bound_violation: float
    maximum_residual_bound_violation: float

    def __post_init__(self) -> None:
        count = int(self.raw_contribution.numel())
        vectors = (
            self.raw_contribution,
            self.dcs_contribution,
            self.likelihood_normalization,
            self.log_likelihood,
        )
        if count < 2 or any(value.shape != (count,) for value in vectors):
            raise ValueError("full-latent DCS vectors must be matching nontrivial vectors")
        if any(
            value.device.type != "cpu"
            or value.dtype != torch.float64
            or not torch.isfinite(value).all()
            for value in vectors
        ):
            raise ValueError("full-latent DCS vectors must be finite CPU float64")
        if bool((self.raw_contribution < 0.0).any()) or bool(
            (self.dcs_contribution < 0.0).any()
        ):
            raise ValueError("probability contributions must be nonnegative")
        if self.labels.shape != (count,) or self.labels.dtype != torch.long:
            raise ValueError("mixture labels must be one integer per path")
        if sum(self.component_counts) != count or any(x < 0 for x in self.component_counts):
            raise ValueError("component counts must conserve the path count")
        if (
            self.integration_direction.ndim != 1
            or self.integration_direction.dtype != torch.float64
            or not torch.isfinite(self.integration_direction).all()
            or bool((self.integration_direction <= 0.0).any())
            or not math.isclose(
                float(torch.linalg.vector_norm(self.integration_direction)),
                1.0,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("integration direction must be a positive float64 unit vector")
        errors = (
            self.maximum_local_latent_reconstruction_error,
            self.maximum_price_latent_reconstruction_error,
            self.maximum_path_reconstruction_error,
            self.maximum_coordinate_error,
            self.maximum_component_density_error,
            self.maximum_mixture_density_error,
            self.maximum_full_likelihood_error,
            self.maximum_full_bound_violation,
            self.maximum_residual_bound_violation,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in errors):
            raise ValueError("full-latent exactness diagnostics must be finite and nonnegative")


def _validate_proposal(
    problem: RBergomiBaselineProblem, proposal: FrozenBaselineProposal
) -> None:
    if not isinstance(problem.task, TerminalThresholdTask):
        raise ValueError("V10R1 is frozen to finite-grid terminal threshold tasks")
    if proposal.method != "defensive_cem" or proposal.family != "gaussian_mixture_shift":
        raise ValueError("V10R1 requires a frozen defensive CEM Gaussian mixture")
    if proposal.task_id != problem.task_id:
        raise ValueError("proposal and rBergomi task IDs differ")
    if proposal.dimension != problem.latent_dimension:
        raise ValueError("proposal must retain the complete 3N rBergomi latent dimension")
    if len(proposal.component_means) != 2 or len(proposal.component_weights) != 2:
        raise ValueError("V10R1 requires exactly one natural and one shifted component")
    if any(value != 0.0 for value in proposal.component_means[0]):
        raise ValueError("the first defensive component must be exactly the target law")
    if not proposal.exact_likelihood or proposal.self_normalized:
        raise ValueError("V10R1 requires exact ordinary importance sampling")


def proposal_spec(proposal: FrozenBaselineProposal) -> GaussianMixtureShiftSpec:
    """Return the exact full-dimensional Gaussian-mixture density specification."""

    return GaussianMixtureShiftSpec(
        means=torch.tensor(proposal.component_means, dtype=torch.float64),
        weights=torch.tensor(proposal.component_weights, dtype=torch.float64),
    )


def positive_price_direction(
    proposal: FrozenBaselineProposal, *, steps: int, floor_scale: float = 1e-12
) -> torch.Tensor:
    """Choose a fixed positive price direction without altering the proposal.

    Absolute shifted-price magnitudes preserve the temporal emphasis learned by
    CEM while positivity makes every terminal log-price slope strictly positive.
    A tiny deterministic floor handles exactly-zero coordinates and has no role
    in the proposal density.
    """

    if not math.isfinite(floor_scale) or floor_scale <= 0.0:
        raise ValueError("direction floor scale must be finite and positive")
    if proposal.dimension != 3 * steps:
        raise ValueError("proposal dimension must equal 3 * steps")
    shifted = torch.tensor(proposal.component_means[1], dtype=torch.float64)
    magnitude = torch.abs(shifted[2 * steps :])
    scale = max(1.0, float(torch.linalg.vector_norm(magnitude)))
    direction = magnitude + floor_scale * scale
    return direction / torch.linalg.vector_norm(direction)


def full_latent_integration_basis(
    *, steps: int, direction: torch.Tensor
) -> torch.Tensor:
    """Embed one price direction in the complete ``3N`` latent coordinate system."""

    if direction.shape != (steps,):
        raise ValueError("price direction must have shape (steps,)")
    basis = torch.zeros((3 * steps, 1), dtype=direction.dtype, device=direction.device)
    basis[2 * steps :, 0] = direction
    return basis


def _latent_reconstruction_errors(
    problem: RBergomiBaselineProblem,
    latent: torch.Tensor,
    *,
    target_brownian: torch.Tensor,
    target_local_integral: torch.Tensor,
) -> tuple[float, float]:
    """Recover all independent BLP coordinates from the simulated path augmentation."""

    kernel = blp_fft_kernel(
        problem.simulator(),
        n_steps=problem.steps,
        step_dt=problem.step_dt,
        H=problem.hurst,
        dtype=torch.float64,
    )
    local_pair = torch.stack((target_brownian[:, :, 0], target_local_integral), dim=2)
    recovered_local = local_pair @ torch.linalg.inv(kernel.local_cholesky.T)
    expected_local = latent[:, : problem.local_dimension].reshape(-1, problem.steps, 2)
    recovered_price = target_brownian[:, :, 1] / math.sqrt(problem.step_dt)
    expected_price = latent[:, problem.local_dimension :]
    return (
        float(torch.max(torch.abs(recovered_local - expected_local))),
        float(torch.max(torch.abs(recovered_price - expected_price))),
    )


def _cost(
    *,
    problem: RBergomiBaselineProblem,
    sample_count: int,
    likelihood_passes: int,
    cdf_calls: int,
    wall_seconds: float,
    cpu_seconds: float,
) -> BaselineCostLedger:
    """Use the repository-wide baseline work convention with explicit density passes."""

    base_work = sample_count * (problem.latent_dimension + problem.steps)
    density_work = likelihood_passes * sample_count * problem.latent_dimension
    return BaselineCostLedger(
        final_samples=sample_count,
        likelihood_evaluations=likelihood_passes * sample_count,
        cdf_calls=cdf_calls,
        algorithmic_work_units=float(base_work + density_work + cdf_calls),
        wall_seconds=wall_seconds,
        cpu_seconds=cpu_seconds,
        peak_memory_bytes=process_peak_resident_memory_bytes(),
        measurement_mode="standardized_hardware_wall",
    )


def evaluate_full_latent_dcs(
    problem: RBergomiBaselineProblem,
    proposal: FrozenBaselineProposal,
    *,
    sample_count: int,
    gaussian_seed: int,
    label_seed: int,
    tolerance: float = 1e-10,
) -> FullLatentDCSBatch:
    """Evaluate paired raw/DCS ordinary means on the exact frozen ``3N`` mixture."""

    _validate_proposal(problem, proposal)
    task = problem.task
    if not isinstance(task, TerminalThresholdTask):
        raise AssertionError("proposal validation did not retain the terminal-task contract")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 2:
        raise ValueError("sample_count must be an integer of at least two")
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        for seed in (gaussian_seed, label_seed)
    ):
        raise ValueError("sampling seeds must be nonnegative integers")
    if gaussian_seed == label_seed:
        raise ValueError("Gaussian and label seeds must be disjoint")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")

    spec = proposal_spec(proposal)
    direction = positive_price_direction(proposal, steps=problem.steps)
    basis = full_latent_integration_basis(steps=problem.steps, direction=direction)
    span = build_orthonormal_control_span(spec, basis, tolerance=tolerance)

    simulation_wall = time.perf_counter()
    simulation_cpu = time.process_time()
    sample = sample_gaussian_mixture(
        spec,
        sample_count,
        gaussian_generator=torch.Generator(device="cpu").manual_seed(gaussian_seed),
        label_generator=torch.Generator(device="cpu").manual_seed(label_seed),
    )
    latent = sample.target_coordinates
    paths = problem.simulate_latent(latent)
    simulation_cpu = time.process_time() - simulation_cpu
    simulation_wall = time.perf_counter() - simulation_wall
    if paths.target_brownian_increments is None or paths.target_local_integrals is None:
        raise ValueError("full-latent simulation did not retain augmented coordinates")
    local_error, price_error = _latent_reconstruction_errors(
        problem,
        latent,
        target_brownian=paths.target_brownian_increments,
        target_local_integral=paths.target_local_integrals,
    )

    affine = affine_rbergomi_log_spot(
        spot=paths.spot,
        log_spot=paths.log_spot,
        variance=paths.variance,
        proposal_fine_brownian_increments=paths.target_brownian_increments,
        fine_step_dt=problem.step_dt,
        rho=problem.rho,
        direction=direction,
    )
    threshold = (math.log(task.level) - affine.intercept[:, -1]) / affine.slope[:, -1]
    hard_event = problem.hard_event(paths)
    if not torch.equal(hard_event, affine.coordinate <= threshold):
        raise AssertionError("full-latent scalar threshold is not pathwise exact")

    raw_wall = time.perf_counter()
    raw_cpu = time.process_time()
    direct_log_q_over_p = evaluate_baseline_log_q_over_p(latent, proposal)
    direct_log_likelihood = -direct_log_q_over_p
    direct_likelihood = torch.exp(direct_log_likelihood)
    raw = hard_event.to(torch.float64) * direct_likelihood
    raw_cpu = time.process_time() - raw_cpu
    raw_wall = time.perf_counter() - raw_wall

    dcs_wall = time.perf_counter()
    dcs_cpu = time.process_time()
    density: MarginalLikelihoodEvaluation = evaluate_marginal_likelihood(
        latent, spec, span, tolerance=tolerance
    )
    dcs = scaled_normal_cdf(density.residual_log_likelihood, threshold)
    dcs_cpu = time.process_time() - dcs_cpu
    dcs_wall = time.perf_counter() - dcs_wall
    if not torch.isfinite(raw).all() or not torch.isfinite(dcs).all():
        raise FloatingPointError("full-latent contribution became nonfinite")

    coordinate_error = float(torch.max(torch.abs(affine.coordinate - density.coordinate[:, 0])))
    component_counts = tuple(
        int(torch.count_nonzero(sample.labels == index)) for index in range(spec.components)
    )
    raw_cost = _cost(
        problem=problem,
        sample_count=sample_count,
        likelihood_passes=1,
        cdf_calls=0,
        wall_seconds=simulation_wall + raw_wall,
        cpu_seconds=simulation_cpu + raw_cpu,
    )
    dcs_cost = _cost(
        problem=problem,
        sample_count=sample_count,
        likelihood_passes=2,
        cdf_calls=sample_count,
        wall_seconds=simulation_wall + dcs_wall,
        cpu_seconds=simulation_cpu + dcs_cpu,
    )
    return FullLatentDCSBatch(
        raw_contribution=raw,
        dcs_contribution=dcs,
        likelihood_normalization=direct_likelihood,
        log_likelihood=direct_log_likelihood,
        labels=sample.labels,
        component_counts=component_counts,
        integration_direction=direction,
        raw_cost=raw_cost,
        dcs_cost=dcs_cost,
        maximum_local_latent_reconstruction_error=local_error,
        maximum_price_latent_reconstruction_error=price_error,
        maximum_path_reconstruction_error=affine.maximum_path_reconstruction_error,
        maximum_coordinate_error=coordinate_error,
        maximum_component_density_error=density.maximum_component_reconstruction_error,
        maximum_mixture_density_error=density.maximum_mixture_reconstruction_error,
        maximum_full_likelihood_error=float(
            torch.max(torch.abs(density.full_log_likelihood - direct_log_likelihood))
        ),
        maximum_full_bound_violation=density.maximum_full_bound_violation,
        maximum_residual_bound_violation=density.maximum_residual_bound_violation,
    )
