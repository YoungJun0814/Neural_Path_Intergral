"""Exact rank-one control-span marginalization for defensive rBergomi mixtures."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

from src.path_integral.gaussian_smoothing import (
    downside_excursion_thresholds,
    positive_exponential_direction,
    scaled_normal_cdf,
    scaled_normal_cdf_difference,
    validate_positive_unit_direction,
)
from src.path_integral.mixture import log_mixture_q_over_p
from src.path_integral.path_functionals import DownsideExcursionTask
from src.path_integral.rbergomi_mixture import RBergomiMixtureSample
from src.path_integral.rbergomi_multilevel import CoupledRBergomiMixtureSample
from src.path_integral.rbergomi_smoothing import affine_rbergomi_log_spot


@dataclass(frozen=True)
class RankOneControlSpan:
    """Oriented price-control span and expert coordinates in that span."""

    direction: torch.Tensor
    expert_coefficients: torch.Tensor
    maximum_span_residual: float


@dataclass(frozen=True)
class ControlSpanMarginalizedEstimate:
    """Raw balance-mixture and exact rank-one marginalized contributions."""

    threshold: torch.Tensor
    coordinate: torch.Tensor
    hard_event: torch.Tensor
    threshold_event: torch.Tensor
    raw_mixture_contribution: torch.Tensor
    marginalized_contribution: torch.Tensor
    log_outer_likelihood: torch.Tensor
    outer_likelihood: torch.Tensor
    span: RankOneControlSpan
    defensive_weight: float | None
    maximum_defensive_bound_violation: float
    maximum_component_log_density_error: float
    maximum_mixture_log_density_error: float
    maximum_path_reconstruction_error: float
    maximum_residual_projection: float


@dataclass(frozen=True)
class ControlSpanMarginalizedAdjacentEstimate:
    """Exact fine-minus-coarse DCS-MGI contribution on one coupled sample."""

    fine_threshold: torch.Tensor
    coarse_threshold: torch.Tensor
    raw_correction: torch.Tensor
    marginalized_correction: torch.Tensor
    log_outer_likelihood: torch.Tensor
    outer_likelihood: torch.Tensor
    span: RankOneControlSpan
    defensive_weight: float | None
    maximum_defensive_bound_violation: float
    maximum_component_log_density_error: float
    maximum_mixture_log_density_error: float
    maximum_coordinate_mismatch: float
    maximum_fine_path_reconstruction_error: float
    maximum_coarse_path_reconstruction_error: float
    maximum_residual_projection: float


@dataclass(frozen=True)
class PositiveRankTwoSubspace:
    """Two positive unit vectors and their oblique Gaussian coordinate law."""

    basis: torch.Tensor
    gram: torch.Tensor
    coordinate_covariance: torch.Tensor
    event_direction_decay: float
    minimum_gram_eigenvalue: float


@dataclass(frozen=True)
class RankTwoControlSpanMarginalizedEstimate:
    """Unbiased nested rank-two control/event-subspace estimate."""

    raw_mixture_contribution: torch.Tensor
    marginalized_contribution: torch.Tensor
    conditional_event_probability: torch.Tensor
    log_outer_likelihood: torch.Tensor
    outer_likelihood: torch.Tensor
    control_span: RankOneControlSpan
    subspace: PositiveRankTwoSubspace
    inner_samples: int
    inner_rule: str
    defensive_weight: float | None
    maximum_defensive_bound_violation: float
    maximum_component_log_density_error: float
    maximum_mixture_log_density_error: float
    maximum_path_reconstruction_error: float
    maximum_residual_projection: float


def rank_one_price_control_span(
    schedules: torch.Tensor,
    *,
    step_dt: float,
    tolerance: float = 1e-10,
) -> RankOneControlSpan:
    """Resolve a positive rank-one span containing every expert price shift."""
    if schedules.ndim != 3 or schedules.shape[0] < 1 or schedules.shape[1] < 1:
        raise ValueError("schedules must have shape (experts, steps, drivers)")
    if schedules.shape[2] != 2:
        raise ValueError("rank-one rBergomi smoothing requires exactly two drivers")
    if not schedules.is_floating_point() or not torch.isfinite(schedules).all():
        raise ValueError("schedules must be finite floating tensors")
    if not math.isfinite(step_dt) or step_dt <= 0.0:
        raise ValueError("step_dt must be finite and positive")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")

    price_shifts = math.sqrt(step_dt) * schedules[:, :, 1]
    norms = torch.linalg.vector_norm(price_shifts, dim=1)
    anchor_index = int(torch.argmax(norms))
    anchor_norm = float(norms[anchor_index])
    if anchor_norm <= tolerance:
        raise ValueError("at least one expert must have a nonzero price-driver shift")
    anchor = price_shifts[anchor_index]
    sign_tolerance = tolerance * max(1.0, float(torch.max(torch.abs(anchor))))
    if bool((anchor > sign_tolerance).all()):
        direction = anchor / norms[anchor_index]
    elif bool((anchor < -sign_tolerance).all()):
        direction = -anchor / norms[anchor_index]
    else:
        raise ValueError("the rank-one price shift must be strictly one-signed")
    validate_positive_unit_direction(direction, steps=schedules.shape[1])

    coefficients = price_shifts @ direction
    residual = price_shifts - coefficients[:, None] * direction[None, :]
    maximum_residual = float(torch.max(torch.abs(residual)))
    reference_scale = max(1.0, float(torch.max(torch.abs(price_shifts))))
    if maximum_residual > tolerance * reference_scale:
        raise ValueError("expert price shifts are not contained in one common span")
    return RankOneControlSpan(
        direction=direction,
        expert_coefficients=coefficients,
        maximum_span_residual=maximum_residual,
    )


def positive_rank_two_subspace(
    control_direction: torch.Tensor,
    *,
    decay_candidates: tuple[float, ...] = (-4.0, 0.0, 4.0),
    minimum_gram_eigenvalue: float = 1e-3,
) -> PositiveRankTwoSubspace:
    """Add the best-conditioned positive exponential event direction."""
    validate_positive_unit_direction(control_direction)
    if not decay_candidates or len(set(decay_candidates)) != len(decay_candidates):
        raise ValueError("decay candidates must be nonempty and unique")
    candidates: list[tuple[float, torch.Tensor, float]] = []
    for decay in decay_candidates:
        direction = positive_exponential_direction(
            control_direction.numel(),
            decay=float(decay),
            device=control_direction.device,
            dtype=control_direction.dtype,
        )
        correlation = float(torch.dot(control_direction, direction))
        candidates.append((float(decay), direction, 1.0 - correlation**2))
    decay, event_direction, _score = max(
        candidates,
        key=lambda value: (value[2], -abs(value[0]), -value[0]),
    )
    basis = torch.stack((control_direction, event_direction), dim=1)
    gram = basis.T @ basis
    eigenvalue = float(torch.linalg.eigvalsh(gram)[0])
    if eigenvalue < minimum_gram_eigenvalue:
        raise ValueError("positive rank-two basis is numerically ill-conditioned")
    return PositiveRankTwoSubspace(
        basis=basis,
        gram=gram,
        coordinate_covariance=torch.linalg.inv(gram),
        event_direction_decay=decay,
        minimum_gram_eigenvalue=eigenvalue,
    )


def _deterministic_expert_schedules(
    all_expert_controls: torch.Tensor,
    *,
    tolerance: float,
) -> torch.Tensor:
    if all_expert_controls.ndim != 4 or all_expert_controls.shape[0] < 1:
        raise ValueError("all expert controls must have shape (paths, experts, steps, 2)")
    schedules = all_expert_controls[0]
    maximum_error = float(torch.max(torch.abs(all_expert_controls - schedules.unsqueeze(0))))
    scale = max(1.0, float(torch.max(torch.abs(schedules))))
    if maximum_error > tolerance * scale:
        raise ValueError("control-span marginalization rejects path-dependent experts")
    return schedules


def _outer_component_log_q_over_p(
    schedules: torch.Tensor,
    target_increments: torch.Tensor,
    *,
    step_dt: float,
) -> torch.Tensor:
    first_control = schedules[:, :, 0]
    first_target = target_increments[:, :, 0]
    stochastic = torch.einsum("es,bs->be", first_control, first_target)
    energy = step_dt * torch.sum(first_control.square(), dim=1)
    return stochastic - 0.5 * energy.unsqueeze(0)


def evaluate_control_span_marginalized_mixture(
    sample: RBergomiMixtureSample,
    *,
    task: DownsideExcursionTask,
    rho: float,
    tolerance: float = 1e-10,
) -> ControlSpanMarginalizedEstimate:
    """Evaluate exact rank-one DCS-MGI on an augmented mixture sample."""
    paths = sample.paths
    target = paths.target_brownian_increments
    if target is None:
        raise ValueError("target Brownian increments must be recorded")
    if target.ndim != 3 or target.shape[2] != 2 or target.shape[1] < 1:
        raise ValueError("target increments must have shape (paths, steps, 2)")
    schedules = _deterministic_expert_schedules(
        sample.all_expert_controls,
        tolerance=tolerance,
    )
    span = rank_one_price_control_span(
        schedules,
        step_dt=paths.step_dt,
        tolerance=tolerance,
    )
    affine = affine_rbergomi_log_spot(
        spot=paths.spot,
        variance=paths.variance,
        proposal_fine_brownian_increments=target,
        fine_step_dt=paths.step_dt,
        rho=rho,
        direction=span.direction,
    )
    thresholds = downside_excursion_thresholds(
        affine.intercept,
        affine.slope,
        step_dt=paths.step_dt,
        task=task,
    )
    hard_event = task.hard_event(paths.spot, paths.step_dt)
    threshold_event = affine.coordinate <= thresholds.combined
    if not torch.equal(hard_event, threshold_event):
        mismatches = int(torch.count_nonzero(hard_event != threshold_event))
        raise AssertionError(f"control-span threshold failed on {mismatches} paths")

    outer_component = _outer_component_log_q_over_p(
        schedules,
        target,
        step_dt=paths.step_dt,
    )
    outer_mixture = log_mixture_q_over_p(outer_component, sample.weights)
    log_outer_likelihood = -outer_mixture
    outer_likelihood = torch.exp(log_outer_likelihood)
    marginalized = scaled_normal_cdf(log_outer_likelihood, thresholds.combined)
    raw = hard_event.to(paths.spot.dtype) * torch.exp(sample.mixture_log_likelihood)
    if not torch.isfinite(raw).all() or not torch.isfinite(marginalized).all():
        raise FloatingPointError("control-span contribution became nonfinite")

    reconstructed_component = (
        outer_component
        + affine.coordinate[:, None] * span.expert_coefficients[None, :]
        - 0.5 * span.expert_coefficients.square()[None, :]
    )
    reconstructed_mixture = log_mixture_q_over_p(reconstructed_component, sample.weights)
    component_error = float(
        torch.max(torch.abs(reconstructed_component - sample.component_log_q_over_p))
    )
    mixture_error = float(
        torch.max(torch.abs(reconstructed_mixture - sample.log_mixture_q_over_p))
    )

    zero_expert = torch.amax(torch.abs(schedules), dim=(1, 2)) <= tolerance
    defensive_weight: float | None = None
    bound_violation = 0.0
    if bool(zero_expert.any()):
        defensive_weight = float(sample.weights[zero_expert].sum())
        bound_violation = max(
            0.0,
            float(torch.max(outer_likelihood)) - 1.0 / defensive_weight,
        )
    residual_projection = torch.sum(affine.residual * span.direction, dim=1)
    return ControlSpanMarginalizedEstimate(
        threshold=thresholds.combined,
        coordinate=affine.coordinate,
        hard_event=hard_event,
        threshold_event=threshold_event,
        raw_mixture_contribution=raw,
        marginalized_contribution=marginalized,
        log_outer_likelihood=log_outer_likelihood,
        outer_likelihood=outer_likelihood,
        span=span,
        defensive_weight=defensive_weight,
        maximum_defensive_bound_violation=bound_violation,
        maximum_component_log_density_error=component_error,
        maximum_mixture_log_density_error=mixture_error,
        maximum_path_reconstruction_error=affine.maximum_path_reconstruction_error,
        maximum_residual_projection=float(torch.max(torch.abs(residual_projection))),
    )


def evaluate_control_span_marginalized_adjacent_mixture(
    sample: CoupledRBergomiMixtureSample,
    *,
    task: DownsideExcursionTask,
    rho: float,
    tolerance: float = 1e-10,
) -> ControlSpanMarginalizedAdjacentEstimate:
    """Evaluate the exact fine-minus-coarse rank-one DCS-MGI correction."""
    paths = sample.paths
    target = paths.target_fine_brownian_increments
    if target is None:
        raise ValueError("target fine Brownian increments must be recorded")
    schedules = _deterministic_expert_schedules(
        sample.all_expert_controls,
        tolerance=tolerance,
    )
    span = rank_one_price_control_span(
        schedules,
        step_dt=paths.fine.step_dt,
        tolerance=tolerance,
    )
    fine_affine = affine_rbergomi_log_spot(
        spot=paths.fine.spot,
        variance=paths.fine.variance,
        proposal_fine_brownian_increments=target,
        fine_step_dt=paths.fine.step_dt,
        rho=rho,
        direction=span.direction,
    )
    coarse_affine = affine_rbergomi_log_spot(
        spot=paths.coarse.spot,
        variance=paths.coarse.variance,
        proposal_fine_brownian_increments=target,
        fine_step_dt=paths.fine.step_dt,
        rho=rho,
        direction=span.direction,
        coarse_from_fine_pairs=True,
    )
    coordinate_mismatch = float(torch.max(torch.abs(fine_affine.coordinate - coarse_affine.coordinate)))
    if coordinate_mismatch > tolerance:
        raise AssertionError("fine and coarse control-span coordinates differ")
    fine_threshold = downside_excursion_thresholds(
        fine_affine.intercept,
        fine_affine.slope,
        step_dt=paths.fine.step_dt,
        task=task,
    ).combined
    coarse_threshold = downside_excursion_thresholds(
        coarse_affine.intercept,
        coarse_affine.slope,
        step_dt=paths.coarse.step_dt,
        task=task,
    ).combined
    fine_event = task.hard_event(paths.fine.spot, paths.fine.step_dt)
    coarse_event = task.hard_event(paths.coarse.spot, paths.coarse.step_dt)
    if not torch.equal(fine_event, fine_affine.coordinate <= fine_threshold):
        raise AssertionError("fine control-span threshold replay failed")
    if not torch.equal(coarse_event, coarse_affine.coordinate <= coarse_threshold):
        raise AssertionError("coarse control-span threshold replay failed")

    outer_component = _outer_component_log_q_over_p(
        schedules,
        target,
        step_dt=paths.fine.step_dt,
    )
    outer_mixture = log_mixture_q_over_p(outer_component, sample.weights)
    log_outer_likelihood = -outer_mixture
    outer_likelihood = torch.exp(log_outer_likelihood)
    marginalized_correction = scaled_normal_cdf_difference(
        log_outer_likelihood,
        fine_threshold,
        coarse_threshold,
    )
    raw_correction = (
        fine_event.to(paths.fine.spot.dtype) - coarse_event.to(paths.fine.spot.dtype)
    ) * torch.exp(sample.mixture_log_likelihood)
    if not torch.isfinite(raw_correction).all() or not torch.isfinite(
        marginalized_correction
    ).all():
        raise FloatingPointError("control-span correction became nonfinite")

    reconstructed_component = (
        outer_component
        + fine_affine.coordinate[:, None] * span.expert_coefficients[None, :]
        - 0.5 * span.expert_coefficients.square()[None, :]
    )
    reconstructed_mixture = log_mixture_q_over_p(reconstructed_component, sample.weights)
    component_error = float(
        torch.max(torch.abs(reconstructed_component - sample.component_log_q_over_p))
    )
    mixture_error = float(
        torch.max(torch.abs(reconstructed_mixture - sample.log_mixture_q_over_p))
    )
    zero_expert = torch.amax(torch.abs(schedules), dim=(1, 2)) <= tolerance
    defensive_weight: float | None = None
    bound_violation = 0.0
    if bool(zero_expert.any()):
        defensive_weight = float(sample.weights[zero_expert].sum())
        bound_violation = max(
            0.0,
            float(torch.max(outer_likelihood)) - 1.0 / defensive_weight,
        )
    fine_projection = torch.sum(fine_affine.residual * span.direction, dim=1)
    coarse_projection = torch.sum(coarse_affine.residual * span.direction, dim=1)
    return ControlSpanMarginalizedAdjacentEstimate(
        fine_threshold=fine_threshold,
        coarse_threshold=coarse_threshold,
        raw_correction=raw_correction,
        marginalized_correction=marginalized_correction,
        log_outer_likelihood=log_outer_likelihood,
        outer_likelihood=outer_likelihood,
        span=span,
        defensive_weight=defensive_weight,
        maximum_defensive_bound_violation=bound_violation,
        maximum_component_log_density_error=component_error,
        maximum_mixture_log_density_error=mixture_error,
        maximum_coordinate_mismatch=coordinate_mismatch,
        maximum_fine_path_reconstruction_error=(
            fine_affine.maximum_path_reconstruction_error
        ),
        maximum_coarse_path_reconstruction_error=(
            coarse_affine.maximum_path_reconstruction_error
        ),
        maximum_residual_projection=max(
            float(torch.max(torch.abs(fine_projection))),
            float(torch.max(torch.abs(coarse_projection))),
        ),
    )


def evaluate_rank_two_control_span_marginalized_mixture(
    sample: RBergomiMixtureSample,
    *,
    task: DownsideExcursionTask,
    rho: float,
    inner_samples: int,
    inner_generator: torch.Generator,
    inner_rule: Literal["iid", "stratified"] = "stratified",
    decay_candidates: tuple[float, ...] = (-4.0, 0.0, 4.0),
    tolerance: float = 1e-10,
) -> RankTwoControlSpanMarginalizedEstimate:
    """Integrate two positive price directions with an unbiased nested estimator.

    The price-control direction is fully marginalized.  A second positive event
    direction is sampled in its oblique Gaussian coordinate, and the remaining
    positive coordinate is integrated analytically.  Finite ``inner_samples``
    changes variance but not the expectation.
    """
    if inner_samples <= 0:
        raise ValueError("inner_samples must be positive")
    if not math.isfinite(rho) or not -1.0 < float(rho) < 1.0:
        raise ValueError("rho must be finite and strictly between -1 and 1")
    if not isinstance(inner_generator, torch.Generator):
        raise TypeError("inner_generator must be a torch.Generator")
    if inner_rule not in ("iid", "stratified"):
        raise ValueError("inner_rule must be 'iid' or 'stratified'")
    paths = sample.paths
    target = paths.target_brownian_increments
    if target is None:
        raise ValueError("target Brownian increments must be recorded")
    schedules = _deterministic_expert_schedules(
        sample.all_expert_controls,
        tolerance=tolerance,
    )
    control_span = rank_one_price_control_span(
        schedules,
        step_dt=paths.step_dt,
        tolerance=tolerance,
    )
    subspace = positive_rank_two_subspace(
        control_span.direction,
        decay_candidates=decay_candidates,
    )

    standardized_price = target[:, :, 1] / math.sqrt(paths.step_dt)
    coordinates = standardized_price @ subspace.basis @ subspace.coordinate_covariance
    residual = standardized_price - coordinates @ subspace.basis.T
    price_shift = math.sqrt(paths.step_dt) * schedules[:, :, 1]
    projected_shift = (
        price_shift @ subspace.basis @ subspace.coordinate_covariance @ subspace.basis.T
    )
    maximum_span_residual = float(torch.max(torch.abs(price_shift - projected_shift)))
    if maximum_span_residual > tolerance * max(
        1.0,
        float(torch.max(torch.abs(price_shift))),
    ):
        raise ValueError("expert price shifts escape the rank-two integration subspace")

    rho_perpendicular = math.sqrt(1.0 - float(rho) ** 2)
    increment_slopes = (
        rho_perpendicular
        * torch.sqrt(paths.variance[:, :-1]).unsqueeze(2)
        * math.sqrt(paths.step_dt)
        * subspace.basis.unsqueeze(0)
    )
    slopes = torch.cat(
        (
            torch.zeros(
                paths.spot.shape[0],
                1,
                2,
                device=paths.spot.device,
                dtype=paths.spot.dtype,
            ),
            torch.cumsum(increment_slopes, dim=1),
        ),
        dim=1,
    )
    log_spot = torch.log(paths.spot)
    intercept = log_spot - torch.sum(slopes * coordinates[:, None, :], dim=2)
    reconstructed_path = intercept + torch.sum(slopes * coordinates[:, None, :], dim=2)
    path_error = float(torch.max(torch.abs(reconstructed_path - log_spot)))
    projection = residual @ subspace.basis
    projection_error = float(torch.max(torch.abs(projection)))

    covariance = subspace.coordinate_covariance
    coordinate_one_sd = torch.sqrt(covariance[0, 0])
    conditional_coefficient = covariance[1, 0] / covariance[0, 0]
    conditional_variance = covariance[1, 1] - covariance[1, 0].square() / covariance[0, 0]
    if float(conditional_variance) <= 0.0:
        raise FloatingPointError("rank-two conditional variance is not positive")
    conditional_sd = torch.sqrt(conditional_variance)
    if inner_rule == "iid":
        inner_standard = torch.randn(
            paths.spot.shape[0],
            inner_samples,
            device=paths.spot.device,
            dtype=paths.spot.dtype,
            generator=inner_generator,
        )
    else:
        uniforms = torch.rand(
            paths.spot.shape[0],
            inner_samples,
            device=paths.spot.device,
            dtype=paths.spot.dtype,
            generator=inner_generator,
        )
        strata = torch.arange(
            inner_samples,
            device=paths.spot.device,
            dtype=paths.spot.dtype,
        )
        uniforms = (uniforms + strata.unsqueeze(0)) / inner_samples
        epsilon = torch.finfo(paths.spot.dtype).eps
        inner_standard = torch.special.ndtri(torch.clamp(uniforms, epsilon, 1.0 - epsilon))
    coordinate_one = coordinate_one_sd * inner_standard
    conditional_mean_two = conditional_coefficient * coordinate_one
    inner_intercept = intercept[:, None, :] + slopes[:, None, :, 0] * coordinate_one[:, :, None]
    slope_two = slopes[:, None, :, 1].expand(-1, inner_samples, -1)
    thresholds = downside_excursion_thresholds(
        inner_intercept.reshape(-1, inner_intercept.shape[2]),
        slope_two.reshape(-1, slope_two.shape[2]),
        step_dt=paths.step_dt,
        task=task,
    ).combined.reshape(paths.spot.shape[0], inner_samples)
    conditional_cdf = torch.special.ndtr(
        (thresholds - conditional_mean_two) / conditional_sd
    )
    conditional_probability = torch.mean(conditional_cdf, dim=1)

    outer_component = _outer_component_log_q_over_p(
        schedules,
        target,
        step_dt=paths.step_dt,
    )
    outer_mixture = log_mixture_q_over_p(outer_component, sample.weights)
    log_outer_likelihood = -outer_mixture
    outer_likelihood = torch.exp(log_outer_likelihood)
    marginalized = outer_likelihood * conditional_probability
    hard_event = task.hard_event(paths.spot, paths.step_dt)
    raw = hard_event.to(paths.spot.dtype) * torch.exp(sample.mixture_log_likelihood)
    if not torch.isfinite(marginalized).all() or not torch.isfinite(raw).all():
        raise FloatingPointError("rank-two marginalized contribution became nonfinite")

    price_component = torch.einsum("es,bs->be", price_shift, standardized_price)
    price_component = price_component - 0.5 * torch.sum(price_shift.square(), dim=1)[None, :]
    reconstructed_component = outer_component + price_component
    reconstructed_mixture = log_mixture_q_over_p(reconstructed_component, sample.weights)
    component_error = float(
        torch.max(torch.abs(reconstructed_component - sample.component_log_q_over_p))
    )
    mixture_error = float(
        torch.max(torch.abs(reconstructed_mixture - sample.log_mixture_q_over_p))
    )
    zero_expert = torch.amax(torch.abs(schedules), dim=(1, 2)) <= tolerance
    defensive_weight: float | None = None
    bound_violation = 0.0
    if bool(zero_expert.any()):
        defensive_weight = float(sample.weights[zero_expert].sum())
        bound_violation = max(
            0.0,
            float(torch.max(outer_likelihood)) - 1.0 / defensive_weight,
        )
    return RankTwoControlSpanMarginalizedEstimate(
        raw_mixture_contribution=raw,
        marginalized_contribution=marginalized,
        conditional_event_probability=conditional_probability,
        log_outer_likelihood=log_outer_likelihood,
        outer_likelihood=outer_likelihood,
        control_span=control_span,
        subspace=subspace,
        inner_samples=inner_samples,
        inner_rule=inner_rule,
        defensive_weight=defensive_weight,
        maximum_defensive_bound_violation=bound_violation,
        maximum_component_log_density_error=component_error,
        maximum_mixture_log_density_error=mixture_error,
        maximum_path_reconstruction_error=path_error,
        maximum_residual_projection=max(projection_error, maximum_span_residual),
    )
