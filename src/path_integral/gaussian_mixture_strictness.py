"""Stable strict-variance certificates for scalar Gaussian threshold events.

The certificate is finite-dimensional and proposal-conditional.  It does not
establish a continuous-time event, a mesh rate, or an end-to-end work advantage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import scipy.special
import scipy.stats
import torch

from src.path_integral.mixture import log_mixture_q_over_p


@dataclass(frozen=True)
class ScalarThresholdStrictnessCertificate:
    """Pointwise lower-bound data for the raw-minus-DCS conditional variance."""

    residual_log_likelihood: torch.Tensor
    component_log_posterior_weight: torch.Tensor
    target_event_log_probability: torch.Tensor
    proposal_event_log_probability: torch.Tensor
    proposal_complement_log_probability: torch.Tensor
    conditional_variance_gap_log_lower_bound: torch.Tensor
    finite_threshold: torch.Tensor
    strict_under_theorem: torch.Tensor
    finite_log_certificate: torch.Tensor
    maximum_probability_partition_error: float


@dataclass(frozen=True)
class MomentLocalizedPopulationCertificate:
    """Explicit population lower bound obtained from a threshold moment bound.

    The certificate is for a standard Gaussian target, a deterministic unit
    integration direction, and a finite Gaussian location mixture proposal.
    It turns the abstract localization constants ``(A, M, eta)`` into
    quantities determined by the proposal geometry and a supplied target-law
    threshold-moment upper bound.
    """

    residual_dimension: int
    defensive_mass: float
    threshold_radius: float
    residual_radius: float
    threshold_moment_order: float
    threshold_abs_moment_upper_bound: float
    target_residual_ball_probability: float
    localized_target_probability_lower_bound: float
    residual_density_ratio_log_upper_bound: float
    maximum_projected_component_mean: float
    maximum_residual_component_norm: float
    variance_gap_log_lower_bound: float
    strict_under_theorem: bool


def moment_localized_population_certificate(
    *,
    component_means: torch.Tensor,
    weights: torch.Tensor,
    direction: torch.Tensor,
    threshold_abs_moment_upper_bound: float,
    threshold_moment_order: float,
    threshold_radius: float,
    residual_radius: float,
) -> MomentLocalizedPopulationCertificate:
    r"""Certify an explicit raw-minus-DCS population variance-gap lower bound.

    Write every component mean as ``mu_j = b_j u + v_j`` with ``v_j``
    orthogonal to the unit direction ``u``.  On ``||R|| <= R0``, the residual
    proposal-to-target density ratio obeys

    ``D(R) <= M_R = sum_j w_j exp(||v_j|| R0 - ||v_j||**2 / 2)``.

    If ``E_P[|a(R)|**p] <= K_p``, the target probability of
    ``{||R|| <= R0, |a(R)| <= A}`` is at least

    ``eta = max(0, P(chi^2_(d-1) <= R0**2) - K_p / A**p)``.

    Combining these facts with the pointwise scalar-threshold certificate gives

    ``Var(raw)-Var(DCS) >= eta/M_R * Phi(-A)**2 * Phi(-A-B)``,

    where ``B=max_j |b_j|``.  The bound is deliberately rarity-dependent.  A
    zero-mean component with positive mass is required so the returned object
    also certifies the defensive-mixture contract used by DCS.
    """

    if not isinstance(component_means, torch.Tensor):
        raise TypeError("component_means must be a torch tensor")
    if not isinstance(weights, torch.Tensor):
        raise TypeError("weights must be a torch tensor")
    if not isinstance(direction, torch.Tensor):
        raise TypeError("direction must be a torch tensor")
    if component_means.ndim != 2 or component_means.shape[0] < 1:
        raise ValueError("component_means must have shape (components, dimension)")
    if direction.ndim != 1 or direction.shape[0] != component_means.shape[1]:
        raise ValueError("direction must have shape (dimension,)")
    if weights.ndim != 1 or weights.shape[0] != component_means.shape[0]:
        raise ValueError("weights must have shape (components,)")
    tensors = (component_means, weights, direction)
    if not all(tensor.device.type == "cpu" and tensor.dtype == torch.float64 for tensor in tensors):
        raise TypeError("population certificate inputs must be CPU float64 tensors")
    if not torch.isfinite(component_means).all() or not torch.isfinite(direction).all():
        raise ValueError("component_means and direction must be finite")
    if not torch.isfinite(weights).all():
        raise ValueError("weights must be finite")
    if bool((weights <= 0.0).any()) or not math.isclose(
        float(torch.sum(weights)), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("weights must be positive and sum to one")
    direction_norm = float(torch.linalg.vector_norm(direction))
    if not math.isclose(direction_norm, 1.0, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError("direction must have unit Euclidean norm")

    scalar_inputs = {
        "threshold_abs_moment_upper_bound": threshold_abs_moment_upper_bound,
        "threshold_moment_order": threshold_moment_order,
        "threshold_radius": threshold_radius,
        "residual_radius": residual_radius,
    }
    if any(not math.isfinite(float(value)) for value in scalar_inputs.values()):
        raise ValueError("localization inputs must be finite")
    if threshold_abs_moment_upper_bound < 0.0:
        raise ValueError("threshold moment upper bound must be nonnegative")
    if threshold_moment_order <= 0.0 or threshold_radius <= 0.0 or residual_radius < 0.0:
        raise ValueError(
            "moment order and threshold radius must be positive; residual radius nonnegative"
        )
    resolved_weights = weights.to(device=component_means.device, dtype=component_means.dtype)
    component_norms = torch.linalg.vector_norm(component_means, dim=1)
    zero_components = component_norms == 0.0
    defensive_mass = float(torch.sum(resolved_weights[zero_components]))
    if defensive_mass <= 0.0:
        raise ValueError("a positive-weight zero-mean defensive component is required")

    projected = component_means @ direction
    residual = component_means - projected.unsqueeze(1) * direction.unsqueeze(0)
    residual_norms = torch.linalg.vector_norm(residual, dim=1)
    log_terms = (
        torch.log(resolved_weights)
        + residual_norms * residual_radius
        - 0.5 * residual_norms.square()
    )
    log_residual_ratio_upper = float(torch.logsumexp(log_terms, dim=0))

    residual_dimension = component_means.shape[1] - 1
    if residual_dimension == 0:
        residual_ball_probability = 1.0
    else:
        residual_ball_probability = float(
            scipy.stats.chi2.cdf(residual_radius**2, df=residual_dimension)
        )
    markov_tail_bound = threshold_abs_moment_upper_bound / (
        threshold_radius**threshold_moment_order
    )
    localized_probability = max(0.0, residual_ball_probability - markov_tail_bound)
    maximum_projected = float(torch.max(torch.abs(projected)))
    maximum_residual = float(torch.max(residual_norms))

    if localized_probability > 0.0:
        log_gap = (
            math.log(localized_probability)
            - log_residual_ratio_upper
            + 2.0 * float(scipy.special.log_ndtr(-threshold_radius))
            + float(scipy.special.log_ndtr(-threshold_radius - maximum_projected))
        )
        strict = math.isfinite(log_gap)
    else:
        log_gap = -math.inf
        strict = False

    return MomentLocalizedPopulationCertificate(
        residual_dimension=residual_dimension,
        defensive_mass=defensive_mass,
        threshold_radius=float(threshold_radius),
        residual_radius=float(residual_radius),
        threshold_moment_order=float(threshold_moment_order),
        threshold_abs_moment_upper_bound=float(threshold_abs_moment_upper_bound),
        target_residual_ball_probability=residual_ball_probability,
        localized_target_probability_lower_bound=localized_probability,
        residual_density_ratio_log_upper_bound=log_residual_ratio_upper,
        maximum_projected_component_mean=maximum_projected,
        maximum_residual_component_norm=maximum_residual,
        variance_gap_log_lower_bound=log_gap,
        strict_under_theorem=strict,
    )


def scalar_threshold_strictness_certificate(
    threshold: torch.Tensor,
    projected_component_means: torch.Tensor,
    residual_component_log_q_over_p: torch.Tensor,
    weights: torch.Tensor,
) -> ScalarThresholdStrictnessCertificate:
    r"""Certify strict Rao--Blackwell improvement for finite scalar thresholds.

    Conditional on a residual value, let the integrated target coordinate be
    ``Z ~ N(0, 1)`` and let the proposal conditional density relative to the
    target be

    ``M(z) = sum_j alpha_j exp(b_j z - b_j**2 / 2)``.

    For ``A={Z<=a}``, define ``s=Q(A|R)``.  Cauchy--Schwarz gives

    ``Var_Q(L 1_A | R) >= Lbar**2 Phi(a)**2 (1-s)/s``.

    Positive Gaussian mixture weights and finite ``a,b_j`` imply ``0<s<1`` and
    ``0<Phi(a)<1``.  Hence the lower bound is strictly positive.  All probability
    calculations are performed in log space so rare finite thresholds remain
    representable well beyond the range of ordinary probabilities.
    """

    if not isinstance(threshold, torch.Tensor):
        raise TypeError("threshold must be a torch tensor")
    if not isinstance(projected_component_means, torch.Tensor):
        raise TypeError("projected component means must be a torch tensor")
    if not isinstance(residual_component_log_q_over_p, torch.Tensor):
        raise TypeError("residual component log densities must be a torch tensor")
    if not isinstance(weights, torch.Tensor):
        raise TypeError("weights must be a torch tensor")
    if threshold.ndim != 1 or threshold.shape[0] < 1:
        raise ValueError("threshold must have shape (batch,)")
    if projected_component_means.ndim != 1:
        raise ValueError("projected component means must have shape (components,)")
    if residual_component_log_q_over_p.ndim != 2 or residual_component_log_q_over_p.shape != (
        threshold.shape[0],
        projected_component_means.shape[0],
    ):
        raise ValueError("residual component log densities must have shape (batch, components)")
    if weights.ndim != 1 or weights.shape != projected_component_means.shape:
        raise ValueError("weights must have shape (components,)")
    tensors = (
        threshold,
        projected_component_means,
        residual_component_log_q_over_p,
    )
    if not all(tensor.is_floating_point() for tensor in tensors):
        raise TypeError("threshold, means, and log densities must be floating point")
    if not all(
        tensor.device == threshold.device and tensor.dtype == threshold.dtype for tensor in tensors
    ):
        raise ValueError("threshold, means, and log densities must share device and dtype")
    if bool(torch.isnan(threshold).any()):
        raise ValueError("threshold must not contain NaN")
    if not torch.isfinite(projected_component_means).all():
        raise ValueError("projected component means must be finite")
    if not torch.isfinite(residual_component_log_q_over_p).all():
        raise ValueError("residual component log densities must be finite")

    residual_log_q_over_p = log_mixture_q_over_p(
        residual_component_log_q_over_p,
        weights,
    )
    resolved_weights = weights.to(device=threshold.device, dtype=threshold.dtype)
    component_log_posterior_weight = (
        residual_component_log_q_over_p
        + torch.log(resolved_weights).unsqueeze(0)
        - residual_log_q_over_p.unsqueeze(1)
    )

    centered_threshold = threshold.unsqueeze(1) - projected_component_means.unsqueeze(0)
    target_event_log_probability = torch.special.log_ndtr(threshold)
    proposal_event_log_probability = torch.logsumexp(
        component_log_posterior_weight + torch.special.log_ndtr(centered_threshold),
        dim=1,
    )
    proposal_complement_log_probability = torch.logsumexp(
        component_log_posterior_weight + torch.special.log_ndtr(-centered_threshold),
        dim=1,
    )
    partition_log_sum = torch.logaddexp(
        proposal_event_log_probability,
        proposal_complement_log_probability,
    )
    maximum_partition_error = float(torch.max(torch.abs(partition_log_sum)))

    finite_threshold = torch.isfinite(threshold)
    negative_infinity = torch.full_like(threshold, -math.inf)
    raw_log_lower_bound = (
        -2.0 * residual_log_q_over_p
        + 2.0 * target_event_log_probability
        + proposal_complement_log_probability
        - proposal_event_log_probability
    )
    log_lower_bound = torch.where(
        finite_threshold,
        raw_log_lower_bound,
        negative_infinity,
    )
    finite_log_certificate = finite_threshold & torch.isfinite(log_lower_bound)

    return ScalarThresholdStrictnessCertificate(
        residual_log_likelihood=-residual_log_q_over_p,
        component_log_posterior_weight=component_log_posterior_weight,
        target_event_log_probability=target_event_log_probability,
        proposal_event_log_probability=proposal_event_log_probability,
        proposal_complement_log_probability=proposal_complement_log_probability,
        conditional_variance_gap_log_lower_bound=log_lower_bound,
        finite_threshold=finite_threshold,
        strict_under_theorem=finite_threshold,
        finite_log_certificate=finite_log_certificate,
        maximum_probability_partition_error=maximum_partition_error,
    )
