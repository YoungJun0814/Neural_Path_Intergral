"""Stable strict-variance certificates for scalar Gaussian threshold events.

The certificate is finite-dimensional and proposal-conditional.  It does not
establish a continuous-time event, a mesh rate, or an end-to-end work advantage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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
    if (
        residual_component_log_q_over_p.ndim != 2
        or residual_component_log_q_over_p.shape
        != (threshold.shape[0], projected_component_means.shape[0])
    ):
        raise ValueError(
            "residual component log densities must have shape (batch, components)"
        )
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
        tensor.device == threshold.device and tensor.dtype == threshold.dtype
        for tensor in tensors
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

    centered_threshold = (
        threshold.unsqueeze(1) - projected_component_means.unsqueeze(0)
    )
    target_event_log_probability = torch.special.log_ndtr(threshold)
    proposal_event_log_probability = torch.logsumexp(
        component_log_posterior_weight
        + torch.special.log_ndtr(centered_threshold),
        dim=1,
    )
    proposal_complement_log_probability = torch.logsumexp(
        component_log_posterior_weight
        + torch.special.log_ndtr(-centered_threshold),
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
