"""Exact control-span marginalization for deterministic Gaussian mixtures.

This module contains no model-specific simulation code.  It implements the
finite-dimensional density identities in ``docs/theory/G11_THEOREMS.md`` and keeps
raw and Rao--Blackwellized contributions unnormalized.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from src.path_integral.mixture import log_mixture_q_over_p, sample_mixture_labels


@dataclass(frozen=True)
class GaussianMixtureShiftSpec:
    """A randomized mixture of deterministic identity-covariance Gaussian shifts."""

    means: torch.Tensor
    weights: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.means, torch.Tensor) or not isinstance(
            self.weights, torch.Tensor
        ):
            raise TypeError("means and weights must be torch tensors")
        if self.means.ndim != 2 or self.means.shape[0] < 1 or self.means.shape[1] < 1:
            raise ValueError("means must have shape (components, dimension)")
        if self.weights.ndim != 1 or self.weights.shape[0] != self.means.shape[0]:
            raise ValueError("weights must have shape (components,)")
        if not self.means.is_floating_point() or not self.weights.is_floating_point():
            raise TypeError("means and weights must be floating point")
        if self.means.device != self.weights.device or self.means.dtype != self.weights.dtype:
            raise ValueError("means and weights must share device and dtype")
        if not torch.isfinite(self.means).all() or not torch.isfinite(self.weights).all():
            raise ValueError("means and weights must be finite")
        if bool((self.weights <= 0.0).any()):
            raise ValueError("mixture weights must be strictly positive")
        tolerance = 64.0 * torch.finfo(self.means.dtype).eps
        if not bool(torch.abs(torch.sum(self.weights) - 1.0) <= tolerance):
            raise ValueError("mixture weights must sum to one")

    @property
    def components(self) -> int:
        return int(self.means.shape[0])

    @property
    def dimension(self) -> int:
        return int(self.means.shape[1])

    def natural_component_weight(self, *, tolerance: float = 1e-12) -> float | None:
        """Return the total mass of zero-mean components, if present."""

        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("tolerance must be finite and nonnegative")
        mask = torch.amax(torch.abs(self.means), dim=1) <= tolerance
        if not bool(mask.any()):
            return None
        return float(torch.sum(self.weights[mask]))


@dataclass(frozen=True)
class OrthonormalControlSpan:
    """An orthonormal integration basis and projected proposal means."""

    basis: torch.Tensor
    projected_means: torch.Tensor
    residual_means: torch.Tensor
    maximum_orthonormality_error: float
    maximum_residual_projection: float

    @property
    def dimension(self) -> int:
        return int(self.basis.shape[0])

    @property
    def rank(self) -> int:
        return int(self.basis.shape[1])


@dataclass(frozen=True)
class MarginalLikelihoodEvaluation:
    """Full and residual mixture densities on target-coordinate proposal samples."""

    coordinate: torch.Tensor
    residual: torch.Tensor
    full_component_log_q_over_p: torch.Tensor
    residual_component_log_q_over_p: torch.Tensor
    full_log_q_over_p: torch.Tensor
    residual_log_q_over_p: torch.Tensor
    full_log_likelihood: torch.Tensor
    residual_log_likelihood: torch.Tensor
    full_likelihood: torch.Tensor
    residual_likelihood: torch.Tensor
    defensive_weight: float | None
    maximum_full_bound_violation: float
    maximum_residual_bound_violation: float
    maximum_component_reconstruction_error: float
    maximum_mixture_reconstruction_error: float
    maximum_sample_residual_projection: float


@dataclass(frozen=True)
class MarginalizedFunctionEvaluation:
    """Matched raw and exact target-conditional contributions."""

    density: MarginalLikelihoodEvaluation
    raw_value: torch.Tensor
    conditional_target_value: torch.Tensor
    raw_contribution: torch.Tensor
    marginalized_contribution: torch.Tensor


@dataclass(frozen=True)
class GaussianMixtureSample:
    """Target coordinates sampled under a randomized Gaussian-shift mixture."""

    target_coordinates: torch.Tensor
    labels: torch.Tensor


def build_orthonormal_control_span(
    spec: GaussianMixtureShiftSpec,
    basis: torch.Tensor,
    *,
    tolerance: float = 1e-10,
) -> OrthonormalControlSpan:
    """Validate an orthonormal basis and project every proposal mean."""

    if not isinstance(basis, torch.Tensor):
        raise TypeError("basis must be a torch tensor")
    if basis.ndim != 2 or basis.shape[0] != spec.dimension:
        raise ValueError("basis must have shape (dimension, rank)")
    if basis.shape[1] > spec.dimension:
        raise ValueError("span rank cannot exceed the ambient dimension")
    if basis.device != spec.means.device or basis.dtype != spec.means.dtype:
        raise ValueError("basis and mixture means must share device and dtype")
    if not basis.is_floating_point() or not torch.isfinite(basis).all():
        raise ValueError("basis must be finite and floating point")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")

    rank = int(basis.shape[1])
    if rank == 0:
        orthonormality_error = 0.0
    else:
        gram = basis.T @ basis
        identity = torch.eye(rank, device=basis.device, dtype=basis.dtype)
        orthonormality_error = float(torch.max(torch.abs(gram - identity)))
        if orthonormality_error > tolerance:
            raise ValueError("basis columns must be orthonormal")

    projected = spec.means @ basis
    residual = spec.means - projected @ basis.T
    if rank == 0:
        residual_projection = 0.0
    else:
        residual_projection = float(torch.max(torch.abs(residual @ basis)))
    scale = max(1.0, float(torch.max(torch.abs(spec.means))))
    if residual_projection > tolerance * scale:
        raise FloatingPointError("proposal-mean residual is not orthogonal to the span")
    return OrthonormalControlSpan(
        basis=basis,
        projected_means=projected,
        residual_means=residual,
        maximum_orthonormality_error=orthonormality_error,
        maximum_residual_projection=residual_projection,
    )


def control_span_from_vectors(
    spec: GaussianMixtureShiftSpec,
    vectors: torch.Tensor,
    *,
    tolerance: float = 1e-10,
) -> OrthonormalControlSpan:
    """Construct a deterministic SVD basis for the row span of supplied vectors."""

    if not isinstance(vectors, torch.Tensor):
        raise TypeError("vectors must be a torch tensor")
    if vectors.ndim != 2 or vectors.shape[1] != spec.dimension:
        raise ValueError("vectors must have shape (count, dimension)")
    if vectors.shape[0] < 1:
        raise ValueError("at least one vector is required")
    if vectors.device != spec.means.device or vectors.dtype != spec.means.dtype:
        raise ValueError("vectors and mixture means must share device and dtype")
    if not vectors.is_floating_point() or not torch.isfinite(vectors).all():
        raise ValueError("vectors must be finite and floating point")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")

    _left, singular_values, right = torch.linalg.svd(vectors, full_matrices=False)
    if singular_values.numel() == 0 or float(torch.max(singular_values)) == 0.0:
        basis = torch.empty(
            (spec.dimension, 0), device=spec.means.device, dtype=spec.means.dtype
        )
    else:
        cutoff = tolerance * max(1.0, float(torch.max(singular_values)))
        rank = int(torch.count_nonzero(singular_values > cutoff))
        basis = right[:rank].T.contiguous()
        # SVD column signs are arbitrary.  Orient the largest-magnitude entry positive
        # to keep artifacts deterministic across equivalent calls.
        for column in range(rank):
            pivot = int(torch.argmax(torch.abs(basis[:, column])))
            if float(basis[pivot, column]) < 0.0:
                basis[:, column] = -basis[:, column]
    return build_orthonormal_control_span(spec, basis, tolerance=tolerance)


def _validate_samples(
    samples: torch.Tensor,
    spec: GaussianMixtureShiftSpec,
) -> None:
    if not isinstance(samples, torch.Tensor):
        raise TypeError("samples must be a torch tensor")
    if samples.ndim != 2 or samples.shape[1] != spec.dimension or samples.shape[0] < 1:
        raise ValueError("samples must have shape (batch, dimension)")
    if samples.device != spec.means.device or samples.dtype != spec.means.dtype:
        raise ValueError("samples and mixture means must share device and dtype")
    if not samples.is_floating_point() or not torch.isfinite(samples).all():
        raise ValueError("samples must be finite and floating point")


def component_log_q_over_p(
    samples: torch.Tensor,
    means: torch.Tensor,
) -> torch.Tensor:
    """Evaluate identity-covariance Gaussian shift density ratios."""

    if not isinstance(samples, torch.Tensor) or not isinstance(means, torch.Tensor):
        raise TypeError("samples and means must be torch tensors")
    if samples.ndim != 2 or means.ndim != 2 or samples.shape[1] != means.shape[1]:
        raise ValueError("samples and means must have matching final dimensions")
    if samples.device != means.device or samples.dtype != means.dtype:
        raise ValueError("samples and means must share device and dtype")
    if not samples.is_floating_point() or not means.is_floating_point():
        raise TypeError("samples and means must be floating point")
    if not torch.isfinite(samples).all() or not torch.isfinite(means).all():
        raise ValueError("samples and means must be finite")
    stochastic = samples @ means.T
    energy = torch.sum(means.square(), dim=1)
    return stochastic - 0.5 * energy.unsqueeze(0)


def evaluate_marginal_likelihood(
    samples: torch.Tensor,
    spec: GaussianMixtureShiftSpec,
    span: OrthonormalControlSpan,
    *,
    tolerance: float = 1e-10,
) -> MarginalLikelihoodEvaluation:
    """Evaluate full and span-marginalized balance-mixture likelihoods."""

    _validate_samples(samples, spec)
    if span.dimension != spec.dimension:
        raise ValueError("span and mixture dimensions differ")
    if span.basis.device != samples.device or span.basis.dtype != samples.dtype:
        raise ValueError("span and samples must share device and dtype")
    if span.projected_means.shape != (spec.components, span.rank):
        raise ValueError("span projected means have an invalid shape")
    if span.residual_means.shape != spec.means.shape:
        raise ValueError("span residual means have an invalid shape")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")

    coordinate = samples @ span.basis
    residual = samples - coordinate @ span.basis.T
    full_component = component_log_q_over_p(samples, spec.means)
    residual_component = component_log_q_over_p(residual, span.residual_means)
    parallel_component = (
        coordinate @ span.projected_means.T
        - 0.5 * torch.sum(span.projected_means.square(), dim=1).unsqueeze(0)
    )
    reconstructed_component = residual_component + parallel_component
    full_mixture = log_mixture_q_over_p(full_component, spec.weights)
    residual_mixture = log_mixture_q_over_p(residual_component, spec.weights)
    reconstructed_mixture = log_mixture_q_over_p(reconstructed_component, spec.weights)
    full_log_likelihood = -full_mixture
    residual_log_likelihood = -residual_mixture
    full_likelihood = torch.exp(full_log_likelihood)
    residual_likelihood = torch.exp(residual_log_likelihood)
    if not torch.isfinite(full_likelihood).all() or not torch.isfinite(
        residual_likelihood
    ).all():
        raise FloatingPointError("Gaussian mixture likelihood became nonfinite")

    component_error = float(torch.max(torch.abs(reconstructed_component - full_component)))
    mixture_error = float(torch.max(torch.abs(reconstructed_mixture - full_mixture)))
    if span.rank == 0:
        sample_residual_projection = 0.0
    else:
        sample_residual_projection = float(torch.max(torch.abs(residual @ span.basis)))
    sample_scale = max(1.0, float(torch.max(torch.abs(samples))))
    if sample_residual_projection > tolerance * sample_scale:
        raise FloatingPointError("sample residual is not orthogonal to the span")

    defensive_weight = spec.natural_component_weight(tolerance=tolerance)
    full_violation = 0.0
    residual_violation = 0.0
    if defensive_weight is not None:
        bound = 1.0 / defensive_weight
        full_violation = max(0.0, float(torch.max(full_likelihood)) - bound)
        residual_violation = max(0.0, float(torch.max(residual_likelihood)) - bound)

    return MarginalLikelihoodEvaluation(
        coordinate=coordinate,
        residual=residual,
        full_component_log_q_over_p=full_component,
        residual_component_log_q_over_p=residual_component,
        full_log_q_over_p=full_mixture,
        residual_log_q_over_p=residual_mixture,
        full_log_likelihood=full_log_likelihood,
        residual_log_likelihood=residual_log_likelihood,
        full_likelihood=full_likelihood,
        residual_likelihood=residual_likelihood,
        defensive_weight=defensive_weight,
        maximum_full_bound_violation=full_violation,
        maximum_residual_bound_violation=residual_violation,
        maximum_component_reconstruction_error=component_error,
        maximum_mixture_reconstruction_error=mixture_error,
        maximum_sample_residual_projection=sample_residual_projection,
    )


def evaluate_marginalized_function(
    density: MarginalLikelihoodEvaluation,
    *,
    raw_value: torch.Tensor,
    conditional_target_value: torch.Tensor,
) -> MarginalizedFunctionEvaluation:
    """Build matched unnormalized raw and marginalized contributions."""

    batch = int(density.full_likelihood.shape[0])
    for name, value in (
        ("raw value", raw_value),
        ("conditional target value", conditional_target_value),
    ):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch tensor")
        if value.shape != (batch,):
            raise ValueError(f"{name} must have shape (batch,)")
        if value.device != density.full_likelihood.device:
            raise ValueError(f"{name} and likelihood must share a device")
        if not value.is_floating_point():
            raise TypeError(f"{name} must be floating point")
        if value.dtype != density.full_likelihood.dtype:
            raise ValueError(f"{name} and likelihood must share a dtype")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
    raw = density.full_likelihood * raw_value
    marginalized = density.residual_likelihood * conditional_target_value
    if not torch.isfinite(raw).all() or not torch.isfinite(marginalized).all():
        raise FloatingPointError("marginalized function contribution became nonfinite")
    return MarginalizedFunctionEvaluation(
        density=density,
        raw_value=raw_value,
        conditional_target_value=conditional_target_value,
        raw_contribution=raw,
        marginalized_contribution=marginalized,
    )


def linear_threshold_conditional_probability(
    residual: torch.Tensor,
    span: OrthonormalControlSpan,
    event_normal: torch.Tensor,
    threshold: float,
    *,
    tolerance: float = 1e-14,
) -> torch.Tensor:
    """Evaluate ``P(c^T X <= threshold | residual)`` under the target law."""

    if residual.ndim != 2 or residual.shape[1] != span.dimension:
        raise ValueError("residual must have shape (batch, dimension)")
    if event_normal.shape != (span.dimension,):
        raise ValueError("event normal must have shape (dimension,)")
    if residual.device != event_normal.device or residual.dtype != event_normal.dtype:
        raise ValueError("residual and event normal must share device and dtype")
    if not residual.is_floating_point() or not event_normal.is_floating_point():
        raise TypeError("residual and event normal must be floating point")
    if not torch.isfinite(residual).all() or not torch.isfinite(event_normal).all():
        raise ValueError("residual and event normal must be finite")
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")

    loading = span.basis.T @ event_normal
    conditional_standard_deviation = torch.linalg.vector_norm(loading)
    residual_value = residual @ event_normal
    if float(conditional_standard_deviation) <= tolerance:
        return (residual_value <= threshold).to(residual.dtype)
    argument = (threshold - residual_value) / conditional_standard_deviation
    return torch.special.ndtr(argument)


def sample_gaussian_mixture(
    spec: GaussianMixtureShiftSpec,
    num_samples: int,
    *,
    gaussian_generator: torch.Generator | None = None,
    label_generator: torch.Generator | None = None,
) -> GaussianMixtureSample:
    """Sample target coordinates from the declared randomized proposal mixture."""

    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    labels = sample_mixture_labels(
        spec.weights, num_samples, generator=label_generator
    )
    innovations = torch.randn(
        (num_samples, spec.dimension),
        device=spec.means.device,
        dtype=spec.means.dtype,
        generator=gaussian_generator,
    )
    return GaussianMixtureSample(
        target_coordinates=innovations + spec.means[labels],
        labels=labels,
    )
