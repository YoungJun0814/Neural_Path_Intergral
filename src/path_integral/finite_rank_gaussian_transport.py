"""Exact defensive finite-rank Gaussian transports in whitened path space.

All densities are evaluated against the same standard Gaussian reference.  No
normalizing constant or self-normalized importance weight appears anywhere in this
module.  A positive natural component makes the proposal globally equivalent to the
reference and gives the deterministic bound ``dP/dQ <= 1 / defensive_mass``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from src.path_integral.cameron_martin_modes import RBergomiModeSearchResult
from src.path_integral.volterra_action import (
    RBergomiConditionalAction,
    evaluate_action_derivatives,
)


@dataclass(frozen=True)
class FiniteRankGaussianComponent:
    """Gaussian ``N(mean, I + U diag(lambda - 1) U^T)`` component."""

    mean: torch.Tensor
    directions: torch.Tensor
    variance_eigenvalues: torch.Tensor

    def __post_init__(self) -> None:
        if self.mean.ndim != 1 or self.mean.device.type != "cpu" or self.mean.dtype != torch.float64:
            raise ValueError("component mean must be a one-dimensional CPU float64 tensor")
        if self.directions.shape != (self.mean.numel(), self.variance_eigenvalues.numel()):
            raise ValueError("finite-rank directions have the wrong shape")
        if self.directions.device.type != "cpu" or self.directions.dtype != torch.float64:
            raise ValueError("finite-rank directions must be CPU float64")
        if self.variance_eigenvalues.ndim != 1:
            raise ValueError("variance eigenvalues must be one-dimensional")
        if (
            self.variance_eigenvalues.device.type != "cpu"
            or self.variance_eigenvalues.dtype != torch.float64
        ):
            raise ValueError("variance eigenvalues must be CPU float64")
        tensors = (self.mean, self.directions, self.variance_eigenvalues)
        if any(not torch.isfinite(item).all() for item in tensors):
            raise ValueError("finite-rank component parameters must be finite")
        if torch.any(self.variance_eigenvalues <= 0.0):
            raise ValueError("variance eigenvalues must be strictly positive")
        if self.rank:
            gram = self.directions.T @ self.directions
            identity = torch.eye(self.rank, dtype=torch.float64)
            if float(torch.amax(torch.abs(gram - identity))) > 2e-10:
                raise ValueError("finite-rank directions must be orthonormal")

    @property
    def dimension(self) -> int:
        return int(self.mean.numel())

    @property
    def rank(self) -> int:
        return int(self.variance_eigenvalues.numel())

    @classmethod
    def natural(cls, dimension: int) -> FiniteRankGaussianComponent:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise ValueError("component dimension must be a positive integer")
        return cls(
            mean=torch.zeros(dimension, dtype=torch.float64),
            directions=torch.empty((dimension, 0), dtype=torch.float64),
            variance_eigenvalues=torch.empty(0, dtype=torch.float64),
        )

    def is_natural(self, *, tolerance: float = 1e-14) -> bool:
        return self.rank == 0 and float(torch.linalg.vector_norm(self.mean)) <= tolerance

    def log_q_over_p(self, samples: torch.Tensor) -> torch.Tensor:
        """Return the exact component log density ratio ``log(q_j / p)``."""

        _validate_samples(samples, self.dimension)
        centered = samples - self.mean
        quadratic = torch.sum(centered * centered, dim=1)
        log_determinant = torch.zeros((), dtype=torch.float64)
        if self.rank:
            coordinates = centered @ self.directions
            inverse_correction = 1.0 / self.variance_eigenvalues - 1.0
            quadratic = quadratic + torch.sum(
                coordinates.square() * inverse_correction,
                dim=1,
            )
            log_determinant = torch.sum(torch.log(self.variance_eigenvalues))
        reference_quadratic = torch.sum(samples * samples, dim=1)
        return -0.5 * log_determinant - 0.5 * quadratic + 0.5 * reference_quadratic

    def transform_standard_normal(self, standard_normal: torch.Tensor) -> torch.Tensor:
        _validate_samples(standard_normal, self.dimension)
        transformed = standard_normal
        if self.rank:
            coordinates = standard_normal @ self.directions
            correction = coordinates * (torch.sqrt(self.variance_eigenvalues) - 1.0)
            transformed = standard_normal + correction @ self.directions.T
        return self.mean + transformed


def build_isotropic_small_noise_safety_component(
    dimension: int,
    *,
    epsilon: float,
) -> FiniteRankGaussianComponent:
    """Return the fixed-grid safety law ``N(0, I / epsilon)``.

    The full-rank representation is intentional: inflating only a selected
    subspace does not cover a dominating point outside that subspace and hence
    does not provide the mode-omission efficiency guarantee.
    """

    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise ValueError("component dimension must be a positive integer")
    if not math.isfinite(epsilon) or not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must lie in (0, 1]")
    return FiniteRankGaussianComponent(
        mean=torch.zeros(dimension, dtype=torch.float64),
        directions=torch.eye(dimension, dtype=torch.float64),
        variance_eigenvalues=torch.full(
            (dimension,),
            1.0 / epsilon,
            dtype=torch.float64,
        ),
    )


@dataclass(frozen=True)
class DefensiveFiniteRankGaussianMixture:
    components: tuple[FiniteRankGaussianComponent, ...]
    weights: torch.Tensor

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("proposal mixture must contain at least one component")
        dimension = self.components[0].dimension
        if any(component.dimension != dimension for component in self.components):
            raise ValueError("proposal components must have a common dimension")
        if self.weights.shape != (len(self.components),):
            raise ValueError("proposal weights have the wrong shape")
        if self.weights.device.type != "cpu" or self.weights.dtype != torch.float64:
            raise ValueError("proposal weights must be CPU float64")
        if not torch.isfinite(self.weights).all() or torch.any(self.weights <= 0.0):
            raise ValueError("proposal weights must be finite and strictly positive")
        if not math.isclose(float(torch.sum(self.weights)), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("proposal weights must sum to one")
        if self.defensive_mass <= 0.0:
            raise ValueError("proposal requires a positive natural defensive component")

    @property
    def dimension(self) -> int:
        return self.components[0].dimension

    @property
    def defensive_mass(self) -> float:
        return float(
            sum(
                self.weights[index]
                for index, component in enumerate(self.components)
                if component.is_natural()
            )
        )

    def component_log_q_over_p(self, samples: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [component.log_q_over_p(samples) for component in self.components],
            dim=1,
        )

    def log_q_over_p(self, samples: torch.Tensor) -> torch.Tensor:
        components = self.component_log_q_over_p(samples)
        return torch.logsumexp(components + torch.log(self.weights), dim=1)

    def sample(
        self,
        sample_size: int,
        *,
        path_seed: int,
        label_seed: int,
    ) -> FiniteRankMixtureSample:
        if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 1:
            raise ValueError("sample size must be a positive integer")
        if path_seed == label_seed:
            raise ValueError("path and label seeds must be distinct")
        path_generator = torch.Generator().manual_seed(path_seed)
        label_generator = torch.Generator().manual_seed(label_seed)
        labels = torch.multinomial(
            self.weights,
            sample_size,
            replacement=True,
            generator=label_generator,
        )
        standard = torch.randn(
            (sample_size, self.dimension),
            dtype=torch.float64,
            generator=path_generator,
        )
        samples = torch.empty_like(standard)
        for index, component in enumerate(self.components):
            selected = labels == index
            if torch.any(selected):
                samples[selected] = component.transform_standard_normal(standard[selected])
        component_ratios = self.component_log_q_over_p(samples)
        mixture_ratio = torch.logsumexp(component_ratios + torch.log(self.weights), dim=1)
        return FiniteRankMixtureSample(
            samples=samples,
            labels=labels,
            component_log_q_over_p=component_ratios,
            log_q_over_p=mixture_ratio,
            log_p_over_q=-mixture_ratio,
        )


@dataclass(frozen=True)
class FiniteRankMixtureSample:
    samples: torch.Tensor
    labels: torch.Tensor
    component_log_q_over_p: torch.Tensor
    log_q_over_p: torch.Tensor
    log_p_over_q: torch.Tensor


@dataclass(frozen=True)
class CurvatureTransportConfig:
    defensive_mass: float = 0.1
    asymptotic_safety_mass: float = 0.0
    minimum_variance: float = 0.05
    maximum_variance: float = 20.0
    positive_curvature_tolerance: float = 1e-7

    def __post_init__(self) -> None:
        if not 0.0 < self.defensive_mass < 1.0:
            raise ValueError("defensive mass must lie in (0, 1)")
        if (
            not math.isfinite(self.asymptotic_safety_mass)
            or self.asymptotic_safety_mass < 0.0
            or self.defensive_mass + self.asymptotic_safety_mass >= 1.0
        ):
            raise ValueError(
                "asymptotic safety mass must be nonnegative and leave positive shifted mass"
            )
        if not 0.0 < self.minimum_variance <= self.maximum_variance:
            raise ValueError("variance clipping bounds are invalid")
        if not math.isfinite(self.maximum_variance):
            raise ValueError("maximum variance must be finite")
        if not math.isfinite(self.positive_curvature_tolerance) or (
            self.positive_curvature_tolerance <= 0.0
        ):
            raise ValueError("curvature tolerance must be finite and positive")


def build_curvature_transport(
    action: RBergomiConditionalAction,
    modes: RBergomiModeSearchResult,
    *,
    config: CurvatureTransportConfig | None = None,
) -> DefensiveFiniteRankGaussianMixture:
    """Build the exact local Laplace mixture; reject saddles rather than masking them."""

    config = config or CurvatureTransportConfig()
    if not modes.modes:
        raise ValueError("at least one converged action mode is required")
    natural = FiniteRankGaussianComponent.natural(action.basis.dimension)
    shifted: list[FiniteRankGaussianComponent] = []
    log_evidence: list[torch.Tensor] = []
    for mode in modes.modes:
        derivatives = evaluate_action_derivatives(
            action,
            mode.coefficients,
            include_hessian=True,
        )
        if derivatives.hessian is None:
            raise RuntimeError("action Hessian audit was not computed")
        eigenvalues, eigenvectors = torch.linalg.eigh(derivatives.hessian)
        if float(torch.min(eigenvalues)) <= config.positive_curvature_tolerance:
            raise ValueError("curvature transport cannot be built from a saddle or flat mode")
        variances = torch.clamp(
            1.0 / eigenvalues,
            min=config.minimum_variance,
            max=config.maximum_variance,
        )
        directions = action.basis.matrix @ eigenvectors
        shifted.append(
            FiniteRankGaussianComponent(
                mean=mode.proposal_mean.detach().clone(),
                directions=directions.detach(),
                variance_eigenvalues=variances.detach(),
            )
        )
        log_evidence.append(
            torch.tensor(-mode.action_value / action.epsilon, dtype=torch.float64)
            - 0.5 * torch.sum(torch.log(eigenvalues.detach()))
        )
    components: tuple[FiniteRankGaussianComponent, ...]
    fixed_weights = [config.defensive_mass]
    if config.asymptotic_safety_mass > 0.0:
        # For fixed finite dimension and epsilon -> 0, N(0, I/epsilon) gives
        # an exact, normalized component whose second-moment exponent is twice
        # the contracted probability exponent.  A fixed positive mixture mass
        # therefore prevents missed Laplace modes from destroying logarithmic
        # efficiency.  This full-rank construction is deliberately not claimed
        # to define an equivalent change of Wiener measure in infinite dimension.
        dimension = action.basis.dimension
        broad = build_isotropic_small_noise_safety_component(
            dimension,
            epsilon=action.epsilon,
        )
        components = (natural, broad, *shifted)
        fixed_weights.append(config.asymptotic_safety_mass)
    else:
        components = (natural, *shifted)
    shifted_mass = 1.0 - sum(fixed_weights)
    relative_weights = torch.softmax(torch.stack(log_evidence), dim=0)
    weights = torch.cat(
        (
            torch.tensor(fixed_weights, dtype=torch.float64),
            shifted_mass * relative_weights,
        )
    )
    return DefensiveFiniteRankGaussianMixture(
        components=components,
        weights=weights,
    )


def _validate_samples(samples: torch.Tensor, dimension: int) -> None:
    if samples.ndim != 2 or samples.shape[1] != dimension:
        raise ValueError("samples have the wrong shape")
    if samples.device.type != "cpu" or samples.dtype != torch.float64:
        raise ValueError("samples must be CPU float64")
    if not torch.isfinite(samples).all():
        raise ValueError("samples must be finite")
