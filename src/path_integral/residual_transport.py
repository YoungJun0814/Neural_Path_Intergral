"""Exact defensive Gaussian-translation transports on a residual path subspace.

All densities in this module are Radon--Nikodym derivatives with respect to the
standard Gaussian measure on the hyperplane orthogonal to ``direction``.  The
ambient residual Gaussian is singular with respect to ambient Lebesgue measure; the
implementation never claims or needs an ambient Lebesgue density.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass

import torch

from src.path_integral.baseline_framework import BaselineCostLedger
from src.path_integral.mixture import sample_mixture_labels
from src.path_integral.provenance import process_peak_resident_memory_bytes


@dataclass(frozen=True)
class ResidualGaussianMixtureSpec:
    """Identity-covariance translation mixture on one Gaussian hyperplane."""

    direction: torch.Tensor
    means: torch.Tensor
    weights: torch.Tensor
    tolerance: float = 1e-10

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, torch.Tensor)
            for value in (self.direction, self.means, self.weights)
        ):
            raise TypeError("direction, means, and weights must be torch tensors")
        if self.direction.ndim != 1 or self.direction.numel() < 2:
            raise ValueError("direction must be a vector of dimension at least two")
        if (
            self.means.ndim != 2
            or self.means.shape[0] < 2
            or self.means.shape[1] != self.direction.numel()
        ):
            raise ValueError("means must have shape (at least two, dimension)")
        if self.weights.shape != (self.means.shape[0],):
            raise ValueError("weights must match the component count")
        if any(
            not value.is_floating_point()
            for value in (self.direction, self.means, self.weights)
        ):
            raise TypeError("residual proposal tensors must be floating point")
        if not (
            self.direction.device == self.means.device == self.weights.device
            and self.direction.dtype == self.means.dtype == self.weights.dtype
        ):
            raise ValueError("residual proposal tensors must share device and dtype")
        if not all(
            bool(torch.isfinite(value).all())
            for value in (self.direction, self.means, self.weights)
        ):
            raise ValueError("residual proposal tensors must be finite")
        if not math.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive")
        norm_error = abs(float(torch.linalg.vector_norm(self.direction)) - 1.0)
        if norm_error > self.tolerance:
            raise ValueError("direction must have unit Euclidean norm")
        if bool((self.weights <= 0.0).any()):
            raise ValueError("mixture weights must be strictly positive")
        if abs(float(torch.sum(self.weights)) - 1.0) > self.tolerance:
            raise ValueError("mixture weights must sum to one")
        scale = max(1.0, float(torch.amax(torch.abs(self.means))))
        projection = torch.abs(self.means @ self.direction)
        if float(torch.amax(projection)) > self.tolerance * scale:
            raise ValueError("every residual proposal mean must be orthogonal")
        if float(torch.amax(torch.abs(self.means[0]))) > self.tolerance:
            raise ValueError("component zero must be the natural residual law")

    @property
    def dimension(self) -> int:
        return int(self.direction.numel())

    @property
    def components(self) -> int:
        return int(self.means.shape[0])

    @property
    def defensive_weight(self) -> float:
        return float(self.weights[0])


@dataclass(frozen=True)
class ResidualMixtureSample:
    residual: torch.Tensor
    labels: torch.Tensor
    maximum_projection_error: float


@dataclass(frozen=True)
class ResidualLikelihoodEvaluation:
    component_log_q_over_p: torch.Tensor
    log_q_over_p: torch.Tensor
    log_likelihood: torch.Tensor
    likelihood: torch.Tensor
    maximum_projection_error: float
    maximum_bound_violation: float


@dataclass(frozen=True)
class ResidualContributionEvaluation:
    contribution: torch.Tensor
    conditional_value: torch.Tensor
    likelihood: ResidualLikelihoodEvaluation


@dataclass(frozen=True)
class SignedResidualContributionEvaluation:
    contribution: torch.Tensor
    sign: torch.Tensor
    absolute_conditional_value: torch.Tensor
    likelihood: ResidualLikelihoodEvaluation


def project_orthogonal(values: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Return the orthogonal projection onto ``direction``'s complement."""

    if values.ndim not in {1, 2}:
        raise ValueError("values must be a vector or matrix")
    if direction.ndim != 1 or values.shape[-1] != direction.numel():
        raise ValueError("direction and values dimensions differ")
    if values.device != direction.device or values.dtype != direction.dtype:
        raise ValueError("direction and values must share device and dtype")
    if not values.is_floating_point() or not torch.isfinite(values).all():
        raise ValueError("values must be finite floating point")
    if not direction.is_floating_point() or not torch.isfinite(direction).all():
        raise ValueError("direction must be finite floating point")
    norm_error = abs(float(torch.linalg.vector_norm(direction)) - 1.0)
    if norm_error > 1e-10:
        raise ValueError("direction must be unit norm")
    if values.ndim == 1:
        return values - torch.dot(values, direction) * direction
    return values - (values @ direction).unsqueeze(1) * direction.unsqueeze(0)


def sample_residual_mixture(
    spec: ResidualGaussianMixtureSpec,
    num_samples: int,
    *,
    gaussian_generator: torch.Generator | None = None,
    label_generator: torch.Generator | None = None,
) -> ResidualMixtureSample:
    """Sample target-coordinate residuals from the declared mixture."""

    if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples < 1:
        raise ValueError("num_samples must be a positive integer")
    labels = sample_mixture_labels(spec.weights, num_samples, generator=label_generator)
    innovation = torch.randn(
        (num_samples, spec.dimension),
        dtype=spec.direction.dtype,
        device=spec.direction.device,
        generator=gaussian_generator,
    )
    residual = project_orthogonal(innovation, spec.direction) + spec.means[labels]
    projection_error = float(torch.amax(torch.abs(residual @ spec.direction)))
    scale = max(1.0, float(torch.amax(torch.abs(residual))))
    if projection_error > spec.tolerance * scale:
        raise FloatingPointError("sampled residual left the declared hyperplane")
    return ResidualMixtureSample(
        residual=residual,
        labels=labels,
        maximum_projection_error=projection_error,
    )


def evaluate_residual_likelihood(
    residual: torch.Tensor,
    spec: ResidualGaussianMixtureSpec,
) -> ResidualLikelihoodEvaluation:
    """Evaluate the exact balance-mixture residual likelihood."""

    if residual.ndim != 2 or residual.shape[1] != spec.dimension or residual.shape[0] < 1:
        raise ValueError("residual must have shape (batch, dimension)")
    if residual.device != spec.direction.device or residual.dtype != spec.direction.dtype:
        raise ValueError("residual and proposal must share device and dtype")
    if not residual.is_floating_point() or not torch.isfinite(residual).all():
        raise ValueError("residual must be finite floating point")
    projection_error = float(torch.amax(torch.abs(residual @ spec.direction)))
    scale = max(1.0, float(torch.amax(torch.abs(residual))))
    if projection_error > spec.tolerance * scale:
        raise ValueError("likelihood input is not in the residual hyperplane")
    component = residual @ spec.means.T - 0.5 * torch.sum(
        spec.means.square(), dim=1
    ).unsqueeze(0)
    log_q_over_p = torch.logsumexp(
        component + torch.log(spec.weights).unsqueeze(0), dim=1
    )
    log_likelihood = -log_q_over_p
    likelihood = torch.exp(log_likelihood)
    if not torch.isfinite(likelihood).all():
        raise FloatingPointError("residual likelihood became nonfinite")
    bound = 1.0 / spec.defensive_weight
    violation = max(0.0, float(torch.amax(likelihood)) - bound)
    return ResidualLikelihoodEvaluation(
        component_log_q_over_p=component,
        log_q_over_p=log_q_over_p,
        log_likelihood=log_likelihood,
        likelihood=likelihood,
        maximum_projection_error=projection_error,
        maximum_bound_violation=violation,
    )


def evaluate_residual_contribution(
    residual: torch.Tensor,
    spec: ResidualGaussianMixtureSpec,
    *,
    log_conditional_value: torch.Tensor,
) -> ResidualContributionEvaluation:
    """Return ``g(R) dP_R/dQ_R`` stably for a nonnegative conditional value."""

    if log_conditional_value.shape != (residual.shape[0],):
        raise ValueError("log conditional values must have shape (batch,)")
    if log_conditional_value.device != residual.device or log_conditional_value.dtype != residual.dtype:
        raise ValueError("conditional values and residual must share device and dtype")
    if bool(torch.isnan(log_conditional_value).any()) or bool((log_conditional_value > 1e-14).any()):
        raise ValueError("log conditional values must lie in [-infinity, 0]")
    likelihood = evaluate_residual_likelihood(residual, spec)
    contribution = torch.exp(log_conditional_value + likelihood.log_likelihood)
    conditional = torch.exp(log_conditional_value)
    if not torch.isfinite(contribution).all():
        raise FloatingPointError("residual contribution became nonfinite")
    bound = 1.0 / spec.defensive_weight
    if float(torch.amax(contribution)) > bound * (1.0 + 1e-10):
        raise FloatingPointError("defensive contribution bound was violated")
    return ResidualContributionEvaluation(
        contribution=contribution,
        conditional_value=conditional,
        likelihood=likelihood,
    )


def evaluate_signed_residual_contribution(
    residual: torch.Tensor,
    spec: ResidualGaussianMixtureSpec,
    *,
    sign: torch.Tensor,
    log_absolute_conditional_value: torch.Tensor,
) -> SignedResidualContributionEvaluation:
    """Return a signed correction; proposal fitting uses the absolute value only."""

    if sign.shape != (residual.shape[0],) or log_absolute_conditional_value.shape != sign.shape:
        raise ValueError("signed conditional arrays must have shape (batch,)")
    if sign.device != residual.device or sign.dtype != residual.dtype:
        raise ValueError("sign and residual must share device and dtype")
    if log_absolute_conditional_value.device != residual.device or log_absolute_conditional_value.dtype != residual.dtype:
        raise ValueError("log absolute values and residual must share device and dtype")
    if not torch.isfinite(sign).all() or bool(~torch.isin(sign, torch.tensor([-1.0, 0.0, 1.0], dtype=sign.dtype, device=sign.device)).any()):
        raise ValueError("sign values must be -1, 0, or 1")
    if bool(torch.isnan(log_absolute_conditional_value).any()):
        raise ValueError("log absolute conditional values must not be NaN")
    zero_mismatch = (sign == 0.0) != torch.isneginf(log_absolute_conditional_value)
    if bool(zero_mismatch.any()):
        raise ValueError("zero signs must correspond exactly to log absolute -infinity")
    likelihood = evaluate_residual_likelihood(residual, spec)
    absolute = torch.exp(log_absolute_conditional_value)
    contribution = sign * torch.exp(
        log_absolute_conditional_value + likelihood.log_likelihood
    )
    if not torch.isfinite(contribution).all():
        raise FloatingPointError("signed residual contribution became nonfinite")
    return SignedResidualContributionEvaluation(
        contribution=contribution,
        sign=sign,
        absolute_conditional_value=absolute,
        likelihood=likelihood,
    )


@dataclass(frozen=True)
class ResidualTransportTrainingConfig:
    components: int = 2
    defensive_weight: float = 0.1
    epochs: int = 200
    learning_rate: float = 0.03
    maximum_mean_norm: float = 20.0
    initialization_scale: float = 0.05
    l2_penalty: float = 1e-6
    learn_component_weights: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.components, bool) or not isinstance(self.components, int) or self.components < 2:
            raise ValueError("components must be an integer of at least two")
        if not 0.0 < self.defensive_weight < 1.0:
            raise ValueError("defensive weight must lie in (0, 1)")
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int) or self.epochs < 1:
            raise ValueError("epochs must be a positive integer")
        positive = (
            self.learning_rate,
            self.maximum_mean_norm,
            self.initialization_scale,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("training scales must be finite and positive")
        if not math.isfinite(self.l2_penalty) or self.l2_penalty < 0.0:
            raise ValueError("l2 penalty must be finite and nonnegative")


@dataclass(frozen=True)
class FrozenResidualTransport:
    schema: str
    task_id: str
    direction: tuple[float, ...]
    component_means: tuple[tuple[float, ...], ...]
    component_weights: tuple[float, ...]
    training_seed: int
    training_objective: str
    exact_likelihood: bool
    self_normalized: bool
    frozen: bool
    training_cost: BaselineCostLedger
    sha256: str

    @property
    def dimension(self) -> int:
        return len(self.direction)

    @property
    def components(self) -> int:
        return len(self.component_weights)

    def spec(self) -> ResidualGaussianMixtureSpec:
        return ResidualGaussianMixtureSpec(
            direction=torch.tensor(self.direction, dtype=torch.float64),
            means=torch.tensor(self.component_means, dtype=torch.float64),
            weights=torch.tensor(self.component_weights, dtype=torch.float64),
        )


def _frozen_payload(
    *,
    task_id: str,
    direction: tuple[float, ...],
    means: tuple[tuple[float, ...], ...],
    weights: tuple[float, ...],
    training_seed: int,
    training_objective: str,
    training_cost: BaselineCostLedger,
) -> dict[str, object]:
    return {
        "schema": "npi.g11.v12-frozen-residual-transport.v1",
        "task_id": task_id,
        "direction": direction,
        "component_means": means,
        "component_weights": weights,
        "training_seed": training_seed,
        "training_objective": training_objective,
        "exact_likelihood": True,
        "self_normalized": False,
        "frozen": True,
        "training_cost": asdict(training_cost),
    }


def freeze_residual_transport(
    *,
    task_id: str,
    spec: ResidualGaussianMixtureSpec,
    training_seed: int,
    training_objective: str,
    training_cost: BaselineCostLedger,
) -> FrozenResidualTransport:
    if not task_id.strip():
        raise ValueError("task_id must be nonempty")
    if isinstance(training_seed, bool) or not isinstance(training_seed, int) or training_seed < 0:
        raise ValueError("training seed must be a nonnegative integer")
    if not training_objective.strip():
        raise ValueError("training objective must be nonempty")
    direction = tuple(float(value) for value in spec.direction)
    means = tuple(tuple(float(value) for value in row) for row in spec.means)
    weights = tuple(float(value) for value in spec.weights)
    payload = _frozen_payload(
        task_id=task_id,
        direction=direction,
        means=means,
        weights=weights,
        training_seed=training_seed,
        training_objective=training_objective,
        training_cost=training_cost,
    )
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return FrozenResidualTransport(
        schema="npi.g11.v12-frozen-residual-transport.v1",
        task_id=task_id,
        direction=direction,
        component_means=means,
        component_weights=weights,
        training_seed=training_seed,
        training_objective=training_objective,
        exact_likelihood=True,
        self_normalized=False,
        frozen=True,
        training_cost=training_cost,
        sha256=digest,
    )


@dataclass(frozen=True)
class ResidualTransportTrainingResult:
    proposal: FrozenResidualTransport
    loss_history: tuple[float, ...]
    normalized_weight_ess: float
    finite_target_count: int
    total_target_count: int


def fit_weighted_residual_transport(
    *,
    task_id: str,
    target_residuals: torch.Tensor,
    log_target_weights: torch.Tensor,
    direction: torch.Tensor,
    training_seed: int,
    config: ResidualTransportTrainingConfig | None = None,
) -> ResidualTransportTrainingResult:
    """Fit ``q`` to the conditional residual target by weighted cross entropy.

    Samples must come from the target residual Gaussian.  Normalizing the training
    weights is allowed because it changes only the scale of the optimization
    objective; final probability estimates remain ordinary importance-sampling means.
    """

    return fit_weighted_residual_transport_from_importance_samples(
        task_id=task_id,
        sampled_residuals=target_residuals,
        log_unnormalized_target_over_sampling=log_target_weights,
        direction=direction,
        training_seed=training_seed,
        config=config,
        training_objective="conditional_target_weighted_cross_entropy",
    )


def fit_weighted_residual_transport_from_importance_samples(
    *,
    task_id: str,
    sampled_residuals: torch.Tensor,
    log_unnormalized_target_over_sampling: torch.Tensor,
    direction: torch.Tensor,
    training_seed: int,
    config: ResidualTransportTrainingConfig | None = None,
    training_objective: str = "importance_weighted_conditional_target_cross_entropy",
) -> ResidualTransportTrainingResult:
    """Fit from any exact sampling law using ``log(target/sampling)`` weights.

    For the conditional target proportional to ``g(r) p_R(r)``, samples drawn
    under a proposal ``q`` must therefore use ``log g + log(p_R/q)``.  Weight
    normalization is confined to the training objective; the final estimator
    remains the ordinary, unnormalized importance-sampling mean.
    """

    config = config or ResidualTransportTrainingConfig()
    target_residuals = sampled_residuals
    log_target_weights = log_unnormalized_target_over_sampling
    if target_residuals.ndim != 2 or target_residuals.shape[0] < 2:
        raise ValueError("target residuals must have shape (samples, dimension)")
    if log_target_weights.shape != (target_residuals.shape[0],):
        raise ValueError("log target weights must have shape (samples,)")
    if target_residuals.device.type != "cpu" or target_residuals.dtype != torch.float64:
        raise ValueError("training residuals must be CPU float64")
    if direction.device.type != "cpu" or direction.dtype != torch.float64:
        raise ValueError("training direction must be CPU float64")
    if not torch.isfinite(target_residuals).all() or bool(torch.isnan(log_target_weights).any()):
        raise ValueError("training data must be finite except negative-infinite weights")
    if bool(torch.isposinf(log_target_weights).any()):
        raise ValueError("positive-infinite target weights are invalid")
    if direction.shape != (target_residuals.shape[1],):
        raise ValueError("training direction has the wrong dimension")
    if abs(float(torch.linalg.vector_norm(direction)) - 1.0) > 1e-10:
        raise ValueError("training direction must be unit norm")
    projection = float(torch.amax(torch.abs(target_residuals @ direction)))
    scale = max(1.0, float(torch.amax(torch.abs(target_residuals))))
    if projection > 1e-10 * scale:
        raise ValueError("training samples are not residuals for the direction")
    finite = torch.isfinite(log_target_weights)
    finite_count = int(torch.count_nonzero(finite))
    if finite_count < 1:
        raise ValueError("at least one training target weight must be positive")
    normalized = torch.softmax(log_target_weights, dim=0).detach()
    ess = 1.0 / float(torch.sum(normalized.square()))
    dimension = target_residuals.shape[1]
    shifted_components = config.components - 1
    generator = torch.Generator(device="cpu").manual_seed(training_seed)
    weighted_mean = torch.sum(normalized.unsqueeze(1) * target_residuals, dim=0)
    initial = weighted_mean.unsqueeze(0).expand(shifted_components, -1).clone()
    initial += config.initialization_scale * torch.randn(
        (shifted_components, dimension), dtype=torch.float64, generator=generator
    )
    initial = project_orthogonal(initial, direction)
    raw_means = torch.nn.Parameter(initial)
    logits = torch.nn.Parameter(torch.zeros(shifted_components, dtype=torch.float64))
    parameters: list[torch.nn.Parameter] = [raw_means]
    if config.learn_component_weights and shifted_components > 1:
        parameters.append(logits)
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    losses: list[float] = []
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    for _ in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        means = project_orthogonal(raw_means, direction)
        if config.learn_component_weights and shifted_components > 1:
            learned_weights = (1.0 - config.defensive_weight) * torch.softmax(logits, dim=0)
        else:
            learned_weights = torch.full(
                (shifted_components,),
                (1.0 - config.defensive_weight) / shifted_components,
                dtype=torch.float64,
            )
        components = target_residuals @ means.T - 0.5 * torch.sum(
            means.square(), dim=1
        ).unsqueeze(0)
        natural = torch.full(
            (target_residuals.shape[0], 1),
            math.log(config.defensive_weight),
            dtype=torch.float64,
        )
        shifted = components + torch.log(learned_weights).unsqueeze(0)
        log_q_over_p = torch.logsumexp(torch.cat((natural, shifted), dim=1), dim=1)
        loss = -torch.sum(normalized * log_q_over_p)
        loss = loss + config.l2_penalty * torch.mean(means.square())
        if not torch.isfinite(loss):
            raise FloatingPointError("residual transport loss became nonfinite")
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            projected = project_orthogonal(raw_means, direction)
            norms = torch.linalg.vector_norm(projected, dim=1)
            factors = torch.clamp(config.maximum_mean_norm / torch.clamp(norms, min=1e-300), max=1.0)
            raw_means.copy_(projected * factors.unsqueeze(1))
        losses.append(float(loss.detach()))
    training_cpu = time.process_time() - cpu_started
    training_wall = time.perf_counter() - wall_started
    with torch.no_grad():
        fitted_means = project_orthogonal(raw_means, direction)
        if config.learn_component_weights and shifted_components > 1:
            fitted_weights = (1.0 - config.defensive_weight) * torch.softmax(logits, dim=0)
        else:
            fitted_weights = torch.full(
                (shifted_components,),
                (1.0 - config.defensive_weight) / shifted_components,
                dtype=torch.float64,
            )
        means = torch.cat((torch.zeros((1, dimension), dtype=torch.float64), fitted_means), dim=0)
        weights = torch.cat((torch.tensor([config.defensive_weight], dtype=torch.float64), fitted_weights))
    spec = ResidualGaussianMixtureSpec(direction=direction.clone(), means=means, weights=weights)
    sample_count = int(target_residuals.shape[0])
    work = sample_count * config.epochs * shifted_components * dimension
    cost = BaselineCostLedger(
        training_samples=sample_count,
        optimizer_steps=config.epochs,
        hyperparameter_trials=1,
        algorithmic_work_units=float(work),
        wall_seconds=training_wall,
        cpu_seconds=training_cpu,
        peak_memory_bytes=process_peak_resident_memory_bytes(),
        measurement_mode="standardized_hardware_wall",
    )
    proposal = freeze_residual_transport(
        task_id=task_id,
        spec=spec,
        training_seed=training_seed,
        training_objective=training_objective,
        training_cost=cost,
    )
    return ResidualTransportTrainingResult(
        proposal=proposal,
        loss_history=tuple(losses),
        normalized_weight_ess=ess,
        finite_target_count=finite_count,
        total_target_count=sample_count,
    )
