"""Common, fail-closed lifecycle contracts for strong rare-event baselines.

The framework standardizes proposal likelihoods, independent inferential units,
integer planning, and training-inclusive cost accounting. It does not claim that
a baseline is competitive merely because its contract passes.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import torch
from scipy.special import ndtri
from scipy.stats import qmc

BaselineMethod = Literal[
    "crude_mc",
    "antithetic_mc",
    "conditional_rbergomi",
    "pure_cem",
    "defensive_cem",
    "smoothing_rqmc",
    "ld_subspace_is",
    "flow_is",
]
ProposalFamily = Literal[
    "target_gaussian",
    "rqmc_target_gaussian",
    "gaussian_shift",
    "gaussian_mixture_shift",
    "coupling_flow",
]
InferentialUnit = Literal["iid_path", "antithetic_pair", "rqmc_randomization"]

BASELINE_METHODS: tuple[BaselineMethod, ...] = (
    "crude_mc",
    "antithetic_mc",
    "conditional_rbergomi",
    "pure_cem",
    "defensive_cem",
    "smoothing_rqmc",
    "ld_subspace_is",
    "flow_is",
)
_TRAINED_METHODS: frozenset[BaselineMethod] = frozenset(
    {"pure_cem", "defensive_cem", "ld_subspace_is", "flow_is"}
)

_METHOD_FAMILY: dict[BaselineMethod, ProposalFamily] = {
    "crude_mc": "target_gaussian",
    "antithetic_mc": "target_gaussian",
    "conditional_rbergomi": "target_gaussian",
    "pure_cem": "gaussian_shift",
    "defensive_cem": "gaussian_mixture_shift",
    "smoothing_rqmc": "rqmc_target_gaussian",
    "ld_subspace_is": "gaussian_mixture_shift",
    "flow_is": "coupling_flow",
}
_METHOD_UNIT: dict[BaselineMethod, InferentialUnit] = {
    "crude_mc": "iid_path",
    "antithetic_mc": "antithetic_pair",
    "conditional_rbergomi": "iid_path",
    "pure_cem": "iid_path",
    "defensive_cem": "iid_path",
    "smoothing_rqmc": "rqmc_randomization",
    "ld_subspace_is": "iid_path",
    "flow_is": "iid_path",
}


@dataclass(frozen=True)
class BaselineCostLedger:
    """Complete cost categories; unused categories must be explicit zeros."""

    training_samples: int = 0
    optimizer_steps: int = 0
    hyperparameter_trials: int = 0
    failed_restarts: int = 0
    screening_samples: int = 0
    planning_samples: int = 0
    final_samples: int = 0
    likelihood_evaluations: int = 0
    cdf_calls: int = 0
    quadrature_calls: int = 0
    algorithmic_work_units: float = 0.0
    wall_seconds: float = 0.0
    cpu_seconds: float = 0.0
    gpu_seconds: float = 0.0
    peak_memory_bytes: int = 0
    compute_cost_usd: float = 0.0
    energy_kwh: float = 0.0
    measurement_mode: str = "not_measured_development"

    def __post_init__(self) -> None:
        integer_fields = (
            self.training_samples,
            self.optimizer_steps,
            self.hyperparameter_trials,
            self.failed_restarts,
            self.screening_samples,
            self.planning_samples,
            self.final_samples,
            self.likelihood_evaluations,
            self.cdf_calls,
            self.quadrature_calls,
            self.peak_memory_bytes,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in integer_fields
        ):
            raise ValueError("cost-ledger integer counts must be nonnegative")
        float_fields = (
            self.algorithmic_work_units,
            self.wall_seconds,
            self.cpu_seconds,
            self.gpu_seconds,
            self.compute_cost_usd,
            self.energy_kwh,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in float_fields):
            raise ValueError("cost-ledger continuous values must be finite and nonnegative")
        if self.measurement_mode not in {
            "not_measured_development",
            "standardized_hardware_wall",
            "provider_billed",
            "energy_metered",
        }:
            raise ValueError("unsupported cost measurement mode")
        if self.measurement_mode == "provider_billed" and self.compute_cost_usd <= 0.0:
            raise ValueError("provider-billed cost mode requires positive compute cost")
        if self.measurement_mode == "energy_metered" and self.energy_kwh <= 0.0:
            raise ValueError("energy-metered cost mode requires positive energy")

    def plus(self, other: BaselineCostLedger) -> BaselineCostLedger:
        """Add ledgers while retaining the least formal measurement mode."""

        modes = {self.measurement_mode, other.measurement_mode}
        if len(modes) == 1:
            mode = self.measurement_mode
        elif "not_measured_development" in modes:
            mode = "not_measured_development"
        else:
            mode = "standardized_hardware_wall"
        return BaselineCostLedger(
            training_samples=self.training_samples + other.training_samples,
            optimizer_steps=self.optimizer_steps + other.optimizer_steps,
            hyperparameter_trials=(
                self.hyperparameter_trials + other.hyperparameter_trials
            ),
            failed_restarts=self.failed_restarts + other.failed_restarts,
            screening_samples=self.screening_samples + other.screening_samples,
            planning_samples=self.planning_samples + other.planning_samples,
            final_samples=self.final_samples + other.final_samples,
            likelihood_evaluations=(
                self.likelihood_evaluations + other.likelihood_evaluations
            ),
            cdf_calls=self.cdf_calls + other.cdf_calls,
            quadrature_calls=self.quadrature_calls + other.quadrature_calls,
            algorithmic_work_units=(
                self.algorithmic_work_units + other.algorithmic_work_units
            ),
            wall_seconds=self.wall_seconds + other.wall_seconds,
            cpu_seconds=self.cpu_seconds + other.cpu_seconds,
            gpu_seconds=self.gpu_seconds + other.gpu_seconds,
            peak_memory_bytes=max(self.peak_memory_bytes, other.peak_memory_bytes),
            compute_cost_usd=self.compute_cost_usd + other.compute_cost_usd,
            energy_kwh=self.energy_kwh + other.energy_kwh,
            measurement_mode=mode,
        )


@dataclass(frozen=True)
class FrozenBaselineProposal:
    """Canonical frozen proposal and exact-density declaration."""

    schema: str
    method: BaselineMethod
    family: ProposalFamily
    task_id: str
    dimension: int
    training_seed: int
    training_budget_work_units: float
    location: tuple[float, ...]
    component_means: tuple[tuple[float, ...], ...]
    component_weights: tuple[float, ...]
    flow_split: int
    flow_scale_matrix: tuple[tuple[float, ...], ...]
    flow_scale_bias: tuple[float, ...]
    flow_shift_matrix: tuple[tuple[float, ...], ...]
    flow_shift_bias: tuple[float, ...]
    flow_max_log_scale: float
    exact_likelihood: bool
    self_normalized: bool
    dcs_extension_eligible: bool
    conditional_integral: str
    frozen: bool
    training_cost: BaselineCostLedger
    sha256: str


@dataclass(frozen=True)
class BaselineAllocationPlan:
    """Pilot-frozen integer allocation in independent inferential units."""

    schema: str
    method: BaselineMethod
    task_id: str
    proposal_sha256: str
    pilot_seed: int
    final_seed: int
    pilot_variance: float
    target_variance: float
    planned_units: int
    points_per_unit: int
    planned_final_samples: int
    inferential_unit: InferentialUnit
    planning_cost: BaselineCostLedger
    frozen: bool


@dataclass(frozen=True)
class BaselineEstimateArtifact:
    """Final ordinary-mean estimate over the declared independent units."""

    schema: str
    method: BaselineMethod
    task_id: str
    proposal_sha256: str
    final_seed: int
    inferential_unit: InferentialUnit
    unit_count: int
    points_per_unit: int
    final_sample_count: int
    estimate: float
    unit_sample_variance: float
    estimator_variance: float
    ordinary_mean: bool
    likelihood_clipped: bool
    final_cost: BaselineCostLedger


@dataclass(frozen=True)
class BaselineLifecycleAudit:
    """Machine-readable audit of one train-plan-estimate chain."""

    checks: tuple[tuple[str, bool], ...]
    failures: tuple[str, ...]
    total_cost: BaselineCostLedger
    passed: bool


def _canonical_payload(
    *,
    method: BaselineMethod,
    family: ProposalFamily,
    task_id: str,
    dimension: int,
    training_seed: int,
    training_budget_work_units: float,
    location: tuple[float, ...],
    component_means: tuple[tuple[float, ...], ...],
    component_weights: tuple[float, ...],
    flow_split: int,
    flow_scale_matrix: tuple[tuple[float, ...], ...],
    flow_scale_bias: tuple[float, ...],
    flow_shift_matrix: tuple[tuple[float, ...], ...],
    flow_shift_bias: tuple[float, ...],
    flow_max_log_scale: float,
    conditional_integral: str,
    training_cost: BaselineCostLedger,
) -> dict[str, object]:
    return {
        "schema": "npi.g11.v8-frozen-baseline-proposal.v1",
        "method": method,
        "family": family,
        "task_id": task_id,
        "dimension": dimension,
        "training_seed": training_seed,
        "training_budget_work_units": training_budget_work_units,
        "location": location,
        "component_means": component_means,
        "component_weights": component_weights,
        "flow_split": flow_split,
        "flow_scale_matrix": flow_scale_matrix,
        "flow_scale_bias": flow_scale_bias,
        "flow_shift_matrix": flow_shift_matrix,
        "flow_shift_bias": flow_shift_bias,
        "flow_max_log_scale": flow_max_log_scale,
        "exact_likelihood": True,
        "self_normalized": False,
        "dcs_extension_eligible": False,
        "conditional_integral": conditional_integral,
        "frozen": True,
        "training_cost": asdict(training_cost),
    }


def _hash_payload(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def freeze_baseline_proposal(
    *,
    method: BaselineMethod,
    task_id: str,
    dimension: int,
    training_seed: int,
    training_cost: BaselineCostLedger,
    training_budget_work_units: float = 0.0,
    location: tuple[float, ...] = (),
    component_means: tuple[tuple[float, ...], ...] = (),
    component_weights: tuple[float, ...] = (),
    flow_split: int = 0,
    flow_scale_matrix: tuple[tuple[float, ...], ...] = (),
    flow_scale_bias: tuple[float, ...] = (),
    flow_shift_matrix: tuple[tuple[float, ...], ...] = (),
    flow_shift_bias: tuple[float, ...] = (),
    flow_max_log_scale: float = 0.0,
    conditional_integral: str = "none",
) -> FrozenBaselineProposal:
    """Freeze one method-specific proposal after training or calibration."""

    if method not in BASELINE_METHODS:
        raise ValueError("unsupported baseline method")
    if not task_id.strip():
        raise ValueError("task_id must be nonempty")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise ValueError("dimension must be a positive integer")
    if (
        isinstance(training_seed, bool)
        or not isinstance(training_seed, int)
        or training_seed < 0
    ):
        raise ValueError("training_seed must be a nonnegative integer")
    if (
        not math.isfinite(training_budget_work_units)
        or training_budget_work_units < 0.0
    ):
        raise ValueError("training budget must be finite and nonnegative")
    if method in _TRAINED_METHODS:
        if training_budget_work_units <= 0.0:
            raise ValueError("trained baseline requires a positive training budget")
        if training_cost.algorithmic_work_units <= 0.0:
            raise ValueError("trained baseline must charge positive training work")
        if training_cost.algorithmic_work_units > training_budget_work_units:
            raise ValueError("actual training work exceeds the declared budget")
    elif training_budget_work_units != 0.0:
        raise ValueError("non-trained baseline must use a zero training budget")
    family = _METHOD_FAMILY[method]
    location_array = np.asarray(location, dtype=np.float64)
    means_array = np.asarray(component_means, dtype=np.float64)
    weights_array = np.asarray(component_weights, dtype=np.float64)
    scale_matrix_array = np.asarray(flow_scale_matrix, dtype=np.float64)
    scale_bias_array = np.asarray(flow_scale_bias, dtype=np.float64)
    shift_matrix_array = np.asarray(flow_shift_matrix, dtype=np.float64)
    shift_bias_array = np.asarray(flow_shift_bias, dtype=np.float64)
    if any(
        not np.isfinite(array).all()
        for array in (
            location_array,
            means_array,
            weights_array,
            scale_matrix_array,
            scale_bias_array,
            shift_matrix_array,
            shift_bias_array,
        )
    ):
        raise ValueError("proposal parameters must be finite")

    if family in {"target_gaussian", "rqmc_target_gaussian"}:
        if (
            location
            or component_means
            or component_weights
            or flow_scale_matrix
            or flow_shift_matrix
        ):
            raise ValueError("target proposal must not contain tilt parameters")
    elif family == "gaussian_shift":
        if location_array.shape != (dimension,):
            raise ValueError("Gaussian shift location must match dimension")
        if component_means or component_weights or flow_scale_matrix:
            raise ValueError("Gaussian shift has only one location parameter")
    elif family == "gaussian_mixture_shift":
        if (
            means_array.ndim != 2
            or means_array.shape[0] < 1
            or means_array.shape[1] != dimension
        ):
            raise ValueError("mixture component means must have shape (J, dimension)")
        if weights_array.shape != (means_array.shape[0],):
            raise ValueError("mixture weights must match component count")
        if np.any(weights_array <= 0.0) or not math.isclose(
            float(np.sum(weights_array)),
            1.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("mixture weights must be positive and sum to one")
        if location or flow_scale_matrix:
            raise ValueError("mixture proposal must not contain unrelated parameters")
        if method == "defensive_cem" and not np.any(
            np.all(means_array == 0.0, axis=1)
        ):
            raise ValueError("defensive CEM requires a natural zero-mean component")
    elif family == "coupling_flow":
        if location_array.shape != (dimension,):
            raise ValueError("flow location must match dimension")
        if (
            isinstance(flow_split, bool)
            or not isinstance(flow_split, int)
            or not 0 < flow_split < dimension
        ):
            raise ValueError("flow split must lie strictly inside the dimension")
        transformed = dimension - flow_split
        expected_matrix = (transformed, flow_split)
        if (
            scale_matrix_array.shape != expected_matrix
            or shift_matrix_array.shape != expected_matrix
            or scale_bias_array.shape != (transformed,)
            or shift_bias_array.shape != (transformed,)
        ):
            raise ValueError("flow coupling parameter shapes do not match split")
        if not math.isfinite(flow_max_log_scale) or flow_max_log_scale <= 0.0:
            raise ValueError("flow maximum log scale must be finite and positive")
        if component_means or component_weights:
            raise ValueError("coupling flow must not contain mixture parameters")
        if conditional_integral != "baseline_only":
            raise ValueError("entangling flow is a baseline, not a DCS extension")

    if method == "conditional_rbergomi" and conditional_integral not in {
        "analytic_gaussian_cdf",
        "rigorous_quadrature",
    }:
        raise ValueError("conditional rBergomi must declare its conditional integral")
    if method == "smoothing_rqmc" and conditional_integral not in {
        "analytic_gaussian_cdf",
        "rigorous_quadrature",
    }:
        raise ValueError("smoothing RQMC must declare its smoothing integral")

    payload = _canonical_payload(
        method=method,
        family=family,
        task_id=task_id,
        dimension=dimension,
        training_seed=training_seed,
        training_budget_work_units=training_budget_work_units,
        location=location,
        component_means=component_means,
        component_weights=component_weights,
        flow_split=flow_split,
        flow_scale_matrix=flow_scale_matrix,
        flow_scale_bias=flow_scale_bias,
        flow_shift_matrix=flow_shift_matrix,
        flow_shift_bias=flow_shift_bias,
        flow_max_log_scale=flow_max_log_scale,
        conditional_integral=conditional_integral,
        training_cost=training_cost,
    )
    digest = _hash_payload(payload)
    return FrozenBaselineProposal(
        schema="npi.g11.v8-frozen-baseline-proposal.v1",
        method=method,
        family=family,
        task_id=task_id,
        dimension=dimension,
        training_seed=training_seed,
        training_budget_work_units=training_budget_work_units,
        location=location,
        component_means=component_means,
        component_weights=component_weights,
        flow_split=flow_split,
        flow_scale_matrix=flow_scale_matrix,
        flow_scale_bias=flow_scale_bias,
        flow_shift_matrix=flow_shift_matrix,
        flow_shift_bias=flow_shift_bias,
        flow_max_log_scale=flow_max_log_scale,
        exact_likelihood=True,
        self_normalized=False,
        dcs_extension_eligible=False,
        conditional_integral=conditional_integral,
        frozen=True,
        training_cost=training_cost,
        sha256=digest,
    )


def evaluate_baseline_log_q_over_p(
    samples: torch.Tensor,
    proposal: FrozenBaselineProposal,
) -> torch.Tensor:
    """Evaluate the exact proposal-to-target log density on target coordinates."""

    if samples.ndim != 2 or samples.shape[1] != proposal.dimension:
        raise ValueError("samples must have shape (batch, proposal.dimension)")
    if not samples.is_floating_point() or not torch.isfinite(samples).all():
        raise ValueError("samples must be finite floating point")
    if not proposal.exact_likelihood or proposal.self_normalized:
        raise ValueError("baseline proposal violates the exact-likelihood contract")

    if proposal.family in {"target_gaussian", "rqmc_target_gaussian"}:
        return torch.zeros(samples.shape[0], dtype=samples.dtype, device=samples.device)
    if proposal.family == "gaussian_shift":
        mean = torch.tensor(
            proposal.location,
            dtype=samples.dtype,
            device=samples.device,
        )
        return samples @ mean - 0.5 * torch.sum(mean.square())
    if proposal.family == "gaussian_mixture_shift":
        means = torch.tensor(
            proposal.component_means,
            dtype=samples.dtype,
            device=samples.device,
        )
        weights = torch.tensor(
            proposal.component_weights,
            dtype=samples.dtype,
            device=samples.device,
        )
        component = samples @ means.T - 0.5 * torch.sum(means.square(), dim=1)
        return torch.logsumexp(component + torch.log(weights), dim=1)
    if proposal.family == "coupling_flow":
        location = torch.tensor(
            proposal.location,
            dtype=samples.dtype,
            device=samples.device,
        )
        scale_matrix = torch.tensor(
            proposal.flow_scale_matrix,
            dtype=samples.dtype,
            device=samples.device,
        )
        scale_bias = torch.tensor(
            proposal.flow_scale_bias,
            dtype=samples.dtype,
            device=samples.device,
        )
        shift_matrix = torch.tensor(
            proposal.flow_shift_matrix,
            dtype=samples.dtype,
            device=samples.device,
        )
        shift_bias = torch.tensor(
            proposal.flow_shift_bias,
            dtype=samples.dtype,
            device=samples.device,
        )
        split = proposal.flow_split
        latent_first = samples[:, :split] - location[:split]
        log_scale = proposal.flow_max_log_scale * torch.tanh(
            latent_first @ scale_matrix.T + scale_bias
        )
        shift = latent_first @ shift_matrix.T + shift_bias
        latent_second = (
            samples[:, split:] - location[split:] - shift
        ) * torch.exp(-log_scale)
        latent = torch.cat((latent_first, latent_second), dim=1)
        return 0.5 * (
            torch.sum(samples.square(), dim=1)
            - torch.sum(latent.square(), dim=1)
        ) - torch.sum(log_scale, dim=1)
    raise AssertionError("unreachable proposal family")


def sample_baseline_proposal(
    proposal: FrozenBaselineProposal,
    *,
    sample_count: int,
    seed: int,
) -> torch.Tensor:
    """Draw proposal points; RQMC returns one scrambled Sobol randomization."""

    if isinstance(sample_count, bool) or sample_count < 1:
        raise ValueError("sample_count must be a positive integer")
    if isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    dtype = torch.float64
    dimension = proposal.dimension

    if proposal.method == "antithetic_mc":
        if sample_count % 2:
            raise ValueError("antithetic sample count must be even")
        half = torch.randn(
            (sample_count // 2, dimension),
            generator=generator,
            dtype=dtype,
        )
        return torch.stack((half, -half), dim=1).reshape(sample_count, dimension)
    if proposal.family == "rqmc_target_gaussian":
        if sample_count & (sample_count - 1):
            raise ValueError("RQMC points per randomization must be a power of two")
        engine = qmc.Sobol(d=dimension, scramble=True, seed=seed)
        uniforms = engine.random_base2(int(math.log2(sample_count)))
        lower = np.nextafter(0.0, 1.0)
        upper = np.nextafter(1.0, 0.0)
        normals = ndtri(np.clip(uniforms, lower, upper))
        return torch.from_numpy(normals).to(dtype=dtype)

    latent = torch.randn((sample_count, dimension), generator=generator, dtype=dtype)
    if proposal.family == "target_gaussian":
        return latent
    if proposal.family == "gaussian_shift":
        return latent + torch.tensor(proposal.location, dtype=dtype)
    if proposal.family == "gaussian_mixture_shift":
        weights = torch.tensor(proposal.component_weights, dtype=dtype)
        labels = torch.multinomial(
            weights,
            num_samples=sample_count,
            replacement=True,
            generator=generator,
        )
        means = torch.tensor(proposal.component_means, dtype=dtype)
        return latent + means[labels]
    if proposal.family == "coupling_flow":
        location = torch.tensor(proposal.location, dtype=dtype)
        scale_matrix = torch.tensor(proposal.flow_scale_matrix, dtype=dtype)
        scale_bias = torch.tensor(proposal.flow_scale_bias, dtype=dtype)
        shift_matrix = torch.tensor(proposal.flow_shift_matrix, dtype=dtype)
        shift_bias = torch.tensor(proposal.flow_shift_bias, dtype=dtype)
        split = proposal.flow_split
        first = latent[:, :split]
        log_scale = proposal.flow_max_log_scale * torch.tanh(
            first @ scale_matrix.T + scale_bias
        )
        shift = first @ shift_matrix.T + shift_bias
        second = torch.exp(log_scale) * latent[:, split:] + shift
        return torch.cat((first, second), dim=1) + location
    raise AssertionError("unreachable proposal family")


def ordinary_is_contributions(
    event_or_conditional_value: torch.Tensor,
    log_q_over_p: torch.Tensor,
) -> torch.Tensor:
    """Return unclipped ordinary-IS contributions ``F * dP/dQ``."""

    if (
        event_or_conditional_value.ndim != 1
        or log_q_over_p.ndim != 1
        or event_or_conditional_value.shape != log_q_over_p.shape
    ):
        raise ValueError("values and log densities must be matching vectors")
    if (
        not event_or_conditional_value.is_floating_point()
        or event_or_conditional_value.dtype != log_q_over_p.dtype
        or event_or_conditional_value.device != log_q_over_p.device
    ):
        raise ValueError("values and log densities must share floating dtype/device")
    if (
        not torch.isfinite(event_or_conditional_value).all()
        or not torch.isfinite(log_q_over_p).all()
    ):
        raise ValueError("ordinary-IS inputs must be finite")
    if bool((event_or_conditional_value < 0.0).any()) or bool(
        (event_or_conditional_value > 1.0).any()
    ):
        raise ValueError("event or conditional values must lie in [0, 1]")
    contributions = event_or_conditional_value * torch.exp(-log_q_over_p)
    if not torch.isfinite(contributions).all():
        raise FloatingPointError("ordinary IS contribution is not representable")
    return contributions


def plan_baseline_allocation(
    proposal: FrozenBaselineProposal,
    *,
    pilot_variance: float,
    target_variance: float,
    pilot_seed: int,
    final_seed: int,
    pilot_units: int,
    minimum_units: int = 2,
    maximum_units: int = 10**9,
    points_per_unit: int | None = None,
    planning_cost: BaselineCostLedger | None = None,
) -> BaselineAllocationPlan:
    """Freeze an integer achieved-variance allocation from pilot data only."""

    numeric = (pilot_variance, target_variance)
    if any(not math.isfinite(value) for value in numeric):
        raise ValueError("pilot and target variances must be finite")
    if pilot_variance < 0.0 or target_variance <= 0.0:
        raise ValueError("pilot variance must be nonnegative and target positive")
    integers = (pilot_seed, final_seed, pilot_units, minimum_units, maximum_units)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in integers
    ):
        raise ValueError("allocation seeds and counts must be nonnegative integers")
    if pilot_units < 2 or minimum_units < 2 or maximum_units < minimum_units:
        raise ValueError("invalid pilot/minimum/maximum inferential-unit counts")
    if len({proposal.training_seed, pilot_seed, final_seed}) != 3:
        raise ValueError("training, pilot, and final seeds must be disjoint")

    unit = _METHOD_UNIT[proposal.method]
    if points_per_unit is None:
        if unit == "antithetic_pair":
            points = 2
        elif unit == "rqmc_randomization":
            raise ValueError("RQMC requires explicit points_per_unit")
        else:
            points = 1
    else:
        points = points_per_unit
    if isinstance(points, bool) or not isinstance(points, int) or points < 1:
        raise ValueError("points_per_unit must be a positive integer")
    if unit == "antithetic_pair" and points != 2:
        raise ValueError("one antithetic inferential unit contains exactly two points")
    if unit == "rqmc_randomization" and points & (points - 1):
        raise ValueError("RQMC points_per_unit must be a power of two")
    if unit == "iid_path" and points != 1:
        raise ValueError("iid baseline uses one path per inferential unit")

    required = math.ceil(pilot_variance / target_variance)
    planned_units = min(max(required, minimum_units), maximum_units)
    raw_pilot_samples = pilot_units * points
    cost = planning_cost or BaselineCostLedger(
        planning_samples=raw_pilot_samples
    )
    if cost.planning_samples < raw_pilot_samples:
        raise ValueError("planning cost omits raw pilot samples")
    return BaselineAllocationPlan(
        schema="npi.g11.v8-baseline-allocation.v1",
        method=proposal.method,
        task_id=proposal.task_id,
        proposal_sha256=proposal.sha256,
        pilot_seed=pilot_seed,
        final_seed=final_seed,
        pilot_variance=pilot_variance,
        target_variance=target_variance,
        planned_units=planned_units,
        points_per_unit=points,
        planned_final_samples=planned_units * points,
        inferential_unit=unit,
        planning_cost=cost,
        frozen=True,
    )


def finalize_baseline_estimate(
    proposal: FrozenBaselineProposal,
    plan: BaselineAllocationPlan,
    unit_contributions: torch.Tensor,
    *,
    final_cost: BaselineCostLedger,
) -> BaselineEstimateArtifact:
    """Finalize an ordinary mean over iid paths, pairs, or RQMC randomizations."""

    values = torch.as_tensor(
        unit_contributions,
        dtype=torch.float64,
        device="cpu",
    ).reshape(-1)
    if values.numel() != plan.planned_units:
        raise ValueError("unit contribution count does not match frozen allocation")
    if values.numel() < 2 or not torch.isfinite(values).all():
        raise ValueError("final inferential units must be finite and contain at least two")
    if proposal.sha256 != plan.proposal_sha256:
        raise ValueError("plan does not bind the supplied frozen proposal")
    if final_cost.final_samples != plan.planned_final_samples:
        raise ValueError("final cost does not match planned raw sample count")
    if proposal.family not in {"target_gaussian", "rqmc_target_gaussian"} and (
        final_cost.likelihood_evaluations < plan.planned_final_samples
    ):
        raise ValueError("final cost omits proposal likelihood evaluations")

    sample_variance = float(torch.var(values, unbiased=True))
    return BaselineEstimateArtifact(
        schema="npi.g11.v8-baseline-estimate.v1",
        method=proposal.method,
        task_id=proposal.task_id,
        proposal_sha256=proposal.sha256,
        final_seed=plan.final_seed,
        inferential_unit=plan.inferential_unit,
        unit_count=plan.planned_units,
        points_per_unit=plan.points_per_unit,
        final_sample_count=plan.planned_final_samples,
        estimate=float(torch.mean(values)),
        unit_sample_variance=sample_variance,
        estimator_variance=sample_variance / plan.planned_units,
        ordinary_mean=True,
        likelihood_clipped=False,
        final_cost=final_cost,
    )


def audit_baseline_lifecycle(
    proposal: FrozenBaselineProposal,
    plan: BaselineAllocationPlan,
    estimate: BaselineEstimateArtifact,
) -> BaselineLifecycleAudit:
    """Audit hashes, seeds, units, exact likelihood, and total cost categories."""

    payload = _canonical_payload(
        method=proposal.method,
        family=proposal.family,
        task_id=proposal.task_id,
        dimension=proposal.dimension,
        training_seed=proposal.training_seed,
        training_budget_work_units=proposal.training_budget_work_units,
        location=proposal.location,
        component_means=proposal.component_means,
        component_weights=proposal.component_weights,
        flow_split=proposal.flow_split,
        flow_scale_matrix=proposal.flow_scale_matrix,
        flow_scale_bias=proposal.flow_scale_bias,
        flow_shift_matrix=proposal.flow_shift_matrix,
        flow_shift_bias=proposal.flow_shift_bias,
        flow_max_log_scale=proposal.flow_max_log_scale,
        conditional_integral=proposal.conditional_integral,
        training_cost=proposal.training_cost,
    )
    expected_hash = _hash_payload(payload)
    total_cost = proposal.training_cost.plus(plan.planning_cost).plus(
        estimate.final_cost
    )
    checks = {
        "proposal_hash_exact": proposal.sha256 == expected_hash,
        "proposal_frozen": proposal.frozen,
        "plan_frozen": plan.frozen,
        "method_consistent": proposal.method == plan.method == estimate.method,
        "task_consistent": proposal.task_id == plan.task_id == estimate.task_id,
        "proposal_binding_consistent": (
            proposal.sha256 == plan.proposal_sha256 == estimate.proposal_sha256
        ),
        "seeds_disjoint": len(
            {proposal.training_seed, plan.pilot_seed, plan.final_seed}
        )
        == 3,
        "final_seed_bound": plan.final_seed == estimate.final_seed,
        "exact_likelihood": proposal.exact_likelihood,
        "no_self_normalization": not proposal.self_normalized,
        "ordinary_mean": estimate.ordinary_mean,
        "no_likelihood_clipping": not estimate.likelihood_clipped,
        "inferential_unit_exact": (
            plan.inferential_unit
            == estimate.inferential_unit
            == _METHOD_UNIT[proposal.method]
        ),
        "integer_allocation_exact": (
            plan.planned_units == estimate.unit_count
            and plan.planned_final_samples == estimate.final_sample_count
            and plan.planned_final_samples
            == plan.planned_units * plan.points_per_unit
        ),
        "final_samples_charged": (
            estimate.final_cost.final_samples == estimate.final_sample_count
        ),
        "likelihoods_charged": (
            proposal.family in {"target_gaussian", "rqmc_target_gaussian"}
            or estimate.final_cost.likelihood_evaluations
            >= estimate.final_sample_count
        ),
        "conditional_integrals_charged": (
            proposal.conditional_integral == "none"
            or proposal.conditional_integral == "baseline_only"
            or (
                proposal.conditional_integral == "analytic_gaussian_cdf"
                and estimate.final_cost.cdf_calls >= estimate.final_sample_count
            )
            or (
                proposal.conditional_integral == "rigorous_quadrature"
                and estimate.final_cost.quadrature_calls
                >= estimate.final_sample_count
            )
        ),
        "training_budget_respected": (
            (
                proposal.method in _TRAINED_METHODS
                and proposal.training_budget_work_units > 0.0
                and 0.0
                < proposal.training_cost.algorithmic_work_units
                <= proposal.training_budget_work_units
            )
            or (
                proposal.method not in _TRAINED_METHODS
                and proposal.training_budget_work_units == 0.0
            )
        ),
        "finite_estimate": all(
            math.isfinite(value)
            for value in (
                estimate.estimate,
                estimate.unit_sample_variance,
                estimate.estimator_variance,
            )
        ),
        "variance_relation_exact": math.isclose(
            estimate.estimator_variance,
            estimate.unit_sample_variance / estimate.unit_count,
            rel_tol=1e-13,
            abs_tol=1e-15,
        ),
        "flow_not_relabelled_dcs": (
            proposal.method != "flow_is" or not proposal.dcs_extension_eligible
        ),
        "rqmc_uses_randomization_units": (
            proposal.method != "smoothing_rqmc"
            or (
                estimate.inferential_unit == "rqmc_randomization"
                and plan.points_per_unit & (plan.points_per_unit - 1) == 0
            )
        ),
        "antithetic_uses_pair_units": (
            proposal.method != "antithetic_mc"
            or (
                estimate.inferential_unit == "antithetic_pair"
                and plan.points_per_unit == 2
            )
        ),
        "total_work_nonzero": total_cost.algorithmic_work_units > 0.0,
        "heterogeneous_compute_cost_backed": (
            total_cost.gpu_seconds == 0.0
            or total_cost.compute_cost_usd > 0.0
            or total_cost.energy_kwh > 0.0
        ),
    }
    failures = tuple(name for name, passed in checks.items() if not passed)
    return BaselineLifecycleAudit(
        checks=tuple(checks.items()),
        failures=failures,
        total_cost=total_cost,
        passed=not failures,
    )
