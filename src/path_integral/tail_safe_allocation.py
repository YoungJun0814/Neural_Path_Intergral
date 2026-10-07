"""Distribution-free allocation guards and streaming sufficient statistics.

The legacy benchmark lifecycle intentionally remains unchanged because its artifacts
are frozen.  New protocols use this module so that a rare pilot with zero empirical
variance cannot certify a zero-variance estimator.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import torch

UnitKind = Literal["iid_path", "antithetic_pair", "rqmc_randomization"]


@dataclass
class StreamingMoments:
    """Mergeable float64 Chan/Welford statistics for inferential units."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    sum_squares: float = 0.0
    sum_fourth_powers: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf
    nonzero_count: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 0:
            raise ValueError("streaming count must be a nonnegative integer")
        if (
            isinstance(self.nonzero_count, bool)
            or not isinstance(self.nonzero_count, int)
            or not 0 <= self.nonzero_count <= self.count
        ):
            raise ValueError("streaming nonzero count is invalid")
        numeric = (self.mean, self.m2, self.sum_squares, self.sum_fourth_powers)
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("streaming moments must be finite")
        if self.m2 < -1e-12 or self.sum_squares < -1e-12 or self.sum_fourth_powers < -1e-12:
            raise ValueError("streaming squared moments must be nonnegative")
        if self.count == 0:
            if self.minimum != math.inf or self.maximum != -math.inf:
                raise ValueError("empty streaming moments require infinite sentinels")
        elif (
            not math.isfinite(self.minimum)
            or not math.isfinite(self.maximum)
            or self.minimum > self.maximum
        ):
            raise ValueError("nonempty streaming extrema are invalid")

    @property
    def sample_variance(self) -> float:
        if self.count < 2:
            raise ValueError("sample variance requires at least two units")
        return max(0.0, self.m2 / (self.count - 1))

    @property
    def mean_square(self) -> float:
        if self.count < 1:
            raise ValueError("mean square requires at least one unit")
        return max(0.0, self.sum_squares / self.count)

    @property
    def mean_square_sample_variance(self) -> float:
        """Unbiased sample variance of the transformed units ``Y**2``."""

        if self.count < 2:
            raise ValueError("mean-square sample variance requires at least two units")
        centered = self.sum_fourth_powers - self.count * self.mean_square**2
        scale = max(1.0, self.sum_fourth_powers)
        if centered < -1e-12 * scale:
            raise FloatingPointError("fourth moments are internally inconsistent")
        return max(0.0, centered / (self.count - 1))

    def update(self, values: Sequence[float] | torch.Tensor) -> None:
        sample = torch.as_tensor(values, dtype=torch.float64, device="cpu").reshape(-1)
        if sample.numel() < 1 or not torch.isfinite(sample).all():
            raise ValueError("streaming update requires finite nonempty values")
        batch_count = int(sample.numel())
        batch_mean = float(torch.mean(sample))
        centered = sample - batch_mean
        batch_m2 = float(torch.sum(centered.square()))
        batch_squares = float(torch.sum(sample.square()))
        batch_fourth_powers = float(torch.sum(sample.square().square()))
        batch_minimum = float(torch.amin(sample))
        batch_maximum = float(torch.amax(sample))
        batch_nonzero = int(torch.count_nonzero(sample))
        self.merge(
            StreamingMoments(
                count=batch_count,
                mean=batch_mean,
                m2=batch_m2,
                sum_squares=batch_squares,
                sum_fourth_powers=batch_fourth_powers,
                minimum=batch_minimum,
                maximum=batch_maximum,
                nonzero_count=batch_nonzero,
            )
        )

    def merge(self, other: StreamingMoments) -> None:
        """Merge another accumulator with the Chan parallel-variance identity."""

        if not isinstance(other, StreamingMoments):
            raise TypeError("can only merge StreamingMoments")
        if other.count == 0:
            return
        if self.count == 0:
            self.count = other.count
            self.mean = other.mean
            self.m2 = other.m2
            self.sum_squares = other.sum_squares
            self.sum_fourth_powers = other.sum_fourth_powers
            self.minimum = other.minimum
            self.maximum = other.maximum
            self.nonzero_count = other.nonzero_count
            return
        total = self.count + other.count
        delta = other.mean - self.mean
        self.m2 = self.m2 + other.m2 + delta * delta * self.count * other.count / total
        self.mean += delta * other.count / total
        self.sum_squares += other.sum_squares
        self.sum_fourth_powers += other.sum_fourth_powers
        self.minimum = min(self.minimum, other.minimum)
        self.maximum = max(self.maximum, other.maximum)
        self.nonzero_count += other.nonzero_count
        self.count = total


@dataclass(frozen=True)
class BoundedRange:
    """Almost-sure range declared before inspecting pilot or final outcomes."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.lower) or not math.isfinite(self.upper):
            raise ValueError("bounds must be finite")
        if self.lower >= self.upper:
            raise ValueError("lower bound must be strictly below upper bound")

    @property
    def maximum_absolute(self) -> float:
        return max(abs(self.lower), abs(self.upper))

    @property
    def popoviciu_variance_bound(self) -> float:
        return (self.upper - self.lower) ** 2 / 4.0


@dataclass(frozen=True)
class BoundedVarianceCertificate:
    """One-sided distribution-free variance certificate from bounded samples."""

    count: int
    confidence_level: float
    empirical_mean: float
    empirical_sample_variance: float
    empirical_mean_square: float
    mean_square_upper: float
    variance_upper: float
    bounds: BoundedRange


@dataclass(frozen=True)
class EmpiricalBernsteinVarianceCertificate(BoundedVarianceCertificate):
    """Bonferroni-valid intersection of Hoeffding and empirical Bernstein UCBs."""

    squared_unit_sample_variance: float
    hoeffding_mean_square_upper: float
    empirical_bernstein_mean_square_upper: float
    popoviciu_variance_upper: float
    per_bound_failure_probability: float
    simultaneous_confidence_level: float


def bounded_variance_certificate(
    moments: StreamingMoments,
    *,
    bounds: BoundedRange,
    confidence_level: float = 0.95,
    tolerance: float = 1e-12,
) -> BoundedVarianceCertificate:
    """Bound ``Var(Y)`` by Hoeffding on ``Y^2`` and Popoviciu's inequality.

    The inferential units must be independent.  In particular, individual points in
    one scrambled Sobol randomization are not valid inputs; the randomization average
    is one unit.
    """

    if moments.count < 2:
        raise ValueError("a variance certificate requires at least two units")
    if not math.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence level must lie in (0, 1)")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")
    scale = max(1.0, abs(bounds.lower), abs(bounds.upper))
    if moments.minimum < bounds.lower - tolerance * scale:
        raise ValueError("observed value violates the declared lower bound")
    if moments.maximum > bounds.upper + tolerance * scale:
        raise ValueError("observed value violates the declared upper bound")
    maximum_square = bounds.maximum_absolute**2
    if maximum_square <= 0.0:
        raise ValueError("declared range has no nonzero square scale")
    alpha = 1.0 - confidence_level
    normalized_mean_square = min(1.0, max(0.0, moments.mean_square / maximum_square))
    radius = math.sqrt(math.log(1.0 / alpha) / (2.0 * moments.count))
    mean_square_upper = maximum_square * min(1.0, normalized_mean_square + radius)
    variance_upper = min(mean_square_upper, bounds.popoviciu_variance_bound)
    return BoundedVarianceCertificate(
        count=moments.count,
        confidence_level=confidence_level,
        empirical_mean=moments.mean,
        empirical_sample_variance=moments.sample_variance,
        empirical_mean_square=moments.mean_square,
        mean_square_upper=mean_square_upper,
        variance_upper=variance_upper,
        bounds=bounds,
    )


def empirical_bernstein_variance_certificate(
    moments: StreamingMoments,
    *,
    bounds: BoundedRange,
    confidence_level: float = 0.95,
    tolerance: float = 1e-12,
) -> EmpiricalBernsteinVarianceCertificate:
    """Certify ``Var(Y)`` using a simultaneous Hoeffding/EB intersection.

    For ``Z=Y**2/M**2 in [0,1]``, each stochastic upper bound receives failure
    probability ``(1-confidence_level)/2``.  The union bound therefore makes
    their intersection valid at the requested confidence.  Popoviciu's bound
    is deterministic and consumes no error probability.
    """

    if moments.count < 2:
        raise ValueError("an empirical-Bernstein certificate requires at least two units")
    if not math.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence level must lie in (0, 1)")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")
    scale = max(1.0, abs(bounds.lower), abs(bounds.upper))
    if moments.minimum < bounds.lower - tolerance * scale:
        raise ValueError("observed value violates the declared lower bound")
    if moments.maximum > bounds.upper + tolerance * scale:
        raise ValueError("observed value violates the declared upper bound")
    maximum_square = bounds.maximum_absolute**2
    if maximum_square <= 0.0:
        raise ValueError("declared range has no nonzero square scale")

    count = moments.count
    normalized_mean = min(1.0, max(0.0, moments.mean_square / maximum_square))
    normalized_sample_variance = min(
        1.0,
        max(0.0, moments.mean_square_sample_variance / maximum_square**2),
    )
    per_bound_alpha = (1.0 - confidence_level) / 2.0
    hoeffding_radius = math.sqrt(math.log(1.0 / per_bound_alpha) / (2.0 * count))
    empirical_log = math.log(2.0 / per_bound_alpha)
    empirical_radius = math.sqrt(
        2.0 * normalized_sample_variance * empirical_log / count
    ) + 7.0 * empirical_log / (3.0 * (count - 1))
    hoeffding_upper = maximum_square * min(1.0, normalized_mean + hoeffding_radius)
    empirical_upper = maximum_square * min(1.0, normalized_mean + empirical_radius)
    mean_square_upper = min(hoeffding_upper, empirical_upper)
    popoviciu = bounds.popoviciu_variance_bound
    variance_upper = min(mean_square_upper, popoviciu)
    return EmpiricalBernsteinVarianceCertificate(
        count=count,
        confidence_level=confidence_level,
        empirical_mean=moments.mean,
        empirical_sample_variance=moments.sample_variance,
        empirical_mean_square=moments.mean_square,
        mean_square_upper=mean_square_upper,
        variance_upper=variance_upper,
        bounds=bounds,
        squared_unit_sample_variance=moments.mean_square_sample_variance,
        hoeffding_mean_square_upper=hoeffding_upper,
        empirical_bernstein_mean_square_upper=empirical_upper,
        popoviciu_variance_upper=popoviciu,
        per_bound_failure_probability=per_bound_alpha,
        simultaneous_confidence_level=1.0 - 2.0 * per_bound_alpha,
    )


@dataclass(frozen=True)
class TailSafeAllocationPolicy:
    confidence_level: float = 0.95
    minimum_iid_units: int = 32
    minimum_rqmc_randomizations: int = 16
    maximum_units: int = 10**8
    streaming_chunk_units: int = 65536

    def __post_init__(self) -> None:
        if not math.isfinite(self.confidence_level) or not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence level must lie in (0, 1)")
        integers = (
            self.minimum_iid_units,
            self.minimum_rqmc_randomizations,
            self.maximum_units,
            self.streaming_chunk_units,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 2 for value in integers
        ):
            raise ValueError("tail-safe unit counts must be integers of at least two")
        if self.minimum_iid_units > self.maximum_units:
            raise ValueError("IID minimum exceeds maximum units")
        if self.minimum_rqmc_randomizations > self.maximum_units:
            raise ValueError("RQMC minimum exceeds maximum units")


@dataclass(frozen=True)
class TailSafeAllocationPlan:
    unit_kind: UnitKind
    target_estimator_variance: float
    pilot_certificate: BoundedVarianceCertificate
    plugin_required_units: int
    certified_required_units: int
    planned_units: int
    resource_censored: bool
    policy: TailSafeAllocationPolicy


def plan_tail_safe_allocation(
    pilot: StreamingMoments,
    *,
    bounds: BoundedRange,
    target_estimator_variance: float,
    unit_kind: UnitKind,
    policy: TailSafeAllocationPolicy | None = None,
) -> TailSafeAllocationPlan:
    """Plan from a one-sided variance upper bound; plugin variance is diagnostic."""

    policy = policy or TailSafeAllocationPolicy()
    if not math.isfinite(target_estimator_variance) or target_estimator_variance <= 0.0:
        raise ValueError("target estimator variance must be finite and positive")
    if unit_kind not in {"iid_path", "antithetic_pair", "rqmc_randomization"}:
        raise ValueError("unsupported inferential unit kind")
    certificate = bounded_variance_certificate(
        pilot,
        bounds=bounds,
        confidence_level=policy.confidence_level,
    )
    minimum = (
        policy.minimum_rqmc_randomizations
        if unit_kind == "rqmc_randomization"
        else policy.minimum_iid_units
    )
    plugin_required = max(
        minimum,
        math.ceil(certificate.empirical_sample_variance / target_estimator_variance),
    )
    certified_required = max(
        minimum,
        math.ceil(certificate.variance_upper / target_estimator_variance),
    )
    censored = certified_required > policy.maximum_units
    return TailSafeAllocationPlan(
        unit_kind=unit_kind,
        target_estimator_variance=target_estimator_variance,
        pilot_certificate=certificate,
        plugin_required_units=plugin_required,
        certified_required_units=certified_required,
        planned_units=min(certified_required, policy.maximum_units),
        resource_censored=censored,
        policy=policy,
    )


def plan_empirical_bernstein_tail_safe_allocation(
    pilot: StreamingMoments,
    *,
    bounds: BoundedRange,
    target_estimator_variance: float,
    unit_kind: UnitKind,
    policy: TailSafeAllocationPolicy | None = None,
) -> TailSafeAllocationPlan:
    """V13 allocation using the simultaneous Hoeffding/EB certificate."""

    policy = policy or TailSafeAllocationPolicy()
    if not math.isfinite(target_estimator_variance) or target_estimator_variance <= 0.0:
        raise ValueError("target estimator variance must be finite and positive")
    if unit_kind not in {"iid_path", "antithetic_pair", "rqmc_randomization"}:
        raise ValueError("unsupported inferential unit kind")
    certificate = empirical_bernstein_variance_certificate(
        pilot,
        bounds=bounds,
        confidence_level=policy.confidence_level,
    )
    minimum = (
        policy.minimum_rqmc_randomizations
        if unit_kind == "rqmc_randomization"
        else policy.minimum_iid_units
    )
    plugin_required = max(
        minimum,
        math.ceil(certificate.empirical_sample_variance / target_estimator_variance),
    )
    certified_required = max(
        minimum,
        math.ceil(certificate.variance_upper / target_estimator_variance),
    )
    return TailSafeAllocationPlan(
        unit_kind=unit_kind,
        target_estimator_variance=target_estimator_variance,
        pilot_certificate=certificate,
        plugin_required_units=plugin_required,
        certified_required_units=certified_required,
        planned_units=min(certified_required, policy.maximum_units),
        resource_censored=certified_required > policy.maximum_units,
        policy=policy,
    )


@dataclass(frozen=True)
class TailSafeFinalCertificate:
    allocation: TailSafeAllocationPlan
    final_certificate: BoundedVarianceCertificate
    estimator_variance_upper: float
    target_attained: bool


def certify_tail_safe_final(
    plan: TailSafeAllocationPlan,
    final: StreamingMoments,
    *,
    bounds: BoundedRange,
) -> TailSafeFinalCertificate:
    """Independently certify final variance without trusting zero sample variance."""

    if final.count != plan.planned_units:
        raise ValueError("final unit count differs from the frozen plan")
    certificate = bounded_variance_certificate(
        final,
        bounds=bounds,
        confidence_level=plan.policy.confidence_level,
    )
    upper = certificate.variance_upper / final.count
    return TailSafeFinalCertificate(
        allocation=plan,
        final_certificate=certificate,
        estimator_variance_upper=upper,
        target_attained=upper <= plan.target_estimator_variance,
    )


def certify_empirical_bernstein_tail_safe_final(
    plan: TailSafeAllocationPlan,
    final: StreamingMoments,
    *,
    bounds: BoundedRange,
) -> TailSafeFinalCertificate:
    """Independently apply the V13 simultaneous certificate to final IID units."""

    if final.count != plan.planned_units:
        raise ValueError("final unit count differs from the frozen plan")
    certificate = empirical_bernstein_variance_certificate(
        final,
        bounds=bounds,
        confidence_level=plan.policy.confidence_level,
    )
    upper = certificate.variance_upper / final.count
    return TailSafeFinalCertificate(
        allocation=plan,
        final_certificate=certificate,
        estimator_variance_upper=upper,
        target_attained=upper <= plan.target_estimator_variance,
    )


def collect_streaming_moments(
    *,
    total_units: int,
    chunk_units: int,
    evaluator: Callable[[int, int], Sequence[float] | torch.Tensor],
) -> StreamingMoments:
    """Evaluate exact requested chunks and retain only mergeable sufficient statistics.

    ``evaluator(offset, count)`` owns RNG allocation.  The offset is supplied so the
    caller can derive a unique stream per chunk instead of reusing a seed.
    """

    if isinstance(total_units, bool) or not isinstance(total_units, int) or total_units < 2:
        raise ValueError("total units must be an integer of at least two")
    if isinstance(chunk_units, bool) or not isinstance(chunk_units, int) or chunk_units < 1:
        raise ValueError("chunk units must be a positive integer")
    result = StreamingMoments()
    offset = 0
    while offset < total_units:
        requested = min(chunk_units, total_units - offset)
        values = torch.as_tensor(
            evaluator(offset, requested), dtype=torch.float64, device="cpu"
        ).reshape(-1)
        if values.numel() != requested:
            raise ValueError("streaming evaluator returned the wrong unit count")
        result.update(values)
        offset += requested
    if result.count != total_units:
        raise AssertionError("streaming collection lost inferential units")
    return result
