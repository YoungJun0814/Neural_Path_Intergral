"""Identical evidence rule for candidates and comparators."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

QualificationStatus = Literal["qualified", "unresolved"]


@dataclass(frozen=True)
class QualificationPolicy:
    maximum_relative_se: float = 0.25
    maximum_reference_relative_se: float = 0.10
    relative_equivalence_margin: float = 0.25
    confidence_z: float = 1.96

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class Qualification:
    status: QualificationStatus
    reason: str
    accuracy_z_diagnostic: float
    relative_standard_error: float
    reference_relative_standard_error: float
    equivalence_upper_difference: float

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)


def qualify_estimate(
    *,
    estimate: float,
    standard_error: float,
    reference_estimate: float,
    reference_standard_error: float,
    reference_independent: bool,
    policy: QualificationPolicy,
) -> Qualification:
    """Fail closed on imprecision; a small z alone is never equivalence."""

    values = (estimate, standard_error, reference_estimate, reference_standard_error)
    if any(not math.isfinite(x) for x in values):
        raise ValueError("qualification inputs must be finite")
    if estimate < 0 or standard_error < 0 or reference_estimate <= 0 or reference_standard_error < 0:
        raise ValueError("invalid probability estimate or standard error")
    if not isinstance(reference_independent, bool):
        raise ValueError("reference_independent must be bool")
    rse = standard_error / estimate if estimate > 0 else math.inf
    ref_rse = reference_standard_error / reference_estimate
    difference = abs(estimate - reference_estimate)
    combined_se = math.hypot(standard_error, reference_standard_error)
    z = difference / combined_se if combined_se > 0 else (0.0 if difference == 0 else math.inf)
    upper = difference + policy.confidence_z * combined_se
    reason = "equivalence interval inside prespecified margin"
    if not reference_independent:
        reason = "reference independence not established"
    elif ref_rse > policy.maximum_reference_relative_se:
        reason = "reference too imprecise"
    elif rse > policy.maximum_relative_se:
        reason = "estimate too imprecise"
    elif upper > policy.relative_equivalence_margin * reference_estimate:
        reason = "equivalence interval exceeds margin"
    status: QualificationStatus = "qualified" if reason.startswith("equivalence") else "unresolved"
    return Qualification(status, reason, z, rse, ref_rse, upper)


def work_normalized_variance(
    sample_variance: float, total_work: float, inferential_units: int
) -> float:
    if (
        not math.isfinite(sample_variance)
        or sample_variance < 0
        or not math.isfinite(total_work)
        or total_work <= 0
        or isinstance(inferential_units, bool)
        or not isinstance(inferential_units, int)
        or inferential_units < 2
    ):
        raise ValueError("invalid variance, work, or independent unit count")
    return sample_variance * total_work / inferential_units
