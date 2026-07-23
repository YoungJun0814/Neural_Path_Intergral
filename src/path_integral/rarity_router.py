"""Pilot-measurable routing for moderate and rare finite-grid events.

The router never accepts a reference/oracle probability.  It uses an exact binomial
interval from a natural-law screening pilot and optional work intervals whose
construction belongs to the pre-final pilot sigma-field.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Literal

from src.path_integral.statistical_gates import (
    BinomialProbabilityInterval,
    exact_binomial_probability_interval,
)

RarityClass = Literal["moderate", "rare", "ambiguous"]
RoutingAction = Literal["crude", "dcs_slis", "profile_hybrid", "continue_screening"]


def _positive_real(value: float, field: str, *, allow_zero: bool = False) -> None:
    if not math.isfinite(value) or (value < 0.0 if allow_zero else value <= 0.0):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{field} must be finite and {qualifier}")


@dataclass(frozen=True)
class RoutingWorkInterval:
    """Pilot-frozen total-work interval for one immediately executable method."""

    method: Literal["crude", "dcs_slis"]
    lower: float
    point: float
    upper: float

    def __post_init__(self) -> None:
        if self.method not in ("crude", "dcs_slis"):
            raise ValueError("unsupported router work method")
        _positive_real(self.lower, "work lower", allow_zero=True)
        _positive_real(self.point, "work point", allow_zero=True)
        _positive_real(self.upper, "work upper", allow_zero=True)
        if not self.lower <= self.point <= self.upper:
            raise ValueError("work interval must satisfy lower <= point <= upper")


@dataclass(frozen=True)
class HybridProfileOpportunity:
    """Pre-profile lower-bound evidence used only to decide whether profiling can pay."""

    minimum_profile_work: float
    optimistic_total_work: float
    external_profile_work_cap: float

    def __post_init__(self) -> None:
        _positive_real(self.minimum_profile_work, "minimum profile work")
        _positive_real(self.optimistic_total_work, "optimistic hybrid total work")
        _positive_real(self.external_profile_work_cap, "external profile work cap")


@dataclass(frozen=True)
class RarityRouterConfig:
    """Predeclared thresholds; changing them creates a new protocol identity."""

    probability_cutoff: float = 0.05
    confidence_level: float = 0.99
    initial_screening_trials: int = 256
    maximum_screening_trials: int = 1024
    minimum_certified_relative_saving: float = 0.10
    maximum_hybrid_profile_work: float = 1e7
    maximum_profile_fraction: float = 0.25
    ambiguous_fallback: Literal["crude", "dcs_slis"] = "dcs_slis"

    def __post_init__(self) -> None:
        if not math.isfinite(self.probability_cutoff) or not 0.0 < self.probability_cutoff < 0.5:
            raise ValueError("probability_cutoff must lie in (0, 0.5)")
        if not math.isfinite(self.confidence_level) or not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must lie in (0, 1)")
        if isinstance(self.initial_screening_trials, bool) or self.initial_screening_trials < 1:
            raise ValueError("initial_screening_trials must be positive")
        if (
            isinstance(self.maximum_screening_trials, bool)
            or self.maximum_screening_trials < self.initial_screening_trials
        ):
            raise ValueError("maximum_screening_trials must cover the initial look")
        if not math.isfinite(self.minimum_certified_relative_saving) or not (
            0.0 <= self.minimum_certified_relative_saving < 1.0
        ):
            raise ValueError("minimum_certified_relative_saving must lie in [0, 1)")
        _positive_real(self.maximum_hybrid_profile_work, "maximum hybrid profile work")
        if not math.isfinite(self.maximum_profile_fraction) or not (
            0.0 < self.maximum_profile_fraction < 1.0
        ):
            raise ValueError("maximum_profile_fraction must lie in (0, 1)")
        if self.ambiguous_fallback not in ("crude", "dcs_slis"):
            raise ValueError("unsupported ambiguous fallback")


@dataclass(frozen=True)
class FrozenRarityRoute:
    """Auditable route that contains no final seed or reference value."""

    action: RoutingAction
    rarity_class: RarityClass
    reason: str
    probability_interval: BinomialProbabilityInterval
    screening_work: float
    effective_profile_work_cap: float | None
    current_best_method: Literal["crude", "dcs_slis"] | None
    current_best_point_work: float | None
    decision_hash: str


def _certified_better(
    challenger: RoutingWorkInterval,
    incumbent: RoutingWorkInterval,
    relative_saving: float,
) -> bool:
    return challenger.upper <= (1.0 - relative_saving) * incumbent.lower


def _current_best(
    crude: RoutingWorkInterval | None,
    dcs: RoutingWorkInterval | None,
    *,
    default: Literal["crude", "dcs_slis"],
) -> tuple[Literal["crude", "dcs_slis"] | None, float | None]:
    available = tuple(item for item in (crude, dcs) if item is not None)
    if not available:
        return None, None
    selected = min(
        available,
        key=lambda item: (item.point, 0 if item.method == default else 1, item.method),
    )
    return selected.method, selected.point


def _decision_hash(payload: dict[str, object]) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def freeze_rarity_route(
    *,
    successes: int,
    trials: int,
    screening_work: float,
    config: RarityRouterConfig,
    crude_work: RoutingWorkInterval | None = None,
    dcs_work: RoutingWorkInterval | None = None,
    hybrid_opportunity: HybridProfileOpportunity | None = None,
) -> FrozenRarityRoute:
    """Freeze one route from pre-final screening and work evidence.

    Work intervals exclude the already sunk screening work because it is common to
    every route.  ``screening_work`` remains in the artifact and must be charged by
    the outer policy ledger.
    """

    _positive_real(screening_work, "screening_work", allow_zero=True)
    for expected, work_interval in (("crude", crude_work), ("dcs_slis", dcs_work)):
        if work_interval is not None and work_interval.method != expected:
            raise ValueError(f"{expected}_work has the wrong method identity")
    probability_interval = exact_binomial_probability_interval(
        successes,
        trials,
        confidence_level=config.confidence_level,
    )
    if probability_interval.lower > config.probability_cutoff:
        rarity_class: RarityClass = "moderate"
    elif probability_interval.upper < config.probability_cutoff:
        rarity_class = "rare"
    else:
        rarity_class = "ambiguous"

    best_method, best_point = _current_best(
        crude_work,
        dcs_work,
        default="crude" if rarity_class == "moderate" else "dcs_slis",
    )
    action: RoutingAction
    reason: str
    effective_cap: float | None = None

    if rarity_class == "ambiguous" and trials < config.maximum_screening_trials:
        action = "continue_screening"
        reason = "probability interval straddles cutoff before the screening cap"
    elif rarity_class == "ambiguous":
        action = config.ambiguous_fallback
        reason = "probability interval remains ambiguous at the screening cap"
    elif rarity_class == "moderate":
        if (
            crude_work is not None
            and dcs_work is not None
            and _certified_better(
                dcs_work,
                crude_work,
                config.minimum_certified_relative_saving,
            )
        ):
            action = "dcs_slis"
            reason = "DCS-SLIS is certified cheaper despite a moderate event"
        else:
            action = "crude"
            reason = "moderate event does not justify Hybrid profiling"
    else:
        fallback: Literal["crude", "dcs_slis"] = best_method or "dcs_slis"
        if hybrid_opportunity is None or best_point is None:
            action = fallback
            reason = "rare event lacks an economically admissible Hybrid profile"
        else:
            effective_cap = min(
                config.maximum_hybrid_profile_work,
                hybrid_opportunity.external_profile_work_cap,
                config.maximum_profile_fraction * best_point,
            )
            potential_relative_saving = (
                best_point - hybrid_opportunity.optimistic_total_work
            ) / best_point
            if hybrid_opportunity.minimum_profile_work > effective_cap:
                action = fallback
                reason = "minimum Hybrid profiling work exceeds the frozen cap"
            elif potential_relative_saving < config.minimum_certified_relative_saving:
                action = fallback
                reason = "optimistic Hybrid saving cannot repay the required margin"
            else:
                action = "profile_hybrid"
                reason = "rare event has an economically admissible Hybrid profile"

    payload: dict[str, object] = {
        "schema": "npi.g11.v6-rarity-route.v1",
        "action": action,
        "rarity_class": rarity_class,
        "reason": reason,
        "probability_interval": asdict(probability_interval),
        "screening_work": screening_work,
        "effective_profile_work_cap": effective_cap,
        "current_best_method": best_method,
        "current_best_point_work": best_point,
        "config": asdict(config),
        "crude_work": None if crude_work is None else asdict(crude_work),
        "dcs_work": None if dcs_work is None else asdict(dcs_work),
        "hybrid_opportunity": (
            None if hybrid_opportunity is None else asdict(hybrid_opportunity)
        ),
    }
    return FrozenRarityRoute(
        action=action,
        rarity_class=rarity_class,
        reason=reason,
        probability_interval=probability_interval,
        screening_work=screening_work,
        effective_profile_work_cap=effective_cap,
        current_best_method=best_method,
        current_best_point_work=best_point,
        decision_hash=_decision_hash(payload),
    )
