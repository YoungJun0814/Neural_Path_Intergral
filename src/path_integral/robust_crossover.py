"""Finite-look, bounded-observation crossover selection.

The confidence allocation is deliberately explicit: each profile contributes two
Hoeffding intervals (mean and second moment) at every predeclared look.  A union
bound over profiles, moments, and looks gives familywise coverage without treating
a fixed-sample interval as an anytime-valid confidence sequence.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BoundedMomentInterval:
    """Auditable mean, second-moment, and variance intervals for ``|X| <= M``."""

    sample_count: int
    absolute_bound: float
    alpha_per_moment: float
    sample_mean: float
    sample_second_moment: float
    sample_variance: float
    mean_interval: tuple[float, float]
    second_moment_interval: tuple[float, float]
    variance_interval: tuple[float, float]


@dataclass(frozen=True)
class LevelProfileInterval:
    """One bounded level term and its deterministic per-sample operation cost."""

    profile_id: str
    cost_per_sample: float
    moments: BoundedMomentInterval


@dataclass(frozen=True)
class CandidateWorkInterval:
    """Sampling and preprocessing work interval for one candidate construction."""

    candidate_id: str
    profile_ids: tuple[str, ...]
    sampling_variance_target: float
    sampling_work_coefficient: tuple[float, float, float]
    preprocessing_work: float
    total_work_interval: tuple[float, float]
    point_total_work: float


@dataclass(frozen=True)
class CandidateElimination:
    """Serialized reason for dropping one candidate at one finite look."""

    candidate_id: str
    look_index: int
    candidate_lower_work: float
    best_upper_work: float
    threshold_work: float
    reason: str


@dataclass(frozen=True)
class CrossoverEliminationResult:
    """Survivors and new eliminations from one simultaneous-interval comparison."""

    surviving_candidates: tuple[str, ...]
    eliminated: tuple[CandidateElimination, ...]
    best_upper_work: float


@dataclass(frozen=True)
class FrozenCrossoverDecision:
    """A pilot-measurable decision that final sampling must not revise."""

    selected_candidate: str
    look_index: int
    reason: str
    surviving_candidates: tuple[str, ...]
    selected_work_interval: tuple[float, float]
    selected_point_work: float
    worst_case_interval_regret_bound: float


@dataclass(frozen=True)
class SequentialCrossoverState:
    """Immutable snapshot of a predeclared finite-look selection run."""

    predeclared_looks: tuple[int, ...]
    look_index: int
    cumulative_sample_count: int
    familywise_alpha: float
    profiles: tuple[LevelProfileInterval, ...]
    candidate_work: tuple[CandidateWorkInterval, ...]
    surviving_candidates: tuple[str, ...]
    elimination_history: tuple[CandidateElimination, ...]
    maximum_profile_work: float | None
    maximum_profile_fraction_of_best_point: float | None
    cumulative_profile_work: float
    effective_profile_work_cap: float | None
    stopped: bool
    stop_reason: str | None
    frozen_decision: FrozenCrossoverDecision | None


def _as_finite_sample(values: torch.Tensor | Sequence[float]) -> torch.Tensor:
    sample = torch.as_tensor(values, dtype=torch.float64, device="cpu").reshape(-1)
    if sample.numel() < 2:
        raise ValueError("at least two observations are required for a variance profile")
    if not torch.isfinite(sample).all():
        raise ValueError("profile observations must be finite")
    return sample


def _bounded_moment_interval(
    values: torch.Tensor | Sequence[float],
    *,
    absolute_bound: float,
    alpha_per_moment: float,
    bound_tolerance: float,
) -> BoundedMomentInterval:
    sample = _as_finite_sample(values)
    if not math.isfinite(absolute_bound) or absolute_bound <= 0.0:
        raise ValueError("absolute_bound must be finite and positive")
    if not 0.0 < alpha_per_moment < 1.0:
        raise ValueError("alpha_per_moment must lie in (0, 1)")
    if not math.isfinite(bound_tolerance) or bound_tolerance < 0.0:
        raise ValueError("bound_tolerance must be finite and nonnegative")
    maximum = float(torch.amax(torch.abs(sample)))
    if maximum > absolute_bound + bound_tolerance * max(1.0, absolute_bound):
        raise ValueError("an observation exceeds its declared deterministic bound")

    count = int(sample.numel())
    mean = float(torch.mean(sample))
    second = float(torch.mean(sample.square()))
    variance = max(0.0, second - mean**2)
    mean_radius = absolute_bound * math.sqrt(2.0 * math.log(2.0 / alpha_per_moment) / count)
    second_radius = absolute_bound**2 * math.sqrt(math.log(2.0 / alpha_per_moment) / (2.0 * count))
    mean_interval = (
        max(-absolute_bound, mean - mean_radius),
        min(absolute_bound, mean + mean_radius),
    )
    second_interval = (
        max(0.0, second - second_radius),
        min(absolute_bound**2, second + second_radius),
    )
    maximum_abs_mean = max(abs(mean_interval[0]), abs(mean_interval[1]))
    if mean_interval[0] <= 0.0 <= mean_interval[1]:
        minimum_abs_mean = 0.0
    else:
        minimum_abs_mean = min(abs(mean_interval[0]), abs(mean_interval[1]))
    variance_lower = max(0.0, second_interval[0] - maximum_abs_mean**2)
    variance_upper = max(
        0.0,
        min(
            absolute_bound**2,
            second_interval[1] - minimum_abs_mean**2,
        ),
    )
    # An empty projected interval can occur when separately valid moment intervals
    # are combined outside their joint feasible set.  The full variance range is a
    # conservative fallback; silently swapping endpoints would not be valid.
    if variance_lower > variance_upper:
        variance_lower, variance_upper = 0.0, absolute_bound**2
    variance = min(max(variance, 0.0), absolute_bound**2)
    return BoundedMomentInterval(
        sample_count=count,
        absolute_bound=absolute_bound,
        alpha_per_moment=alpha_per_moment,
        sample_mean=mean,
        sample_second_moment=second,
        sample_variance=variance,
        mean_interval=mean_interval,
        second_moment_interval=second_interval,
        variance_interval=(variance_lower, variance_upper),
    )


def update_profile_intervals(
    observations_by_profile: Mapping[str, torch.Tensor | Sequence[float]],
    *,
    absolute_bounds: Mapping[str, float],
    costs_per_sample: Mapping[str, float],
    familywise_alpha: float,
    total_predeclared_looks: int,
    bound_tolerance: float = 1e-12,
) -> tuple[LevelProfileInterval, ...]:
    """Build simultaneous profile intervals for one predeclared cumulative look."""

    if not observations_by_profile:
        raise ValueError("at least one profile is required")
    if not 0.0 < familywise_alpha < 1.0:
        raise ValueError("familywise_alpha must lie in (0, 1)")
    if total_predeclared_looks < 1:
        raise ValueError("total_predeclared_looks must be positive")
    profile_ids = tuple(sorted(observations_by_profile))
    if set(absolute_bounds) != set(profile_ids):
        raise ValueError("absolute bounds do not match the profile identifiers")
    if set(costs_per_sample) != set(profile_ids):
        raise ValueError("costs do not match the profile identifiers")
    alpha_per_moment = familywise_alpha / (2.0 * len(profile_ids) * total_predeclared_looks)
    profiles: list[LevelProfileInterval] = []
    for profile_id in profile_ids:
        cost = float(costs_per_sample[profile_id])
        if not math.isfinite(cost) or cost <= 0.0:
            raise ValueError("per-sample costs must be finite and positive")
        profiles.append(
            LevelProfileInterval(
                profile_id=profile_id,
                cost_per_sample=cost,
                moments=_bounded_moment_interval(
                    observations_by_profile[profile_id],
                    absolute_bound=float(absolute_bounds[profile_id]),
                    alpha_per_moment=alpha_per_moment,
                    bound_tolerance=bound_tolerance,
                ),
            )
        )
    return tuple(profiles)


def _work_coefficient(variances: Sequence[float], costs: Sequence[float]) -> float:
    if len(variances) != len(costs) or not variances:
        raise ValueError("a candidate requires aligned nonempty variance and cost terms")
    if any(not math.isfinite(value) or value < 0.0 for value in variances):
        raise ValueError("candidate variances must be finite and nonnegative")
    if any(not math.isfinite(value) or value <= 0.0 for value in costs):
        raise ValueError("candidate costs must be finite and positive")
    return (
        math.fsum(
            math.sqrt(variance * cost) for variance, cost in zip(variances, costs, strict=True)
        )
        ** 2
    )


def candidate_work_intervals(
    profiles: Sequence[LevelProfileInterval],
    *,
    candidate_profiles: Mapping[str, Sequence[str]],
    preprocessing_work: Mapping[str, float],
    sampling_variance_target: float,
) -> tuple[CandidateWorkInterval, ...]:
    """Propagate simultaneous variance intervals through monotone MLMC work."""

    if not math.isfinite(sampling_variance_target) or sampling_variance_target <= 0.0:
        raise ValueError("sampling_variance_target must be finite and positive")
    profile_map = {profile.profile_id: profile for profile in profiles}
    if len(profile_map) != len(tuple(profiles)):
        raise ValueError("profile identifiers must be unique")
    if not candidate_profiles:
        raise ValueError("at least one candidate is required")
    if set(preprocessing_work) != set(candidate_profiles):
        raise ValueError("preprocessing work does not match candidate identifiers")

    result: list[CandidateWorkInterval] = []
    for candidate_id in sorted(candidate_profiles):
        term_ids = tuple(candidate_profiles[candidate_id])
        if not term_ids or len(set(term_ids)) != len(term_ids):
            raise ValueError("candidate profile lists must be nonempty and unique")
        if any(term_id not in profile_map for term_id in term_ids):
            raise ValueError("candidate references an unknown profile")
        fixed = float(preprocessing_work[candidate_id])
        if not math.isfinite(fixed) or fixed < 0.0:
            raise ValueError("preprocessing work must be finite and nonnegative")
        terms = tuple(profile_map[term_id] for term_id in term_ids)
        costs = tuple(term.cost_per_sample for term in terms)
        lower = _work_coefficient(tuple(term.moments.variance_interval[0] for term in terms), costs)
        point = _work_coefficient(tuple(term.moments.sample_variance for term in terms), costs)
        upper = _work_coefficient(tuple(term.moments.variance_interval[1] for term in terms), costs)
        total_lower = fixed + lower / sampling_variance_target
        total_point = fixed + point / sampling_variance_target
        total_upper = fixed + upper / sampling_variance_target
        result.append(
            CandidateWorkInterval(
                candidate_id=candidate_id,
                profile_ids=term_ids,
                sampling_variance_target=sampling_variance_target,
                sampling_work_coefficient=(lower, point, upper),
                preprocessing_work=fixed,
                total_work_interval=(total_lower, total_upper),
                point_total_work=total_point,
            )
        )
    return tuple(result)


def eliminate_dominated_candidates(
    candidates: Sequence[CandidateWorkInterval],
    *,
    look_index: int,
    surviving_candidates: Sequence[str] | None = None,
    relative_tolerance: float = 0.0,
    absolute_tolerance: float = 0.0,
) -> CrossoverEliminationResult:
    """Eliminate only when a lower work bound exceeds the best upper bound."""

    if look_index < 0:
        raise ValueError("look_index must be nonnegative")
    if not math.isfinite(relative_tolerance) or relative_tolerance < 0.0:
        raise ValueError("relative_tolerance must be finite and nonnegative")
    if not math.isfinite(absolute_tolerance) or absolute_tolerance < 0.0:
        raise ValueError("absolute_tolerance must be finite and nonnegative")
    candidate_map = {candidate.candidate_id: candidate for candidate in candidates}
    if len(candidate_map) != len(tuple(candidates)) or not candidate_map:
        raise ValueError("candidate identifiers must be nonempty and unique")
    survivor_ids = (
        tuple(sorted(candidate_map))
        if surviving_candidates is None
        else tuple(surviving_candidates)
    )
    if not survivor_ids or len(set(survivor_ids)) != len(survivor_ids):
        raise ValueError("surviving candidate identifiers must be nonempty and unique")
    if any(candidate_id not in candidate_map for candidate_id in survivor_ids):
        raise ValueError("survivors contain an unknown candidate")
    best_upper = min(candidate_map[item].total_work_interval[1] for item in survivor_ids)
    threshold = best_upper * (1.0 + relative_tolerance) + absolute_tolerance
    kept: list[str] = []
    eliminated: list[CandidateElimination] = []
    for candidate_id in survivor_ids:
        lower = candidate_map[candidate_id].total_work_interval[0]
        if lower > threshold:
            eliminated.append(
                CandidateElimination(
                    candidate_id=candidate_id,
                    look_index=look_index,
                    candidate_lower_work=lower,
                    best_upper_work=best_upper,
                    threshold_work=threshold,
                    reason="lower work bound exceeds the tolerated best upper bound",
                )
            )
        else:
            kept.append(candidate_id)
    if not kept:
        raise AssertionError("simultaneous-interval elimination removed every candidate")
    return CrossoverEliminationResult(tuple(kept), tuple(eliminated), best_upper)


def freeze_crossover_decision(
    candidates: Sequence[CandidateWorkInterval],
    *,
    look_index: int,
    surviving_candidates: Sequence[str],
    simpler_candidate: str,
    reason: str,
    upper_bound_tie_relative_tolerance: float = 0.0,
) -> FrozenCrossoverDecision:
    """Choose the smallest upper bound, with a frozen simplicity tie-break."""

    if not reason:
        raise ValueError("a freeze reason is required")
    if not math.isfinite(upper_bound_tie_relative_tolerance) or not (
        0.0 <= upper_bound_tie_relative_tolerance < 1.0
    ):
        raise ValueError("tie tolerance must lie in [0, 1)")
    candidate_map = {candidate.candidate_id: candidate for candidate in candidates}
    survivor_ids = tuple(surviving_candidates)
    if not survivor_ids or any(item not in candidate_map for item in survivor_ids):
        raise ValueError("surviving candidates must be nonempty and known")
    if simpler_candidate not in survivor_ids:
        raise ValueError("simpler_candidate must be a survivor")
    best_upper = min(candidate_map[item].total_work_interval[1] for item in survivor_ids)
    tied = tuple(
        item
        for item in survivor_ids
        if candidate_map[item].total_work_interval[1]
        <= best_upper * (1.0 + upper_bound_tie_relative_tolerance)
    )
    selected = (
        simpler_candidate
        if simpler_candidate in tied
        else min(tied, key=lambda item: (candidate_map[item].total_work_interval[1], item))
    )
    selected_candidate = candidate_map[selected]
    positive_lowers = [
        candidate_map[item].total_work_interval[0]
        for item in survivor_ids
        if candidate_map[item].total_work_interval[0] > 0.0
    ]
    regret_bound = (
        selected_candidate.total_work_interval[1] / min(positive_lowers)
        if positive_lowers
        else math.inf
    )
    return FrozenCrossoverDecision(
        selected_candidate=selected,
        look_index=look_index,
        reason=reason,
        surviving_candidates=survivor_ids,
        selected_work_interval=selected_candidate.total_work_interval,
        selected_point_work=selected_candidate.point_total_work,
        worst_case_interval_regret_bound=regret_bound,
    )


def advance_sequential_crossover(
    observations_by_profile: Mapping[str, torch.Tensor | Sequence[float]],
    *,
    absolute_bounds: Mapping[str, float],
    costs_per_sample: Mapping[str, float],
    candidate_profiles: Mapping[str, Sequence[str]],
    preprocessing_work: Mapping[str, float],
    sampling_variance_target: float,
    predeclared_looks: Sequence[int],
    look_index: int,
    familywise_alpha: float,
    simpler_candidate: str,
    previous_state: SequentialCrossoverState | None = None,
    elimination_relative_tolerance: float = 0.0,
    practical_equivalence_relative_tolerance: float = 0.0,
    maximum_profile_work: float | None = None,
    maximum_profile_fraction_of_best_point: float | None = None,
) -> SequentialCrossoverState:
    """Advance exactly one cumulative look and freeze only by declared rules.

    Every profile must contain exactly the cumulative sample count named by the
    current look.  This prevents a caller from silently inserting an unregistered
    optional look while still using the finite-look Bonferroni allocation.
    """

    looks = tuple(int(value) for value in predeclared_looks)
    if (
        not looks
        or any(value < 2 for value in looks)
        or any(right <= left for left, right in zip(looks, looks[1:], strict=False))
    ):
        raise ValueError("predeclared looks must be strictly increasing integers >= 2")
    if not 0 <= look_index < len(looks):
        raise ValueError("look_index is outside the predeclared schedule")
    counts = {int(torch.as_tensor(values).numel()) for values in observations_by_profile.values()}
    if counts != {looks[look_index]}:
        raise ValueError("every cumulative profile must match the current declared look")
    if not math.isfinite(practical_equivalence_relative_tolerance) or not (
        0.0 <= practical_equivalence_relative_tolerance < 1.0
    ):
        raise ValueError("practical-equivalence tolerance must lie in [0, 1)")
    if maximum_profile_work is not None and (
        not math.isfinite(maximum_profile_work) or maximum_profile_work <= 0.0
    ):
        raise ValueError("maximum profile work must be finite and positive")
    if maximum_profile_fraction_of_best_point is not None and (
        not math.isfinite(maximum_profile_fraction_of_best_point)
        or not 0.0 < maximum_profile_fraction_of_best_point < 1.0
    ):
        raise ValueError("maximum profile fraction must lie in (0, 1)")
    if previous_state is not None:
        if previous_state.stopped:
            raise ValueError("a frozen crossover state cannot be advanced")
        if previous_state.predeclared_looks != looks:
            raise ValueError("the predeclared look schedule changed")
        if previous_state.look_index + 1 != look_index:
            raise ValueError("sequential looks must be advanced without gaps or repeats")
        if previous_state.familywise_alpha != familywise_alpha:
            raise ValueError("familywise alpha changed between looks")
        if previous_state.maximum_profile_work != maximum_profile_work:
            raise ValueError("maximum profile work changed between looks")
        if (
            previous_state.maximum_profile_fraction_of_best_point
            != maximum_profile_fraction_of_best_point
        ):
            raise ValueError("maximum profile fraction changed between looks")
        prior_survivors = previous_state.surviving_candidates
        history = previous_state.elimination_history
    else:
        if look_index != 0:
            raise ValueError("the first sequential state must use look_index zero")
        prior_survivors = tuple(sorted(candidate_profiles))
        history = ()

    profiles = update_profile_intervals(
        observations_by_profile,
        absolute_bounds=absolute_bounds,
        costs_per_sample=costs_per_sample,
        familywise_alpha=familywise_alpha,
        total_predeclared_looks=len(looks),
    )
    work = candidate_work_intervals(
        profiles,
        candidate_profiles=candidate_profiles,
        preprocessing_work=preprocessing_work,
        sampling_variance_target=sampling_variance_target,
    )
    elimination = eliminate_dominated_candidates(
        work,
        look_index=look_index,
        surviving_candidates=prior_survivors,
        relative_tolerance=elimination_relative_tolerance,
    )
    survivors = elimination.surviving_candidates
    work_map = {candidate.candidate_id: candidate for candidate in work}
    if simpler_candidate not in candidate_profiles:
        raise ValueError("simpler_candidate is not a declared candidate")

    cumulative_profile_work = math.fsum(
        profile.moments.sample_count * profile.cost_per_sample for profile in profiles
    )
    if maximum_profile_work is not None and cumulative_profile_work > maximum_profile_work * (
        1.0 + 1e-12
    ):
        raise ValueError("current declared look exceeds the absolute profile-work budget")
    best_point_work = min(work_map[item].point_total_work for item in survivors)
    caps = []
    if maximum_profile_work is not None:
        caps.append(maximum_profile_work)
    if maximum_profile_fraction_of_best_point is not None:
        caps.append(maximum_profile_fraction_of_best_point * best_point_work)
    effective_profile_work_cap = min(caps) if caps else None

    stop_reason: str | None = None
    if len(survivors) == 1:
        stop_reason = "one candidate remains"
    else:
        minimum_lower = min(work_map[item].total_work_interval[0] for item in survivors)
        maximum_upper = max(work_map[item].total_work_interval[1] for item in survivors)
        if (
            minimum_lower > 0.0
            and maximum_upper <= (1.0 + practical_equivalence_relative_tolerance) * minimum_lower
        ):
            stop_reason = "all surviving candidates are practically equivalent"
        elif (
            effective_profile_work_cap is not None
            and cumulative_profile_work >= effective_profile_work_cap * (1.0 - 1e-12)
        ):
            stop_reason = "frozen profiling work budget reached"
        elif effective_profile_work_cap is not None and look_index < len(looks) - 1:
            next_profile_work = math.fsum(
                looks[look_index + 1] * profile.cost_per_sample for profile in profiles
            )
            if next_profile_work > effective_profile_work_cap * (1.0 + 1e-12):
                stop_reason = "next declared look exceeds the profiling work budget"
        elif look_index == len(looks) - 1:
            stop_reason = "predeclared pilot cap reached"

    decision = None
    if stop_reason is not None:
        tie_break = simpler_candidate
        if tie_break not in survivors:
            tie_break = min(
                survivors,
                key=lambda item: (work_map[item].total_work_interval[1], item),
            )
        decision = freeze_crossover_decision(
            work,
            look_index=look_index,
            surviving_candidates=survivors,
            simpler_candidate=tie_break,
            reason=stop_reason,
            upper_bound_tie_relative_tolerance=(practical_equivalence_relative_tolerance),
        )
    return SequentialCrossoverState(
        predeclared_looks=looks,
        look_index=look_index,
        cumulative_sample_count=looks[look_index],
        familywise_alpha=familywise_alpha,
        profiles=profiles,
        candidate_work=work,
        surviving_candidates=survivors,
        elimination_history=history + elimination.eliminated,
        maximum_profile_work=maximum_profile_work,
        maximum_profile_fraction_of_best_point=maximum_profile_fraction_of_best_point,
        cumulative_profile_work=cumulative_profile_work,
        effective_profile_work_cap=effective_profile_work_cap,
        stopped=decision is not None,
        stop_reason=stop_reason,
        frozen_decision=decision,
    )


def plug_in_relative_error_regret_bound(relative_error: float) -> float:
    """Return ``(1+gamma)/(1-gamma)`` for a finite candidate plug-in selector."""

    if not math.isfinite(relative_error) or not 0.0 <= relative_error < 1.0:
        raise ValueError("relative_error must lie in [0, 1)")
    return (1.0 + relative_error) / (1.0 - relative_error)
