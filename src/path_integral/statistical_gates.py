"""Predeclared statistical summaries and gates for G11 V5 artifacts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import scipy.stats
import torch


@dataclass(frozen=True)
class BinomialProbabilityInterval:
    successes: int
    trials: int
    confidence_level: float
    lower: float
    upper: float


@dataclass(frozen=True)
class PairedLogWorkSummary:
    cluster_count: int
    confidence_level: float
    mean_log_baseline_over_method: float
    standard_error: float
    confidence_interval: tuple[float, float]
    geometric_mean_baseline_over_method: float
    geometric_mean_confidence_interval: tuple[float, float]
    method_better_fraction: float


@dataclass(frozen=True)
class HeavyTailDiagnostics:
    count: int
    mean: float
    variance: float
    maximum_absolute_contribution: float
    maximum_absolute_to_sum_absolute: float
    positive_weight_ess: float | None
    excess_kurtosis: float | None


@dataclass(frozen=True)
class ReferenceAgreement:
    estimate_a: float
    standard_error_a: float
    estimate_b: float
    standard_error_b: float
    combined_z_score: float
    maximum_z_score: float
    agrees: bool


@dataclass(frozen=True)
class PairedPowerForecast:
    alpha: float
    power: float
    mean_log_effect: float
    standard_deviation: float
    required_clusters_normal_approximation: int


def exact_binomial_probability_interval(
    successes: int,
    trials: int,
    *,
    confidence_level: float = 0.95,
) -> BinomialProbabilityInterval:
    """Return the two-sided Clopper--Pearson interval, including zero hits."""

    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("successes and trials must satisfy 0 <= successes <= trials")
    if not math.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    alpha = 1.0 - confidence_level
    lower = (
        0.0
        if successes == 0
        else float(scipy.stats.beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    )
    upper = (
        1.0
        if successes == trials
        else float(scipy.stats.beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
    )
    return BinomialProbabilityInterval(successes, trials, confidence_level, lower, upper)


def conservative_bernoulli_variance_upper(
    interval: BinomialProbabilityInterval,
) -> float:
    """Maximize ``p(1-p)`` over an exact probability interval."""

    if interval.lower <= 0.5 <= interval.upper:
        return 0.25
    return max(
        interval.lower * (1.0 - interval.lower),
        interval.upper * (1.0 - interval.upper),
    )


def paired_log_work_summary(
    method_work: Sequence[float],
    baseline_work: Sequence[float],
    *,
    confidence_level: float = 0.95,
) -> PairedLogWorkSummary:
    """Summarize paired cluster work as log(baseline/method)."""

    method = torch.as_tensor(method_work, dtype=torch.float64).reshape(-1)
    baseline = torch.as_tensor(baseline_work, dtype=torch.float64).reshape(-1)
    if method.shape != baseline.shape or method.numel() < 2:
        raise ValueError("paired work arrays must have the same length of at least two")
    if (
        not torch.isfinite(method).all()
        or not torch.isfinite(baseline).all()
        or bool((method <= 0.0).any())
        or bool((baseline <= 0.0).any())
    ):
        raise ValueError("paired work must be finite and strictly positive")
    if not math.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    log_ratio = torch.log(baseline / method)
    count = int(log_ratio.numel())
    mean = float(torch.mean(log_ratio))
    standard_error = float(torch.std(log_ratio, unbiased=True)) / math.sqrt(count)
    critical = float(scipy.stats.t.ppf(0.5 + confidence_level / 2.0, df=count - 1))
    interval = (mean - critical * standard_error, mean + critical * standard_error)
    return PairedLogWorkSummary(
        cluster_count=count,
        confidence_level=confidence_level,
        mean_log_baseline_over_method=mean,
        standard_error=standard_error,
        confidence_interval=interval,
        geometric_mean_baseline_over_method=math.exp(mean),
        geometric_mean_confidence_interval=(math.exp(interval[0]), math.exp(interval[1])),
        method_better_fraction=float(torch.mean((log_ratio > 0.0).to(torch.float64))),
    )


def holm_rejections(
    p_values: Mapping[str, float], *, familywise_alpha: float = 0.05
) -> dict[str, bool]:
    """Return Holm step-down rejections with deterministic identifier tie-breaks."""

    if not p_values:
        raise ValueError("at least one p-value is required")
    if not math.isfinite(familywise_alpha) or not 0.0 < familywise_alpha < 1.0:
        raise ValueError("familywise_alpha must lie in (0, 1)")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in p_values.values()):
        raise ValueError("p-values must lie in [0, 1]")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    rejected = {name: False for name in p_values}
    for index, (name, value) in enumerate(ordered):
        threshold = familywise_alpha / (len(ordered) - index)
        if value > threshold:
            break
        rejected[name] = True
    return rejected


def heavy_tail_diagnostics(values: Sequence[float] | torch.Tensor) -> HeavyTailDiagnostics:
    """Report contribution concentration without pretending signed ESS is defined."""

    sample = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    if sample.numel() < 2 or not torch.isfinite(sample).all():
        raise ValueError("at least two finite contributions are required")
    absolute = torch.abs(sample)
    absolute_sum = float(torch.sum(absolute))
    centered = sample - torch.mean(sample)
    population_variance = float(torch.mean(centered.square()))
    fourth = float(torch.mean(centered.pow(4)))
    if bool((sample >= 0.0).all()):
        denominator = float(torch.sum(sample.square()))
        ess = float(torch.sum(sample)) ** 2 / denominator if denominator > 0.0 else None
    else:
        ess = None
    return HeavyTailDiagnostics(
        count=int(sample.numel()),
        mean=float(torch.mean(sample)),
        variance=float(torch.var(sample, unbiased=True)),
        maximum_absolute_contribution=float(torch.amax(absolute)),
        maximum_absolute_to_sum_absolute=(
            float(torch.amax(absolute)) / absolute_sum if absolute_sum > 0.0 else 0.0
        ),
        positive_weight_ess=ess,
        excess_kurtosis=(
            fourth / population_variance**2 - 3.0 if population_variance > 0.0 else None
        ),
    )


def reference_agreement(
    estimate_a: float,
    standard_error_a: float,
    estimate_b: float,
    standard_error_b: float,
    *,
    maximum_z_score: float = 4.0,
) -> ReferenceAgreement:
    values = (estimate_a, standard_error_a, estimate_b, standard_error_b, maximum_z_score)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("reference agreement inputs must be finite")
    if standard_error_a < 0.0 or standard_error_b < 0.0 or maximum_z_score <= 0.0:
        raise ValueError("standard errors must be nonnegative and z threshold positive")
    denominator = math.hypot(standard_error_a, standard_error_b)
    if denominator == 0.0:
        z_score = 0.0 if estimate_a == estimate_b else math.inf
    else:
        z_score = abs(estimate_a - estimate_b) / denominator
    return ReferenceAgreement(
        estimate_a,
        standard_error_a,
        estimate_b,
        standard_error_b,
        z_score,
        maximum_z_score,
        z_score <= maximum_z_score,
    )


def paired_power_forecast(
    *,
    mean_log_effect: float,
    standard_deviation: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> PairedPowerForecast:
    """Normal-approximation planning count for a two-sided paired log-work test."""

    if not math.isfinite(mean_log_effect) or mean_log_effect == 0.0:
        raise ValueError("mean_log_effect must be finite and nonzero")
    if not math.isfinite(standard_deviation) or standard_deviation <= 0.0:
        raise ValueError("standard_deviation must be finite and positive")
    if not 0.0 < alpha < 1.0 or not 0.0 < power < 1.0:
        raise ValueError("alpha and power must lie in (0, 1)")
    z_alpha = float(scipy.stats.norm.ppf(1.0 - alpha / 2.0))
    z_power = float(scipy.stats.norm.ppf(power))
    required = math.ceil(((z_alpha + z_power) * standard_deviation / abs(mean_log_effect)) ** 2)
    return PairedPowerForecast(alpha, power, mean_log_effect, standard_deviation, max(2, required))
