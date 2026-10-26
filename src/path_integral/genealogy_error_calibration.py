"""Known rare Gaussian-mixture targets for diagnostic calibration, not gate tuning."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from scipy.integrate import quad
from scipy.special import ndtri
from scipy.stats import spearmanr

from .conditional_risk_target_smc import RiskEvaluation, lineage_summary, normalizer_summary


@dataclass(frozen=True)
class KnownMixtureTarget:
    name: str
    means: tuple[float, float]
    weights: tuple[float, float]
    sd: float

    def __post_init__(self) -> None:
        if (not self.name or not 0 < self.sd < 1 or not math.isfinite(self.sd)
                or len(self.means) != 2 or len(self.weights) != 2
                or any(not math.isfinite(v) for v in self.means+self.weights)
                or any(v <= 0 for v in self.weights) or abs(sum(self.weights)-1) > 1e-14):
            raise ValueError("finite two-component target with sd in (0,1) required")

    @property
    def log_z(self) -> float:
        terms = [math.log(w)-math.log(self.sd)+m*m/(2*(1-self.sd**2))
                 for m, w in zip(self.means, self.weights, strict=True)]
        maximum = max(terms)
        return -maximum-math.log(math.fsum(math.exp(t-maximum) for t in terms))

    def truth(self) -> dict[str, float]:
        return {"log_normalizer": self.log_z, "normalizer": math.exp(self.log_z),
            "mean": math.fsum(m*w for m, w in zip(self.means, self.weights, strict=True)),
            "second_moment": math.fsum(w*(m*m+self.sd**2) for m, w in zip(self.means, self.weights, strict=True)),
            "right_mass": math.fsum(w*.5*math.erfc(-m/(self.sd*math.sqrt(2))) for m, w in zip(self.means, self.weights, strict=True))}

    def evaluate(self, x: torch.Tensor) -> RiskEvaluation:
        if (x.ndim != 2 or x.shape[1] != 1 or x.dtype != torch.float64
                or x.device.type != "cpu" or x.requires_grad or not torch.isfinite(x).all()):
            raise ValueError("detached finite CPU float64 X1 required")
        means = torch.tensor(self.means, dtype=torch.float64)
        weights = torch.tensor(self.weights, dtype=torch.float64)
        log_h = torch.logsumexp(weights.log()-.5*((x-means)/self.sd).square(), 1)-math.log(self.sd)-.5*math.log(2*math.pi)
        log_p = -.5*x[:, 0].square()-.5*math.log(2*math.pi)
        return RiskEvaluation(self.log_z+log_h-log_p, torch.zeros(len(x), dtype=torch.float64), x.clone(), 0.)

    def sample_exact(self, n: int, *, label_seed: int, noise_seed: int) -> torch.Tensor:
        if type(n) is not int or n < 1 or label_seed == noise_seed:
            raise ValueError("positive count and separate exact-label/noise streams required")
        weights, means = torch.tensor(self.weights, dtype=torch.float64), torch.tensor(self.means, dtype=torch.float64)
        labels = torch.multinomial(weights, n, replacement=True, generator=torch.Generator().manual_seed(label_seed))
        return (means[labels]+self.sd*torch.randn(n, dtype=torch.float64, generator=torch.Generator().manual_seed(noise_seed)))[:, None]


TARGETS = (KnownMixtureTarget("moderate-symmetric", (-1.5, 1.5), (.5, .5), .7),
    KnownMixtureTarget("rare-symmetric", (-4., 4.), (.5, .5), .5),
    KnownMixtureTarget("rare-asymmetric", (-5., 6.), (.7, .3), .5))


def independent_reference(target: KnownMixtureTarget) -> dict[str, Any]:
    evaluations = 0

    def log_m2_scalar(x: float) -> float:
        terms = [math.log(w)-math.log(target.sd)-.5*((x-m)/target.sd)**2
                 for m, w in zip(target.means, target.weights, strict=True)]
        maximum = max(terms)
        return target.log_z+maximum+math.log(math.fsum(math.exp(t-maximum) for t in terms))+.5*x*x

    def f(x: float, power: int) -> float:
        nonlocal evaluations
        evaluations += 1
        return math.exp(-.5*x*x-.5*math.log(2*math.pi)+log_m2_scalar(x)-target.log_z)*x**power

    raw, errors, right = [], [], 0.
    for power in range(3):
        halves = [quad(f, a, b, args=(power,), epsabs=1e-11, epsrel=1e-11, limit=200)
                  for a, b in ((-math.inf, 0.), (0., math.inf))]
        raw.append(math.fsum(h[0] for h in halves))
        errors.append(math.fsum(h[1] for h in halves))
        if power == 0:
            right = halves[1][0]
    truth = target.truth()
    disagreement = max(abs(a-b) for a, b in zip(raw+[right],
        [1., truth["mean"], truth["second_moment"], truth["right_mass"]], strict=True))
    return {"target": asdict(target), "truth": truth, "scaled_raw_moments": raw+[right],
        "maximum_reference_disagreement": disagreement, "adaptive_error_proxies": errors,
        "scalar_reference_evaluations": evaluations,
        "pass": disagreement <= 1e-9 and max(errors) <= 1e-9,
        "not_outward_rounded_or_arbitrary_dimensional_certificate": True}


def weighted_moments(x: torch.Tensor, weights: torch.Tensor) -> dict[str, float]:
    if (x.ndim != 2 or x.shape[1] != 1 or weights.shape != (len(x),)
            or not torch.isfinite(x).all() or not torch.isfinite(weights).all()
            or bool((weights < 0).any()) or abs(float(weights.sum())-1) > 1e-10):
        raise ValueError("finite coordinates and normalized diagnostic weights required")
    return {"mean": float(weights @ x[:, 0]), "second_moment": float(weights @ x[:, 0].square()),
        "right_mass": float(weights @ (x[:, 0] > 0).to(torch.float64))}


def truth_errors(moments: dict[str, float], log_z: float, truth: dict[str, float]) -> dict[str, Any]:
    if not math.isfinite(log_z) or any(not math.isfinite(v) for v in moments.values()):
        raise ValueError("finite normalizer and moment diagnostics required")
    limits = {"mean": .25, "second_moment": max(.5, .1*truth["second_moment"]), "right_mass": .10}
    absolute = {k: abs(moments[k]-truth[k]) for k in limits}
    score = max(absolute[k]/limits[k] for k in limits)
    relative_z = abs(math.expm1(log_z-truth["log_normalizer"]))
    return {"absolute_moment_errors": absolute, "moment_limits": limits, "target_error_score": score,
        "target_accurate": score <= 1, "relative_normalizer_error": relative_z,
        "normalizer_accurate": relative_z <= .20, "tolerance_specific_not_population_certificate": True}


def diagnostic_flags(diagnostics: dict[str, Any], full_gate: bool) -> dict[str, bool]:
    return {"full": full_gate, "ancestor_only": diagnostics["initial_ancestors"] >= 16
        and diagnostics["weighted_ancestor_ess"] >= 8 and diagnostics["maximum_ancestor_mass"] <= .25,
        "weight_only": diagnostics["nominal_weight_ess"] >= 64 and diagnostics["maximum_particle_weight"] <= .05}


def constructed_counterexamples(target: KnownMixtureTarget) -> dict[str, Any]:
    quantiles = torch.tensor(ndtri((np.arange(64)+.5)/64), dtype=torch.float64)
    x = torch.cat([m+target.sd*quantiles for m in target.means])[:, None]
    weights = torch.tensor([w/64 for w in target.weights for _ in range(64)], dtype=torch.float64)
    collapsed = lineage_summary(torch.zeros(128, dtype=torch.int64), weights, x)
    balanced_moments = weighted_moments(x, weights)
    y = (target.means[0]+target.sd*torch.tensor(ndtri((np.arange(128)+.5)/128), dtype=torch.float64))[:, None]
    uniform = torch.full((128,), 1/128, dtype=torch.float64)
    diverse = lineage_summary(torch.arange(128), uniform, y)
    missed_moments = weighted_moments(y, uniform)
    return {"constructed_cubature_not_IID_or_actual_SMC": True,
        "collapsed_lineage_good_moments": {"diagnostics": collapsed, "moments": balanced_moments,
            "errors": truth_errors(balanced_moments, target.log_z, target.truth()),
            "ancestor_gate": diagnostic_flags(collapsed, False)["ancestor_only"]},
        "many_ancestors_missed_mode": {"diagnostics": diverse, "moments": missed_moments,
            "errors": truth_errors(missed_moments, target.log_z, target.truth()),
            "ancestor_gate": diagnostic_flags(diverse, False)["ancestor_only"],
            "weight_gate": diagnostic_flags(diverse, False)["weight_only"]},
        "normalizer_supplied_from_formula_not_estimated_by_cubature": True,
        "movement_trace_not_fabricated_or_full_gate_claimed": True}


def confusion(predicted: list[bool], accurate: list[bool]) -> dict[str, Any]:
    if len(predicted) != len(accurate) or not predicted:
        raise ValueError("matched nonempty diagnostic labels required")
    counts = {"accept_accurate": 0, "accept_inaccurate": 0, "reject_accurate": 0, "reject_inaccurate": 0}
    for p, a in zip(predicted, accurate, strict=True):
        counts[("accept_" if p else "reject_")+("accurate" if a else "inaccurate")] += 1
    accepted, rejected = sum(predicted), len(predicted)-sum(predicted)
    return {**counts, "inaccurate_among_accepted": counts["accept_inaccurate"]/accepted if accepted else None,
        "accurate_among_rejected": counts["reject_accurate"]/rejected if rejected else None,
        "descriptive_tolerance_labels_not_learned_gate_or_confirmation": True}


def method_summary(rows: list[dict[str, Any]], truth: dict[str, float]) -> dict[str, Any]:
    if len(rows) != 32:
        raise ValueError("thirty-two independent fixed-order replicates required")
    logs = torch.tensor([r["log_normalizer"] for r in rows], dtype=torch.float64)
    values = (logs-logs.max()).exp()
    log_mean = float(logs.max()+values.mean().log())
    groups = []
    for start in range(0, 32, 4):
        stats = normalizer_summary([r["log_normalizer"] for r in rows[start:start+4]])
        error = abs(math.expm1(stats["log_mean_normalizer"]-truth["log_normalizer"]))
        groups.append({"replicate_range": [start, start+3], **stats, "known_relative_Z_error": error,
            "repeatability_qualified_but_Z_error_over_20pct": stats["qualified"] and error > .2})
    return {"log_mean_normalizer": log_mean, "rse": float(values.std(unbiased=True)/(math.sqrt(32)*values.mean())),
        "known_relative_Z_error": abs(math.expm1(log_mean-truth["log_normalizer"])),
        "maximum_replicate_fraction": float(values.max()/values.sum()),
        "individual_target_accurate": sum(r["errors"]["target_accurate"] for r in rows),
        "individual_Z_accurate": sum(r["errors"]["normalizer_accurate"] for r in rows),
        "groups_of_four": groups, "not_tail_coverage_or_actual_Volterra_qualification": True}


def rank_correlation(first: list[float], second: list[float]) -> float | None:
    if len(first) != len(second) or len(first) < 2 or not all(math.isfinite(v) for v in first+second):
        raise ValueError("matched finite rank-correlation inputs required")
    if len(set(first)) < 2 or len(set(second)) < 2:
        return None
    value = float(spearmanr(first, second).statistic)
    return value if math.isfinite(value) else None
