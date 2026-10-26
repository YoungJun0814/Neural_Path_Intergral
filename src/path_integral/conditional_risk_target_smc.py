"""Fixed-schedule SMC for q_X (m2/Dq^2)^beta; diagnostic bank, not IID data."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import torch

from .gaussian_span_marginalization import GaussianMixtureShiftSpec, sample_gaussian_mixture
from .weighted_bank_mixture import _weighted_kmeans
from .weighted_tempered_smc import _resample

Arm = Literal["prior", "component"]
Mutation = Literal["prior", "component", "global"]
STREAMS = ("initial-path", "initial-label", "global-path", "global-label", "gibbs", "noise", "accept", "resample")


@dataclass(frozen=True)
class RiskEvaluation:
    log_m2: torch.Tensor
    log_dq: torch.Tensor
    features: torch.Tensor
    maximum_relative_error_proxy: float

    @property
    def log_f(self) -> torch.Tensor:
        return self.log_m2-2*self.log_dq

    def validate(self, count: int) -> None:
        if (self.log_m2.shape != (count,) or self.log_dq.shape != (count,)
                or self.features.ndim != 2 or self.features.shape[0] != count or self.features.shape[1] < 1
                or any(t.dtype != torch.float64 or t.device.type != "cpu" or t.requires_grad
                       or not torch.isfinite(t).all() for t in (self.log_m2, self.log_dq, self.features))
                or not torch.isfinite(self.log_f).all()
                or bool((self.log_m2 > 1e-10).any())
                or not math.isfinite(self.maximum_relative_error_proxy)
                or not 0 <= self.maximum_relative_error_proxy <= 1e-6):
            raise ValueError("every proposal needs finite m2, full Dq and resolved individual numerical proxy")


@dataclass(frozen=True)
class TargetSMCConfig:
    particles: int
    temperatures: tuple[float, ...]
    arm: Arm
    scale: float = .35
    resample_every: int = 4

    def __post_init__(self) -> None:
        if (type(self.particles) is not int or self.particles < 4 or type(self.resample_every) is not int
                or self.resample_every < 1 or self.arm not in ("prior", "component")
                or not math.isfinite(self.scale) or not 0 < self.scale < 1
                or len(self.temperatures) < 3 or self.temperatures[0] != 0 or self.temperatures[-1] != 1
                or any(not math.isfinite(b) or not 0 <= b <= 1 for b in self.temperatures)
                or any(b <= a for a, b in zip(self.temperatures[:-1], self.temperatures[1:], strict=True))):
            raise ValueError("fixed valid particle/schedule/arm/scale configuration required")


def mutation_ratio(kind: Mutation, beta: float, old: RiskEvaluation, new: RiskEvaluation) -> torch.Tensor:
    if kind not in ("prior", "component", "global") or not math.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("declared mutation and beta in [0,1] required")
    ratio = beta*(new.log_f-old.log_f)
    return ratio+new.log_dq-old.log_dq if kind == "prior" else ratio


def lineage_summary(lineage: torch.Tensor, weights: torch.Tensor, x: torch.Tensor) -> dict[str, Any]:
    n = len(weights)
    if (lineage.shape != (n,) or lineage.dtype != torch.int64 or bool((lineage < 0).any())
            or bool((lineage >= n).any()) or weights.dtype != torch.float64 or weights.device.type != "cpu"
            or not torch.isfinite(weights).all() or bool((weights < 0).any()) or abs(float(weights.sum())-1) > 1e-10
            or x.ndim != 2 or len(x) != n or not torch.isfinite(x).all()):
        raise ValueError("valid normalized weights and original ancestry required")
    masses = torch.zeros(n, dtype=torch.float64).scatter_add_(0, lineage, weights)
    return {"initial_ancestors": int(lineage.unique().numel()), "weighted_ancestor_ess": float(1/masses.square().sum()),
        "maximum_ancestor_mass": float(masses.max()), "nominal_weight_ess": float(1/weights.square().sum()),
        "maximum_particle_weight": float(weights.max()), "distinct_particles": int(torch.unique(x, dim=0).shape[0]),
        "not_independent_sample_ess": True}


def island_gate(diagnostics: dict[str, Any], stages: list[dict[str, Any]], late_jump: torch.Tensor) -> dict[str, Any]:
    late = stages[len(stages)//2:]
    accepted = sum(sum(s["accepted"].values()) for s in late)
    proposed = sum(sum(s["proposed"].values()) for s in late)
    rate = accepted/proposed if proposed else 0.
    median_jump = float(torch.quantile(late_jump, .5))
    tests = {"ancestors": diagnostics["initial_ancestors"] >= 16,
        "ancestor_ess": diagnostics["weighted_ancestor_ess"] >= 8,
        "ancestor_mass": diagnostics["maximum_ancestor_mass"] <= .25,
        "nominal_ess": diagnostics["nominal_weight_ess"] >= 64,
        "particle_mass": diagnostics["maximum_particle_weight"] <= .05,
        "distinct": diagnostics["distinct_particles"] >= 96,
        "late_acceptance": rate >= .1, "late_jump": median_jump >= .5}
    return {"checks": tests, "pass": all(tests.values()), "late_acceptance": rate,
        "late_median_cumulative_squared_jump": median_jump,
        "conservative_development_gate_not_necessary_mixing_theorem": True}


def run_target_smc(q: GaussianMixtureShiftSpec, evaluate: Callable[[torch.Tensor], RiskEvaluation], *,
                   config: TargetSMCConfig, seeds: dict[str, int],
                   observer: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    if (set(seeds) != set(STREAMS) or len(set(seeds.values())) != len(STREAMS)
            or any(type(s) is not int or not 0 < s < 2**63 for s in seeds.values())
            or q.means.dtype != torch.float64 or q.means.device.type != "cpu"
            or q.means.requires_grad or q.weights.requires_grad):
        raise ValueError("separate valid streams and detached CPU float64 mixture required")
    generators = {k: torch.Generator().manual_seed(v) for k, v in seeds.items()}
    n = config.particles

    def draw(prefix: str) -> torch.Tensor:
        return sample_gaussian_mixture(q, n, gaussian_generator=generators[prefix+"-path"],
            label_generator=generators[prefix+"-label"]).target_coordinates

    def checked(x: torch.Tensor) -> RiskEvaluation:
        result = evaluate(x)
        result.validate(n)
        direct = torch.logsumexp(x @ q.means.T-.5*q.means.square().sum(1)+q.weights.log(), 1)
        if not torch.allclose(direct, result.log_dq, rtol=0, atol=1e-10):
            raise ValueError("potential Dq must match actual normalized starting q")
        return result

    x = draw("initial")
    value = checked(x)
    weights = torch.full((n,), -math.log(n), dtype=torch.float64)
    lineage = torch.arange(n)
    log_z = 0.
    stages: list[dict[str, Any]] = []
    late_jump = torch.zeros(n, dtype=torch.float64)
    max_error = value.maximum_relative_error_proxy
    retained = math.sqrt(1-config.scale**2)
    calls = n
    total_stages = len(config.temperatures)-1
    for stage, (left, right) in enumerate(zip(config.temperatures[:-1], config.temperatures[1:], strict=True)):
        updated = weights+(right-left)*value.log_f
        increment = float(updated.logsumexp(0))
        if not math.isfinite(increment):
            raise FloatingPointError("finite normalizer increment required")
        log_z += increment
        weights = updated-increment
        pre = lineage_summary(lineage, weights.exp(), x)
        resampled = (stage+1) % config.resample_every == 0 and stage+1 != total_stages
        if resampled:
            indices = _resample(weights.exp(), generator=generators["resample"], scheme="stratified")
            x, lineage, late_jump = x[indices], lineage[indices], late_jump[indices]
            value = RiskEvaluation(value.log_m2[indices], value.log_dq[indices], value.features[indices], value.maximum_relative_error_proxy)
            weights.fill_(-math.log(n))
        accepted: dict[str, int] = {}
        proposed: dict[str, int] = {}
        jump_total = 0.
        kinds: tuple[Mutation, ...] = (("component", "prior", "global") if config.arm == "component"
                                      else ("prior", "prior", "global"))
        for kind in kinds:
            if kind == "global":
                candidate = draw("global")
            else:
                noise = torch.randn(x.shape, dtype=torch.float64, generator=generators["noise"])
                if kind == "component":
                    logits = x @ q.means.T-.5*q.means.square().sum(1)+q.weights.log()
                    label = torch.multinomial(logits.softmax(1), 1, generator=generators["gibbs"])[:, 0]
                    mean = q.means[label]
                    candidate = mean+retained*(x-mean)+config.scale*noise
                else:
                    candidate = retained*x+config.scale*noise
            new = checked(candidate)
            calls += n
            max_error = max(max_error, new.maximum_relative_error_proxy)
            ratio = mutation_ratio(kind, right, value, new)
            if not torch.isfinite(ratio).all():
                raise FloatingPointError("nonfinite MH ratio; no clipping or silent rejection")
            take = torch.rand(n, dtype=torch.float64, generator=generators["accept"]).log() < torch.minimum(ratio, torch.zeros_like(ratio))
            jump = (candidate-x).square().sum(1)*take
            if stage >= total_stages//2:
                late_jump += jump
            jump_total += float(jump.sum())
            accepted[kind] = accepted.get(kind, 0)+int(take.sum())
            proposed[kind] = proposed.get(kind, 0)+n
            x = torch.where(take[:, None], candidate, x)
            value = RiskEvaluation(torch.where(take, new.log_m2, value.log_m2), torch.where(take, new.log_dq, value.log_dq),
                torch.where(take[:, None], new.features, value.features), max_error)
        row = {"stage": stage, "beta": right, "resampled": resampled, "pre_resampling": pre,
            "post_mutation": lineage_summary(lineage, weights.exp(), x), "accepted": accepted,
            "proposed": proposed, "accepted_squared_jump_sum": jump_total, "log_normalizer": log_z}
        stages.append(row)
        if observer:
            observer(row)
    diagnostics = lineage_summary(lineage, weights.exp(), x)
    return {"log_normalizer": log_z, "potential_contexts": calls, "stages": stages,
        "final_x": x.tolist(), "final_weights": weights.exp().tolist(), "final_log_m2": value.log_m2.tolist(),
        "final_log_dq": value.log_dq.tolist(), "final_features": value.features.tolist(),
        "initial_lineage": lineage.tolist(), "late_jump_by_particle": late_jump.tolist(),
        "diagnostics": diagnostics, "gate": island_gate(diagnostics, stages, late_jump),
        "maximum_relative_numerical_proxy": max_error,
        "normalizer_for_numerical_m2_not_certified_exact_true_M2": True}


def normalizer_summary(log_values: list[float]) -> dict[str, Any]:
    if len(log_values) != 4 or not all(math.isfinite(x) for x in log_values):
        raise ValueError("four complete independent islands required")
    logs = torch.tensor(log_values, dtype=torch.float64)
    values = (logs-logs.max()).exp()
    mean = values.mean()
    rse = float(values.std(unbiased=True)/(2*mean))
    max_fraction, spread = float(values.max()/values.sum()), max(log_values)-min(log_values)
    return {"log_mean_normalizer": float(logs.max()+mean.log()), "rse": rse,
        "maximum_island_fraction": max_fraction, "log_spread": spread,
        "qualified": rse <= .2 and max_fraction <= .5 and spread <= math.log(4),
        "four_islands_not_population_or_tail_coverage_certificate": True}


def mode_diagnostics(islands: list[dict[str, Any]]) -> dict[str, Any]:
    if len(islands) != 4:
        raise ValueError("four complete primary banks required")
    features = torch.tensor([v for i in islands for v in i["final_features"]], dtype=torch.float64)
    weights = torch.tensor([w/4 for i in islands for w in i["final_weights"]], dtype=torch.float64)
    mean = weights @ features
    scale = torch.sqrt((weights[:, None]*(features-mean).square()).sum(0).clamp_min(1e-8))
    transformed = (features-mean)/scale
    labels, centers = _weighted_kmeans(transformed, weights, clusters=3, iterations=20)
    masses = torch.zeros((4, 3), dtype=torch.float64)
    offset = 0
    for j, island in enumerate(islands):
        count = len(island["final_weights"])
        w = torch.tensor(island["final_weights"], dtype=torch.float64)
        for k in range(3):
            masses[j, k] = w[labels[offset:offset+count] == k].sum()
        offset += count
    pooled = masses.mean(0)
    passed = bool(((pooled < .1)[None, :] | (masses >= .01)).all())
    return {"feature_mean": mean.tolist(), "feature_scale": scale.tolist(), "centers": centers.tolist(),
        "pooled_cluster_masses": pooled.tolist(), "island_cluster_masses": masses.tolist(), "pass": passed,
        "descriptive_training_bank_clustering_not_heldout_or_mode_completeness_proof": True}
