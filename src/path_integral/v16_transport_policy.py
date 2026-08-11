"""Frozen structural routing policy for the V16 practical transport."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.path_functionals import TerminalThresholdTask

V16Regime = Literal[
    "deep_rough",
    "deep_rough_high_vol_of_vol",
    "deep_high_vol_of_vol",
    "deep_high_vol_of_vol_strong_negative_correlation",
    "deep_strong_negative_correlation",
    "rough",
    "strong_negative_correlation",
    "regular",
]
V16Initializer = Literal["cm_action", "tempered_smc"]


@dataclass(frozen=True)
class V16TransportRoute:
    policy_id: str
    regime: V16Regime
    initializer: V16Initializer
    drift_modes: int
    bridge_modes: int
    defensive_mass: float
    safety_mass: float
    adaptation_iterations: int
    adaptation_samples: int
    adapt_covariance: bool
    final_components: int
    tempered_particles: int = 0
    tempered_temperature_stages: int = 0
    tempered_temperature_power: float = 0.0
    tempered_mutation_steps: int = 0
    tempered_pcn_scale: float = 0.0
    tempered_replicates: int = 0
    tempered_clustering: Literal["pca_quantile", "kmeans"] = "pca_quantile"
    tempered_kmeans_iterations: int = 25

    def candidate_overrides(self) -> dict:
        overrides = {
            "mode_basis": "mesh_compatible_hybrid",
            "modes": self.drift_modes,
            "bridge_modes": self.bridge_modes,
            "defensive_mass": self.defensive_mass,
            "asymptotic_safety_mass": self.safety_mass,
            "safety_spectrum_decay": 2.0,
            "safety_spectrum_scale": 4.0,
            "safety_complement_decay": 2.0,
        }
        if self.initializer == "tempered_smc":
            overrides["conditional_adaptation"] = None
            overrides["tempered_target"] = {
                "particles": self.tempered_particles,
                "temperature_stages": self.tempered_temperature_stages,
                "temperature_power": self.tempered_temperature_power,
                "mutation_steps": self.tempered_mutation_steps,
                "pcn_scale": self.tempered_pcn_scale,
                "replicates": self.tempered_replicates,
                "components": self.final_components,
                "clustering": self.tempered_clustering,
                "kmeans_iterations": self.tempered_kmeans_iterations,
            }
        else:
            overrides["conditional_adaptation"] = {
                "iterations": self.adaptation_iterations,
                "samples_per_iteration": self.adaptation_samples,
                "smoothing": 0.7,
                "minimum_ess_fraction": 0.1,
                "minimum_variance": 0.05,
                "maximum_variance": 20.0,
                "adapt_covariance": self.adapt_covariance,
                "final_components": self.final_components,
            }
        return overrides


def route_v16_hybrid_v1(problem: RBergomiBaselineProblem) -> V16TransportRoute:
    """Resolve a route from model structure only, never from evaluation outcomes."""

    if problem.hurst <= 0.075:
        return V16TransportRoute(
            policy_id="v16_hybrid_routing_v1",
            regime="rough",
            initializer="cm_action",
            drift_modes=16,
            bridge_modes=8,
            defensive_mass=0.15,
            safety_mass=0.02,
            adaptation_iterations=4,
            adaptation_samples=8192,
            adapt_covariance=False,
            final_components=3,
        )
    if problem.rho <= -0.85:
        return V16TransportRoute(
            policy_id="v16_hybrid_routing_v1",
            regime="strong_negative_correlation",
            initializer="cm_action",
            drift_modes=8,
            bridge_modes=3,
            defensive_mass=0.10,
            safety_mass=0.01,
            adaptation_iterations=4,
            adaptation_samples=8192,
            adapt_covariance=False,
            final_components=3,
        )
    return V16TransportRoute(
        policy_id="v16_hybrid_routing_v1",
        regime="regular",
        initializer="cm_action",
        drift_modes=8,
        bridge_modes=3,
        defensive_mass=0.15,
        safety_mass=0.02,
        adaptation_iterations=3,
        adaptation_samples=4096,
        adapt_covariance=True,
        final_components=1,
    )


def route_v16_hybrid_v2(problem: RBergomiBaselineProblem) -> V16TransportRoute:
    """Route deep terminal tails without consulting a probability estimate.

    The threshold ratio is an observed task definition, not an oracle event
    probability.  Deep-tail branches are frozen from independent confirmation
    experiments; shallower tasks retain the V1 policy verbatim.
    """

    if not isinstance(problem.task, TerminalThresholdTask):
        return replace(
            route_v16_hybrid_v1(problem),
            policy_id="v16_hybrid_routing_v2",
        )
    threshold_ratio = problem.task.level / problem.spot
    if threshold_ratio > 0.02:
        v1 = route_v16_hybrid_v1(problem)
        return replace(v1, policy_id="v16_hybrid_routing_v2")
    if problem.hurst <= 0.075:
        return V16TransportRoute(
            policy_id="v16_hybrid_routing_v2",
            regime="deep_rough",
            initializer="tempered_smc",
            drift_modes=16,
            bridge_modes=8,
            defensive_mass=0.15,
            safety_mass=0.02,
            adaptation_iterations=0,
            adaptation_samples=0,
            adapt_covariance=True,
            final_components=5,
            tempered_particles=4096,
            tempered_temperature_stages=32,
            tempered_temperature_power=2.0,
            tempered_mutation_steps=4,
            tempered_pcn_scale=0.30,
            tempered_replicates=4,
        )
    if problem.eta >= 1.9:
        return V16TransportRoute(
            policy_id="v16_hybrid_routing_v2",
            regime="deep_high_vol_of_vol",
            initializer="cm_action",
            drift_modes=8,
            bridge_modes=3,
            defensive_mass=0.10,
            safety_mass=0.01,
            adaptation_iterations=10,
            adaptation_samples=16384,
            adapt_covariance=True,
            final_components=1,
        )
    if problem.rho <= -0.85:
        return V16TransportRoute(
            policy_id="v16_hybrid_routing_v2",
            regime="deep_strong_negative_correlation",
            initializer="cm_action",
            drift_modes=8,
            bridge_modes=3,
            defensive_mass=0.10,
            safety_mass=0.01,
            adaptation_iterations=4,
            adaptation_samples=8192,
            adapt_covariance=False,
            final_components=3,
        )
    v1 = route_v16_hybrid_v1(problem)
    return replace(v1, policy_id="v16_hybrid_routing_v2")


def route_v16_hybrid_v3(problem: RBergomiBaselineProblem) -> V16TransportRoute:
    """Resolve independently confirmed deep and joint-extreme routes.

    V3 leaves the frozen V2 policy unchanged.  It replaces only terminal tails at
    or below the predeclared two-percent strike ratio, using full-space target-mode
    clustering selected in the joint-regime development matrix.
    """

    if not isinstance(problem.task, TerminalThresholdTask):
        return replace(
            route_v16_hybrid_v2(problem),
            policy_id="v16_hybrid_routing_v3",
        )
    if problem.task.level / problem.spot > 0.02:
        return replace(
            route_v16_hybrid_v2(problem),
            policy_id="v16_hybrid_routing_v3",
        )
    if problem.hurst <= 0.075:
        high_eta = problem.eta >= 1.9
        return V16TransportRoute(
            policy_id="v16_hybrid_routing_v3",
            regime=(
                "deep_rough_high_vol_of_vol" if high_eta else "deep_rough"
            ),
            initializer="tempered_smc",
            drift_modes=16,
            bridge_modes=8,
            defensive_mass=0.10,
            safety_mass=0.01,
            adaptation_iterations=0,
            adaptation_samples=0,
            adapt_covariance=True,
            final_components=5,
            tempered_particles=8192 if high_eta else 4096,
            tempered_temperature_stages=32,
            tempered_temperature_power=2.0,
            tempered_mutation_steps=4,
            tempered_pcn_scale=0.30,
            tempered_replicates=4,
            tempered_clustering="kmeans",
            tempered_kmeans_iterations=25,
        )
    if problem.eta >= 1.9 and problem.rho <= -0.85:
        return V16TransportRoute(
            policy_id="v16_hybrid_routing_v3",
            regime="deep_high_vol_of_vol_strong_negative_correlation",
            initializer="tempered_smc",
            drift_modes=16,
            bridge_modes=8,
            defensive_mass=0.10,
            safety_mass=0.01,
            adaptation_iterations=0,
            adaptation_samples=0,
            adapt_covariance=True,
            final_components=5,
            tempered_particles=4096,
            tempered_temperature_stages=32,
            tempered_temperature_power=2.0,
            tempered_mutation_steps=4,
            tempered_pcn_scale=0.30,
            tempered_replicates=4,
            tempered_clustering="kmeans",
            tempered_kmeans_iterations=25,
        )
    return replace(
        route_v16_hybrid_v2(problem),
        policy_id="v16_hybrid_routing_v3",
    )
