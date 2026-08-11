"""Frozen structural routing policy for the V16 practical transport."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem

V16Regime = Literal["rough", "strong_negative_correlation", "regular"]


@dataclass(frozen=True)
class V16TransportRoute:
    policy_id: str
    regime: V16Regime
    drift_modes: int
    bridge_modes: int
    defensive_mass: float
    safety_mass: float
    adaptation_iterations: int
    adaptation_samples: int
    adapt_covariance: bool
    final_components: int

    def candidate_overrides(self) -> dict:
        return {
            "mode_basis": "mesh_compatible_hybrid",
            "modes": self.drift_modes,
            "bridge_modes": self.bridge_modes,
            "defensive_mass": self.defensive_mass,
            "asymptotic_safety_mass": self.safety_mass,
            "safety_spectrum_decay": 2.0,
            "safety_spectrum_scale": 4.0,
            "safety_complement_decay": 2.0,
            "conditional_adaptation": {
                "iterations": self.adaptation_iterations,
                "samples_per_iteration": self.adaptation_samples,
                "smoothing": 0.7,
                "minimum_ess_fraction": 0.1,
                "minimum_variance": 0.05,
                "maximum_variance": 20.0,
                "adapt_covariance": self.adapt_covariance,
                "final_components": self.final_components,
            },
        }


def route_v16_hybrid_v1(problem: RBergomiBaselineProblem) -> V16TransportRoute:
    """Resolve a route from model structure only, never from evaluation outcomes."""

    common = {"policy_id": "v16_hybrid_routing_v1"}
    if problem.hurst <= 0.075:
        return V16TransportRoute(
            **common,
            regime="rough",
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
            **common,
            regime="strong_negative_correlation",
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
        **common,
        regime="regular",
        drift_modes=8,
        bridge_modes=3,
        defensive_mass=0.15,
        safety_mass=0.02,
        adaptation_iterations=3,
        adaptation_samples=4096,
        adapt_covariance=True,
        final_components=1,
    )
