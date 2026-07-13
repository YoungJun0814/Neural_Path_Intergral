"""Path-integral control primitives and analytic verification oracles."""

from .action import brownian_log_likelihood, log_tilted_weight, path_action
from .divergence import TiltedDivergenceDiagnostics, tilted_divergence_diagnostics
from .gaussian_oracles import (
    gaussian_exponential_tilt_log_normalizer,
    gaussian_exponential_tilt_optimal_control,
    gaussian_exponential_tilt_pi_gap,
    gaussian_exponential_tilt_pi_objective,
    gaussian_exponential_tilt_relative_variance,
    gaussian_left_tail_doob_drift,
    gaussian_left_tail_probability,
)
from .pice import ConstantPICEFit, fit_constant_pice, reconstruct_candidate_increments
from .potentials import terminal_left_tail_potential

__all__ = [
    "ConstantPICEFit",
    "TiltedDivergenceDiagnostics",
    "brownian_log_likelihood",
    "fit_constant_pice",
    "gaussian_exponential_tilt_log_normalizer",
    "gaussian_exponential_tilt_optimal_control",
    "gaussian_exponential_tilt_pi_gap",
    "gaussian_exponential_tilt_pi_objective",
    "gaussian_exponential_tilt_relative_variance",
    "gaussian_left_tail_doob_drift",
    "gaussian_left_tail_probability",
    "log_tilted_weight",
    "path_action",
    "reconstruct_candidate_increments",
    "terminal_left_tail_potential",
    "tilted_divergence_diagnostics",
]
