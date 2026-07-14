"""Path-integral control primitives and analytic verification oracles."""

from .action import brownian_log_likelihood, log_tilted_weight, path_action
from .controllers import VFOBranchDiagnostics, VolterraFollmerOperator
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
from .heston_oracle import (
    HestonLogDesirabilityGradient,
    HestonOracleControl,
    HestonOracleNumerics,
    heston_log_desirability_gradient,
    heston_soft_left_tail_desirability,
    heston_soft_oracle_control,
)
from .memory import SOEKernelBank, fit_positive_soe_kernel
from .pice import ConstantPICEFit, fit_constant_pice, reconstruct_candidate_increments
from .potentials import terminal_left_tail_potential

__all__ = [
    "ConstantPICEFit",
    "TiltedDivergenceDiagnostics",
    "VFOBranchDiagnostics",
    "VolterraFollmerOperator",
    "HestonLogDesirabilityGradient",
    "HestonOracleControl",
    "HestonOracleNumerics",
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
    "heston_log_desirability_gradient",
    "heston_soft_left_tail_desirability",
    "heston_soft_oracle_control",
    "path_action",
    "reconstruct_candidate_increments",
    "SOEKernelBank",
    "fit_positive_soe_kernel",
    "terminal_left_tail_potential",
    "tilted_divergence_diagnostics",
]
