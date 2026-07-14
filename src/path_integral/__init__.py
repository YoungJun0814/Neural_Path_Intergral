"""Path-integral control primitives and analytic verification oracles."""

from .action import brownian_log_likelihood, log_tilted_weight, path_action
from .controllers import (
    CEMAnchoredResidualControl,
    ConstantTwoDriverControl,
    LeanRBergomiControl,
    RBergomiTaskMode,
    VFOBranchDiagnostics,
    VolterraFollmerOperator,
)
from .divergence import TiltedDivergenceDiagnostics, tilted_divergence_diagnostics
from .gaussian_mixture_oracle import (
    gaussian_single_drift_second_moment,
    gaussian_symmetric_mixture_log_q_over_p,
    gaussian_symmetric_mixture_second_moment,
    gaussian_two_tail_probability,
)
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
from .mixture import (
    all_expert_log_q_over_p,
    log_mixture_q_over_p,
    sample_mixture_labels,
    selected_component_log_p_over_q,
)
from .pice import ConstantPICEFit, fit_constant_pice, reconstruct_candidate_increments
from .potentials import terminal_left_tail_potential
from .rbergomi_mixture import (
    RBergomiMixtureSample,
    replay_rbergomi_control_on_target_paths,
    simulate_rbergomi_mixture,
)

__all__ = [
    "ConstantPICEFit",
    "CEMAnchoredResidualControl",
    "ConstantTwoDriverControl",
    "LeanRBergomiControl",
    "RBergomiMixtureSample",
    "RBergomiTaskMode",
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
    "replay_rbergomi_control_on_target_paths",
    "SOEKernelBank",
    "fit_positive_soe_kernel",
    "all_expert_log_q_over_p",
    "gaussian_single_drift_second_moment",
    "gaussian_symmetric_mixture_log_q_over_p",
    "gaussian_symmetric_mixture_second_moment",
    "gaussian_two_tail_probability",
    "log_mixture_q_over_p",
    "sample_mixture_labels",
    "selected_component_log_p_over_q",
    "simulate_rbergomi_mixture",
    "terminal_left_tail_potential",
    "tilted_divergence_diagnostics",
]
