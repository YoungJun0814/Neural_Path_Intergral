"""Evaluation utilities: backtesting, VaR tests, efficiency metrics."""

from .backtest import (
    christoffersen_independence,
    compute_var_series,
    efficiency_metrics,
    kupiec_pof_test,
    var_exceptions,
)
from .heston_reference import (
    HestonReferenceParams,
    HestonTerminalCDFStateDerivatives,
    heston_call_price,
    heston_characteristic_function,
    heston_left_tail_quantile,
    heston_put_price,
    heston_terminal_cdf,
    heston_terminal_cdf_state_derivatives_vectorized,
    heston_terminal_cdf_vectorized,
)
from .likelihood import LikelihoodDiagnostics, likelihood_diagnostics
from .multilevel import (
    MLMCAllocation,
    break_even_query_count,
    optimal_mlmc_sample_counts,
    single_level_online_work,
)
from .protocol import FrozenExperimentProtocol, FrozenSeedSplit, load_frozen_protocol
from .statistics import RepeatedEstimateReport, repeated_estimate_report
from .volterra_branching import (
    AdaptiveBranchedCorrection,
    BoundaryBranchingPolicy,
    BoundaryPolicyCalibration,
    boundary_branch_counts,
    calibrate_boundary_branching_policy,
    coarse_boundary_margins,
    evaluate_adaptive_branched_correction,
    evaluate_variable_branched_correction,
)

__all__ = [
    "compute_var_series",
    "var_exceptions",
    "kupiec_pof_test",
    "christoffersen_independence",
    "efficiency_metrics",
    "HestonReferenceParams",
    "HestonTerminalCDFStateDerivatives",
    "heston_characteristic_function",
    "heston_call_price",
    "heston_put_price",
    "heston_terminal_cdf",
    "heston_terminal_cdf_state_derivatives_vectorized",
    "heston_terminal_cdf_vectorized",
    "heston_left_tail_quantile",
    "LikelihoodDiagnostics",
    "AdaptiveBranchedCorrection",
    "BoundaryBranchingPolicy",
    "BoundaryPolicyCalibration",
    "MLMCAllocation",
    "break_even_query_count",
    "boundary_branch_counts",
    "calibrate_boundary_branching_policy",
    "coarse_boundary_margins",
    "evaluate_adaptive_branched_correction",
    "evaluate_variable_branched_correction",
    "likelihood_diagnostics",
    "FrozenSeedSplit",
    "FrozenExperimentProtocol",
    "load_frozen_protocol",
    "optimal_mlmc_sample_counts",
    "RepeatedEstimateReport",
    "repeated_estimate_report",
    "single_level_online_work",
]
