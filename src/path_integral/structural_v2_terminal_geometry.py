"""Predeclared terminal geometry bins; correlated SMC particles are not IID."""

from __future__ import annotations

import math
from typing import Any

import torch

from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.path_functionals import TerminalThresholdTask

PARTITION = {"peak_time_bins": 4, "largest_left_variance_share_edges": [.5, .9, .99],
             "mode_index": "4*peak_time_quartile+dominance_bin", "mode_count": 16}


def terminal_geometry(problem: RBergomiBaselineProblem, samples: torch.Tensor,
                      weights: torch.Tensor) -> dict[str, Any]:
    if (weights.shape != (len(samples),) or weights.dtype != torch.float64 or weights.device.type != "cpu"
            or not torch.isfinite(weights).all() or bool((weights < 0).any())
            or not math.isclose(float(weights.sum()), 1., abs_tol=1e-10)):
        raise ValueError("invalid normalized terminal SMC weights")
    paths = problem.simulate_local(samples)
    variance = paths.variance[:, :-1]
    total = variance.sum(1)
    share, peak = variance.max(1)
    share = share / total
    quartile = torch.div(4 * peak, problem.steps, rounding_mode="floor")
    dominance = torch.bucketize(share, torch.tensor(PARTITION["largest_left_variance_share_edges"], dtype=torch.float64))
    mode = 4 * quartile + dominance
    masses = torch.zeros(16, dtype=torch.float64).scatter_add_(0, mode, weights)
    counts = torch.bincount(mode, minlength=16)
    integrated = problem.step_dt * total
    if not isinstance(problem.task, TerminalThresholdTask):
        raise ValueError("terminal diagnostic requires a terminal threshold")
    logg = torch.special.log_ndtr((math.log(problem.task.level) - paths.log_spot[:, -1]) /
                                  torch.sqrt((1 - problem.rho**2) * integrated))
    if not torch.isfinite(logg).all():
        raise FloatingPointError("nonfinite diagnostic conditional payoff")
    return {"partition": PARTITION, "weighted_mode_masses": masses.tolist(), "particle_mode_counts": counts.tolist(),
            "weighted_integrated_variance": float(weights @ integrated),
            "weighted_log_conditional_probability": float(weights @ logg),
            "weighted_largest_left_variance_share": float(weights @ share),
            "weighted_peak_time_fraction": float(weights @ (peak.to(torch.float64) / problem.steps)),
            "terminal_weight_ess": 1 / float(weights.square().sum()), "particles": len(samples),
            "scope": "correlated_terminal_geometry_not_iid_or_tail_coverage_certificate"}


def mode_contributions(whole_runs: list[dict[str, Any]]) -> dict[str, Any]:
    if len(whole_runs) < 2:
        raise ValueError("need independent whole runs for mode sensitivity")
    logs = torch.tensor([r["log_estimand_estimate"] for r in whole_runs], dtype=torch.float64)
    masses = torch.tensor([r["terminal_geometry"]["weighted_mode_masses"] for r in whole_runs], dtype=torch.float64)
    if (masses.shape != (len(logs), 16) or not torch.isfinite(logs).all() or not torch.isfinite(masses).all()
            or bool((masses < 0).any()) or float(torch.max(torch.abs(masses.sum(1) - 1))) > 1e-10):
        raise ValueError("invalid mode partition masses")
    scale = torch.exp(logs - logs.max())
    contributions = scale[:, None] * masses
    total = float(scale.mean())
    return {"partition": PARTITION, "whole_run_count": len(logs),
            "relative_mean_contribution": (contributions.mean(0) / total).tolist(),
            "whole_run_se_relative_to_total_mean": (contributions.std(0, unbiased=True) / math.sqrt(len(logs)) / total).tolist(),
            "whole_runs_with_mode_mass_over_1e_minus_5": (masses > 1e-5).sum(0).tolist(),
            "maximum_single_whole_run_share_of_mode_contribution": [
                float(contributions[:, i].max() / contributions[:, i].sum()) if float(contributions[:, i].sum()) > 0 else None
                for i in range(16)],
            "scope": "fixed_geometry_partition_not_fitted_clustering_or_missing_mode_exclusion"}
