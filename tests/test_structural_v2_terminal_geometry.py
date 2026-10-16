from __future__ import annotations

import copy
import math

import pytest
import torch

from experiments.post_audit_r1_diagnostics import _problem
from src.path_integral.structural_v2_terminal_geometry import mode_contributions, terminal_geometry
from src.path_integral.volterra_conditional_payoffs import evaluate_rbergomi_conditional_terminal


def test_geometry_matches_conditional_payoff_and_weighted_paths() -> None:
    problem = _problem({"spot": 100., "maturity": 1., "hurst": .1, "eta": 1.5, "xi": .04, "rho": -.7},
                       {"id": "toy", "threshold": 80.}, 4)
    x = torch.randn((16, 8), dtype=torch.float64, generator=torch.Generator().manual_seed(8))
    weights = torch.arange(1, 17, dtype=torch.float64)
    weights /= weights.sum()
    result = terminal_geometry(problem, x, weights)
    payoff = evaluate_rbergomi_conditional_terminal(problem, x)
    assert result["weighted_log_conditional_probability"] == pytest.approx(float(weights @ payoff.payoffs.log_left_probability), abs=1e-12)
    assert result["weighted_integrated_variance"] == pytest.approx(float(weights @ payoff.integrated_variance), abs=1e-14)
    assert sum(result["weighted_mode_masses"]) == pytest.approx(1.)
    assert sum(result["particle_mode_counts"]) == 16
    with pytest.raises(ValueError):
        terminal_geometry(problem, x, weights * 2)


def test_mode_zero_is_not_missing_mode_certificate_and_whole_run_se() -> None:
    runs = []
    for i, value in enumerate((1., 2., 3., 4.)):
        masses = [0.] * 16
        masses[0 if i < 3 else 1] = 1.
        runs.append({"log_estimand_estimate": math.log(value), "terminal_geometry": {"weighted_mode_masses": masses}})
    result = mode_contributions(runs)
    assert result["relative_mean_contribution"][:2] == pytest.approx([.6, .4])
    assert result["whole_runs_with_mode_mass_over_1e_minus_5"][:3] == [3, 1, 0]
    assert result["maximum_single_whole_run_share_of_mode_contribution"][1] == 1.
    assert result["maximum_single_whole_run_share_of_mode_contribution"][2] is None
    bad = copy.deepcopy(runs)
    bad[0]["terminal_geometry"]["weighted_mode_masses"][0] = .5
    with pytest.raises(ValueError):
        mode_contributions(bad)
