"""R1 oracle checks for exact density and conditional geometry diagnostics."""

from __future__ import annotations

import math

import torch

from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.r1_bottleneck_diagnostics import (
    conditional_log_payoff_gradients,
    fit_projected_mean_shift,
    mean_shift_proposal,
    nested_reference_complement_generic,
    summarize_log_contributions,
    weighted_target_directions,
)
from src.path_integral.tempered_conditional_smc import (
    TemperedSMCConfig,
    estimate_tempered_normalizer,
)


def test_projected_mean_shift_has_exact_dense_gaussian_density() -> None:
    direction = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
    proposal = mean_shift_proposal(direction, torch.tensor([1.2], dtype=torch.float64),
                                   defensive_mass=0.2)
    x = torch.tensor([[0.4, -0.3], [2.0, 1.0]], dtype=torch.float64)
    log_shift = 1.2 * x[:, 0] - 0.5 * 1.2**2
    expected = torch.logaddexp(
        torch.full((2,), math.log(0.2), dtype=torch.float64),
        math.log(0.8) + log_shift,
    )
    torch.testing.assert_close(proposal.log_q_over_p(x), expected, rtol=1e-12, atol=1e-12)


def test_fis_matrix_is_not_weighted_pca() -> None:
    generator = torch.Generator().manual_seed(54)
    x = torch.randn((300, 2), dtype=torch.float64, generator=generator)
    log_g = torch.zeros(300, dtype=torch.float64)
    log_p_over_q = torch.zeros_like(log_g)
    gradients = torch.stack((
        0.01 * torch.ones(300, dtype=torch.float64),
        5.0 * torch.ones(300, dtype=torch.float64),
    ), dim=1)
    direction = weighted_target_directions(
        x, log_g, log_p_over_q, rank=1, kind="fis_matrix", gradients=gradients,
    )
    assert abs(float(direction[1, 0])) > 0.999


def test_kl_and_m2_fit_use_exact_ordinary_bank_weights() -> None:
    generator = torch.Generator().manual_seed(26)
    x = torch.randn((1000, 2), dtype=torch.float64, generator=generator)
    log_g = torch.special.log_ndtr(x[:, 0] - 2.0)
    zeros = torch.zeros(1000, dtype=torch.float64)
    direction = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
    for objective in ("kl", "m2"):
        proposal, loss = fit_projected_mean_shift(
            x, log_g, zeros, direction, objective=objective, steps=50,
        )
        assert math.isfinite(loss)
        assert float(proposal.components[1].mean[0]) > 0.0
        assert proposal.defensive_mass == 0.1
        log_q = proposal.log_q_over_p(x)
        if objective == "kl":
            expected = -torch.sum(torch.softmax(log_g, dim=0) * log_q)
        else:
            expected = torch.logsumexp(2.0 * log_g - log_q, dim=0) - math.log(x.shape[0])
        assert math.isclose(loss, float(expected), rel_tol=1e-11, abs_tol=1e-11)


def test_conditional_log_payoff_gradient_matches_finite_difference() -> None:
    problem = RBergomiBaselineProblem(
        "gradient-oracle", TerminalThresholdTask(80.0), 100.0, 1.0, 4,
        0.12, 1.5, 0.04, -0.7,
    )
    x = torch.tensor([[0.2, -0.1, 0.3, 0.0, -0.4, 0.2, 0.1, -0.2]], dtype=torch.float64)
    gradient = conditional_log_payoff_gradients(problem, x, batch_size=1)
    from src.path_integral.volterra_conditional_payoffs import (
        evaluate_rbergomi_conditional_terminal,
    )

    h = 1e-5
    plus = x.clone()
    minus = x.clone()
    plus[0, 2] += h
    minus[0, 2] -= h
    y_plus = evaluate_rbergomi_conditional_terminal(problem, plus).payoffs.log_left_probability
    y_minus = evaluate_rbergomi_conditional_terminal(problem, minus).payoffs.log_left_probability
    finite_difference = float((y_plus - y_minus) / (2.0 * h))
    assert math.isclose(float(gradient[0, 2]), finite_difference, rel_tol=1e-5, abs_tol=1e-5)


def test_nested_informed_one_direction_has_zero_complement_variance() -> None:
    basis = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
    diagnostic = nested_reference_complement_generic(
        basis, log_payoff=lambda x: torch.special.log_ndtr(x[:, 0] - 1.0),
        outer_count=100, inner_counts=(16, 64, 256),
        shift=torch.tensor([1.0], dtype=torch.float64),
        outer_seed=7, inner_seed=8,
    )
    for cell in diagnostic["inner_sweep"].values():
        assert isinstance(cell, dict)
        assert abs(cell["log_floor_plugin"] - 2.0 * cell["log_mean_plugin"]) < 1e-12
        assert abs(cell["log_reference_m2_plugin"] - cell["log_floor_plugin"]) > 0.01


def test_log_summary_explicitly_marks_zero_hit_unresolved() -> None:
    zero = summarize_log_contributions(torch.full((20,), -math.inf, dtype=torch.float64))
    assert zero.log_mean is None and zero.relative_se is None
    values = summarize_log_contributions(torch.log(torch.tensor((1.0, 2.0, 3.0), dtype=torch.float64)))
    assert math.isclose(math.exp(values.log_mean or 0.0), 2.0, rel_tol=1e-12)


def test_smc_reports_lineage_and_normalizer_path_without_changing_estimate() -> None:
    result = estimate_tempered_normalizer(
        lambda x: torch.full((x.shape[0],), math.log(0.25), dtype=torch.float64),
        dimension=2,
        config=TemperedSMCConfig(
            particles=32, temperatures=(0.0, 0.5, 1.0),
            mutation_steps=1, pcn_scale=0.4, replicates=2, seed=31,
        ),
    )
    assert math.isclose(result.mean, 0.25, rel_tol=1e-12)
    assert len(result.replicate_diagnostics) == 2
    for record in result.replicate_diagnostics:
        stages = record["stages"]
        assert len(stages) == 2
        assert stages[0]["resampled"] is True
        assert stages[1]["resampled"] is False
        assert math.isclose(stages[0]["maximum_incremental_weight_fraction"], 1.0 / 32)
        assert record["final_unique_initial_ancestors"] <= 32
