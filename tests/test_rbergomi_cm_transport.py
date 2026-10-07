import math

import pytest
import torch

from src.path_integral.baseline_framework import BaselineCostLedger
from src.path_integral.baselines.conditional_rbergomi import (
    evaluate_conditional_terminal_units,
    freeze_conditional_rbergomi_proposal,
)
from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.cameron_martin_modes import ActionSolverConfig, ModeSearchConfig
from src.path_integral.finite_rank_gaussian_transport import CurvatureTransportConfig
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.rbergomi_cm_transport import (
    assert_transport_unchanged,
    evaluate_rbergomi_cm_transport,
    train_rbergomi_cm_transport,
)
from src.path_integral.v15_baseline_protocol import (
    V15BaselineProtocol,
    summarize_v15_method,
    work_efficiency_ratio,
)


def _problem() -> RBergomiBaselineProblem:
    return RBergomiBaselineProblem(
        task_id="v15-end-to-end",
        task=TerminalThresholdTask(level=70.0),
        spot=100.0,
        maturity=1.0,
        steps=8,
        hurst=0.12,
        eta=1.5,
        xi=0.04,
        rho=-0.7,
    )


def test_v15_transport_is_exact_and_agrees_with_natural_conditional_mc() -> None:
    problem = _problem()
    trained = train_rbergomi_cm_transport(
        problem,
        modes_per_driver=2,
        mode_search=ModeSearchConfig(
            methods=("lbfgs", "trust-ncg"),
            random_starts=1,
            random_seed=701,
            start_scale=1.0,
            solver=ActionSolverConfig(maximum_iterations=100, gradient_tolerance=1e-6),
        ),
        transport_config=CurvatureTransportConfig(defensive_mass=0.2),
    )
    assert_transport_unchanged(trained.proposal, trained.proposal_sha256)
    candidate = evaluate_rbergomi_cm_transport(
        problem,
        trained.proposal,
        sample_count=20_000,
        path_seed=702,
        label_seed=703,
    )
    natural_proposal = freeze_conditional_rbergomi_proposal(problem, training_seed=704)
    natural = evaluate_conditional_terminal_units(
        problem,
        natural_proposal,
        sample_count=20_000,
        seed=705,
    ).unit_contributions
    difference = float(torch.mean(candidate.contribution) - torch.mean(natural))
    combined_se = math.sqrt(
        float(torch.var(candidate.contribution, unbiased=True)) / candidate.contribution.numel()
        + float(torch.var(natural, unbiased=True)) / natural.numel()
    )
    assert abs(difference) < 4.0 * combined_se + 3e-4
    assert candidate.maximum_likelihood_bound_violation <= 2e-12
    assert float(torch.max(candidate.likelihood)) <= 5.0 + 2e-12


def test_protocol_requires_all_strong_primary_comparators() -> None:
    with pytest.raises(ValueError, match="missing primary comparators"):
        V15BaselineProtocol(primary_comparators=("conditional_rbergomi",))
    protocol = V15BaselineProtocol(
        primary_comparators=(
            "conditional_rbergomi",
            "v14_local_volterra",
            "defensive_cem",
            "ld_subspace_is",
            "smoothing_rqmc",
        )
    )
    assert protocol.query_counts[-1] == 1_000


def test_work_ratio_includes_training_cost_and_query_count() -> None:
    values = torch.tensor([0.0, 0.1, 0.2, 0.3], dtype=torch.float64)
    baseline = summarize_v15_method(
        "baseline",
        values,
        training_cost=BaselineCostLedger(algorithmic_work_units=0.0),
        evaluation_cost=BaselineCostLedger(algorithmic_work_units=40.0),
    )
    candidate = summarize_v15_method(
        "candidate",
        values / 2.0,
        training_cost=BaselineCostLedger(algorithmic_work_units=1_000.0),
        evaluation_cost=BaselineCostLedger(algorithmic_work_units=80.0),
    )
    assert work_efficiency_ratio(baseline, candidate, query_count=1) < 1.0
    assert work_efficiency_ratio(baseline, candidate, query_count=10_000) > 1.0
