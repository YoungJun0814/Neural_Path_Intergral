"""P4 common lifecycle, exact-likelihood, and cost-oracle tests."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import torch
from scipy.special import logsumexp

from src.path_integral import (
    BASELINE_METHODS,
    BaselineCostLedger,
    audit_baseline_lifecycle,
    evaluate_baseline_log_q_over_p,
    finalize_baseline_estimate,
    freeze_baseline_proposal,
    ordinary_is_contributions,
    plan_baseline_allocation,
    sample_baseline_proposal,
)
from src.path_integral.baseline_framework import BaselineMethod


def _training_cost(trained: bool) -> BaselineCostLedger:
    if not trained:
        return BaselineCostLedger()
    return BaselineCostLedger(
        training_samples=128,
        optimizer_steps=8,
        hyperparameter_trials=2,
        failed_restarts=1,
        screening_samples=32,
        algorithmic_work_units=512.0,
        wall_seconds=1.0,
        cpu_seconds=1.0,
        peak_memory_bytes=1024,
        measurement_mode="standardized_hardware_wall",
    )


def _proposal(method: BaselineMethod):
    if method in {"crude_mc", "antithetic_mc"}:
        return freeze_baseline_proposal(
            method=method,
            task_id="terminal-h012-p1e-4",
            dimension=3,
            training_seed=101,
            training_cost=_training_cost(False),
        )
    if method in {"conditional_rbergomi", "smoothing_rqmc"}:
        return freeze_baseline_proposal(
            method=method,
            task_id="terminal-h012-p1e-4",
            dimension=3,
            training_seed=101,
            training_cost=_training_cost(False),
            conditional_integral="analytic_gaussian_cdf",
        )
    if method == "pure_cem":
        return freeze_baseline_proposal(
            method=method,
            task_id="terminal-h012-p1e-4",
            dimension=3,
            training_seed=101,
            training_cost=_training_cost(True),
            training_budget_work_units=600.0,
            location=(-0.8, 0.2, -0.1),
        )
    if method == "defensive_cem":
        return freeze_baseline_proposal(
            method=method,
            task_id="terminal-h012-p1e-4",
            dimension=3,
            training_seed=101,
            training_cost=_training_cost(True),
            training_budget_work_units=600.0,
            component_means=((0.0, 0.0, 0.0), (-0.8, 0.2, -0.1)),
            component_weights=(0.2, 0.8),
        )
    if method == "ld_subspace_is":
        return freeze_baseline_proposal(
            method=method,
            task_id="terminal-h012-p1e-4",
            dimension=3,
            training_seed=101,
            training_cost=_training_cost(True),
            training_budget_work_units=600.0,
            component_means=((-1.1, 0.3, 0.0), (0.2, -0.5, 0.0)),
            component_weights=(0.7, 0.3),
        )
    if method == "flow_is":
        return freeze_baseline_proposal(
            method=method,
            task_id="terminal-h012-p1e-4",
            dimension=3,
            training_seed=101,
            training_cost=_training_cost(True),
            training_budget_work_units=600.0,
            location=(-0.3, 0.2, 0.1),
            flow_split=1,
            flow_scale_matrix=((0.4,), (-0.3,)),
            flow_scale_bias=(0.1, -0.2),
            flow_shift_matrix=((0.2,), (0.1,)),
            flow_shift_bias=(-0.1, 0.2),
            flow_max_log_scale=0.5,
            conditional_integral="baseline_only",
        )
    raise AssertionError(f"unhandled test method: {method}")


def _final_cost(proposal, final_samples: int) -> BaselineCostLedger:
    tilted = proposal.family not in {
        "target_gaussian",
        "rqmc_target_gaussian",
    }
    analytic = proposal.conditional_integral == "analytic_gaussian_cdf"
    quadrature = proposal.conditional_integral == "rigorous_quadrature"
    return BaselineCostLedger(
        final_samples=final_samples,
        likelihood_evaluations=final_samples if tilted else 0,
        cdf_calls=final_samples if analytic else 0,
        quadrature_calls=final_samples if quadrature else 0,
        algorithmic_work_units=float(final_samples * 10),
        wall_seconds=0.1,
        cpu_seconds=0.1,
        peak_memory_bytes=2048,
        measurement_mode="standardized_hardware_wall",
    )


def test_registry_contains_every_predeclared_p4_baseline() -> None:
    assert BASELINE_METHODS == (
        "crude_mc",
        "antithetic_mc",
        "conditional_rbergomi",
        "pure_cem",
        "defensive_cem",
        "smoothing_rqmc",
        "ld_subspace_is",
        "flow_is",
    )
    assert {_proposal(method).method for method in BASELINE_METHODS} == set(
        BASELINE_METHODS
    )


def test_target_and_single_shift_likelihoods_are_exact() -> None:
    samples = torch.tensor(
        [[-1.0, 0.2, 0.7], [0.5, -0.4, 1.1]],
        dtype=torch.float64,
    )
    target = _proposal("crude_mc")
    shift = _proposal("pure_cem")

    assert torch.equal(
        evaluate_baseline_log_q_over_p(samples, target),
        torch.zeros(2, dtype=torch.float64),
    )
    mean = torch.tensor(shift.location, dtype=torch.float64)
    expected = samples @ mean - 0.5 * torch.sum(mean.square())
    assert torch.allclose(
        evaluate_baseline_log_q_over_p(samples, shift),
        expected,
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.parametrize("method", ["defensive_cem", "ld_subspace_is"])
def test_mixture_likelihood_matches_independent_scipy_density(
    method: BaselineMethod,
) -> None:
    proposal = _proposal(method)
    samples = np.asarray(
        [[-1.0, 0.2, 0.7], [0.5, -0.4, 1.1]],
        dtype=np.float64,
    )
    observed = evaluate_baseline_log_q_over_p(
        torch.from_numpy(samples),
        proposal,
    ).numpy()
    normalizer = -1.5 * math.log(2.0 * math.pi)
    target_log = np.asarray(
        [normalizer - 0.5 * float(np.sum(sample**2)) for sample in samples]
    )
    component_logs = np.column_stack(
        [
            math.log(weight)
            + np.asarray(
                [
                    normalizer
                    - 0.5
                    * sum(
                        (float(value) - float(center)) ** 2
                        for value, center in zip(sample, mean, strict=True)
                    )
                    for sample in samples
                ]
            )
            for mean, weight in zip(
                proposal.component_means,
                proposal.component_weights,
                strict=True,
            )
        ]
    )
    expected = logsumexp(component_logs, axis=1) - target_log

    assert np.allclose(observed, expected, atol=2e-15, rtol=0.0)


def test_nonlinear_coupling_flow_likelihood_matches_manual_inverse() -> None:
    proposal = _proposal("flow_is")
    samples = np.asarray(
        [[-1.0, 0.2, 0.7], [0.5, -0.4, 1.1]],
        dtype=np.float64,
    )
    observed = evaluate_baseline_log_q_over_p(
        torch.from_numpy(samples),
        proposal,
    ).numpy()
    expected = []
    for sample in samples:
        first = float(sample[0]) - proposal.location[0]
        log_scale = [
            proposal.flow_max_log_scale
            * math.tanh(weight[0] * first + bias)
            for weight, bias in zip(
                proposal.flow_scale_matrix,
                proposal.flow_scale_bias,
                strict=True,
            )
        ]
        shift = [
            weight[0] * first + bias
            for weight, bias in zip(
                proposal.flow_shift_matrix,
                proposal.flow_shift_bias,
                strict=True,
            )
        ]
        latent_second = [
            (float(value) - center - translated) * math.exp(-scale)
            for value, center, translated, scale in zip(
                sample[1:],
                proposal.location[1:],
                shift,
                log_scale,
                strict=True,
            )
        ]
        latent = [first, *latent_second]
        expected.append(
            0.5
            * (
                sum(float(value) ** 2 for value in sample)
                - sum(value**2 for value in latent)
            )
            - sum(log_scale)
        )

    assert np.allclose(observed, np.asarray(expected), atol=2e-15, rtol=0.0)
    assert not proposal.dcs_extension_eligible
    assert proposal.conditional_integral == "baseline_only"


def test_antithetic_sampler_returns_adjacent_exact_pairs() -> None:
    samples = sample_baseline_proposal(
        _proposal("antithetic_mc"),
        sample_count=16,
        seed=404,
    )

    assert torch.equal(
        samples.reshape(-1, 2, 3).sum(dim=1),
        torch.zeros((8, 3), dtype=torch.float64),
    )


def test_rqmc_sampler_uses_scrambled_power_of_two_replicates() -> None:
    proposal = _proposal("smoothing_rqmc")
    first = sample_baseline_proposal(proposal, sample_count=32, seed=501)
    replay = sample_baseline_proposal(proposal, sample_count=32, seed=501)
    independent = sample_baseline_proposal(proposal, sample_count=32, seed=502)

    assert torch.equal(first, replay)
    assert not torch.equal(first, independent)
    assert torch.isfinite(first).all()
    with pytest.raises(ValueError, match="power of two"):
        sample_baseline_proposal(proposal, sample_count=24, seed=501)


@pytest.mark.parametrize("method", BASELINE_METHODS)
def test_every_baseline_passes_the_common_lifecycle_oracle(
    method: BaselineMethod,
) -> None:
    proposal = _proposal(method)
    points = 16 if method == "smoothing_rqmc" else None
    pilot_points = (
        16
        if method == "smoothing_rqmc"
        else 2
        if method == "antithetic_mc"
        else 1
    )
    plan = plan_baseline_allocation(
        proposal,
        pilot_variance=0.2,
        target_variance=0.05,
        pilot_seed=202,
        final_seed=303,
        pilot_units=8,
        points_per_unit=points,
        planning_cost=BaselineCostLedger(
            planning_samples=8 * pilot_points,
            algorithmic_work_units=80.0,
            wall_seconds=0.01,
            cpu_seconds=0.01,
            measurement_mode="standardized_hardware_wall",
        ),
    )
    values = torch.linspace(0.01, 0.04, plan.planned_units)
    artifact = finalize_baseline_estimate(
        proposal,
        plan,
        values,
        final_cost=_final_cost(proposal, plan.planned_final_samples),
    )
    audit = audit_baseline_lifecycle(proposal, plan, artifact)

    assert plan.planned_units == 4
    assert artifact.final_sample_count == plan.planned_units * plan.points_per_unit
    assert artifact.estimator_variance == pytest.approx(
        artifact.unit_sample_variance / artifact.unit_count
    )
    assert audit.failures == ()
    assert audit.passed
    assert all(dict(audit.checks).values())


def test_rqmc_variance_is_over_randomizations_not_within_net_points() -> None:
    proposal = _proposal("smoothing_rqmc")
    plan = plan_baseline_allocation(
        proposal,
        pilot_variance=0.2,
        target_variance=0.05,
        pilot_seed=202,
        final_seed=303,
        pilot_units=8,
        points_per_unit=32,
        planning_cost=BaselineCostLedger(
            planning_samples=256,
            algorithmic_work_units=8.0,
        ),
    )
    artifact = finalize_baseline_estimate(
        proposal,
        plan,
        torch.tensor([0.1, 0.2, 0.15, 0.18]),
        final_cost=_final_cost(proposal, 128),
    )

    assert artifact.inferential_unit == "rqmc_randomization"
    assert artifact.unit_count == 4
    assert artifact.points_per_unit == 32
    assert artifact.final_sample_count == 128


def test_ordinary_is_is_unclipped_and_not_self_normalized() -> None:
    values = torch.tensor([1.0, 0.0, 0.5], dtype=torch.float64)
    log_q_over_p = torch.tensor([0.2, -0.1, 0.0], dtype=torch.float64)
    contributions = ordinary_is_contributions(values, log_q_over_p)

    assert torch.allclose(
        contributions,
        values * torch.exp(-log_q_over_p),
        atol=0.0,
        rtol=0.0,
    )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ordinary_is_contributions(
            torch.tensor([1.1], dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64),
        )


def test_finalization_rejects_omitted_likelihood_and_conditional_costs() -> None:
    proposal = _proposal("pure_cem")
    plan = plan_baseline_allocation(
        proposal,
        pilot_variance=0.1,
        target_variance=0.05,
        pilot_seed=202,
        final_seed=303,
        pilot_units=4,
    )
    with pytest.raises(ValueError, match="likelihood"):
        finalize_baseline_estimate(
            proposal,
            plan,
            torch.tensor([0.1, 0.2]),
            final_cost=BaselineCostLedger(
                final_samples=2,
                algorithmic_work_units=2.0,
            ),
        )

    conditional = _proposal("conditional_rbergomi")
    conditional_plan = plan_baseline_allocation(
        conditional,
        pilot_variance=0.1,
        target_variance=0.05,
        pilot_seed=202,
        final_seed=303,
        pilot_units=4,
    )
    artifact = finalize_baseline_estimate(
        conditional,
        conditional_plan,
        torch.tensor([0.1, 0.2]),
        final_cost=BaselineCostLedger(
            final_samples=2,
            algorithmic_work_units=2.0,
        ),
    )
    audit = audit_baseline_lifecycle(conditional, conditional_plan, artifact)
    assert not audit.passed
    assert "conditional_integrals_charged" in audit.failures


def test_seed_overlap_and_noninteger_allocation_fail_closed() -> None:
    proposal = _proposal("crude_mc")
    with pytest.raises(ValueError, match="disjoint"):
        plan_baseline_allocation(
            proposal,
            pilot_variance=0.1,
            target_variance=0.05,
            pilot_seed=101,
            final_seed=303,
            pilot_units=4,
        )
    with pytest.raises(ValueError, match="integers"):
        plan_baseline_allocation(
            proposal,
            pilot_variance=0.1,
            target_variance=0.05,
            pilot_seed=202,
            final_seed=303,
            pilot_units=4.5,  # type: ignore[arg-type]
        )


def test_training_budget_and_defensive_component_fail_closed() -> None:
    with pytest.raises(ValueError, match="budget"):
        freeze_baseline_proposal(
            method="pure_cem",
            task_id="task",
            dimension=2,
            training_seed=1,
            training_cost=_training_cost(True),
            training_budget_work_units=100.0,
            location=(-1.0, 0.0),
        )
    with pytest.raises(ValueError, match="zero-mean"):
        freeze_baseline_proposal(
            method="defensive_cem",
            task_id="task",
            dimension=2,
            training_seed=1,
            training_cost=_training_cost(True),
            training_budget_work_units=600.0,
            component_means=((-1.0, 0.0),),
            component_weights=(1.0,),
        )


def test_flow_rejects_invalid_coupling_and_dcs_relabelling() -> None:
    parameters = {
        "flow_scale_matrix": ((0.2,),),
        "flow_scale_bias": (0.0,),
        "flow_shift_matrix": ((0.1,),),
        "flow_shift_bias": (0.0,),
        "flow_max_log_scale": 0.5,
    }
    with pytest.raises(ValueError, match="split"):
        freeze_baseline_proposal(
            method="flow_is",
            task_id="task",
            dimension=2,
            training_seed=1,
            training_cost=_training_cost(True),
            training_budget_work_units=600.0,
            location=(0.0, 0.0),
            flow_split=0,
            **parameters,
            conditional_integral="baseline_only",
        )
    with pytest.raises(ValueError, match="baseline"):
        freeze_baseline_proposal(
            method="flow_is",
            task_id="task",
            dimension=2,
            training_seed=1,
            training_cost=_training_cost(True),
            training_budget_work_units=600.0,
            location=(0.0, 0.0),
            flow_split=1,
            **parameters,
            conditional_integral="analytic_gaussian_cdf",
        )


def test_cost_ledger_rejects_fractional_counts_and_unbacked_cost_modes() -> None:
    with pytest.raises(ValueError, match="integer"):
        BaselineCostLedger(final_samples=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive compute"):
        BaselineCostLedger(measurement_mode="provider_billed")
    with pytest.raises(ValueError, match="positive energy"):
        BaselineCostLedger(measurement_mode="energy_metered")


def test_proposal_hash_detects_tampering() -> None:
    proposal = _proposal("pure_cem")
    plan = plan_baseline_allocation(
        proposal,
        pilot_variance=0.1,
        target_variance=0.05,
        pilot_seed=202,
        final_seed=303,
        pilot_units=4,
    )
    artifact = finalize_baseline_estimate(
        proposal,
        plan,
        torch.tensor([0.1, 0.2]),
        final_cost=_final_cost(proposal, 2),
    )
    tampered = replace(proposal, location=(-9.0, 0.2, -0.1))
    audit = audit_baseline_lifecycle(tampered, plan, artifact)

    assert not audit.passed
    assert "proposal_hash_exact" in audit.failures
