"""Exact algebra and fail-closed provenance tests for the V8 P3 MLMC contract."""

from __future__ import annotations

import math

import pytest
import torch

from src.path_integral import (
    DiscreteBarrierHitTask,
    conservative_terminal_rbergomi_complexity,
    evaluate_rbergomi_threshold_coupling,
    mlmc_complexity_certificate,
    terminal_rate_contract,
)


def _certificate(
    alpha: float,
    beta: float,
    gamma: float,
    *,
    log_power: float = 0.0,
    bias_evidence: str = "proved_internal",
    variance_evidence: str = "proved_internal",
    cost_evidence: str = "proved_internal",
):
    return mlmc_complexity_certificate(
        weak_bias_exponent=alpha,
        correction_variance_exponent=beta,
        sample_cost_exponent=gamma,
        sample_cost_log_power=log_power,
        weak_bias_evidence=bias_evidence,  # type: ignore[arg-type]
        correction_variance_evidence=variance_evidence,  # type: ignore[arg-type]
        sample_cost_evidence=cost_evidence,  # type: ignore[arg-type]
    )


def test_beta_greater_gamma_has_canonical_power_without_active_log_factor() -> None:
    certificate = _certificate(1.0, 2.0, 1.0, log_power=1.0)

    assert certificate.compatibility_condition_satisfied
    assert certificate.regime == "beta_greater_gamma"
    assert certificate.epsilon_polynomial_exponent == pytest.approx(2.0)
    assert certificate.epsilon_log_power == pytest.approx(0.0)
    assert certificate.finest_sample_polynomial_exponent == pytest.approx(1.0)
    assert certificate.unconditional_complexity_authorized


def test_beta_greater_gamma_boundary_retains_finest_fft_log_factor() -> None:
    certificate = _certificate(0.5, 2.0, 1.0, log_power=1.0)

    assert certificate.regime == "beta_greater_gamma"
    assert certificate.epsilon_polynomial_exponent == pytest.approx(2.0)
    assert certificate.epsilon_log_power == pytest.approx(1.0)
    assert certificate.finest_sample_polynomial_exponent == pytest.approx(2.0)


def test_beta_equal_gamma_adds_allocation_and_fft_log_powers() -> None:
    certificate = _certificate(0.75, 1.0, 1.0, log_power=1.0)

    assert certificate.regime == "beta_equal_gamma"
    assert certificate.epsilon_polynomial_exponent == pytest.approx(2.0)
    assert certificate.epsilon_log_power == pytest.approx(3.0)


def test_beta_less_gamma_matches_rough_terminal_algebra() -> None:
    certificate = _certificate(0.25, 0.5, 1.0, log_power=1.0)

    assert certificate.regime == "beta_less_gamma"
    assert certificate.compatibility_threshold == pytest.approx(0.25)
    assert certificate.epsilon_polynomial_exponent == pytest.approx(4.0)
    assert certificate.epsilon_log_power == pytest.approx(1.0)
    assert certificate.finest_sample_polynomial_exponent == pytest.approx(4.0)


def test_incompatible_rate_triplet_refuses_a_complexity_exponent() -> None:
    certificate = _certificate(0.2, 1.0, 2.0)

    assert not certificate.compatibility_condition_satisfied
    assert certificate.epsilon_polynomial_exponent is None
    assert certificate.epsilon_log_power is None
    assert not certificate.conditional_complexity_authorized
    assert not certificate.unconditional_complexity_authorized


@pytest.mark.parametrize(
    ("status", "expected_class", "conditional", "unconditional", "empirical"),
    [
        ("proved_external", "proved", True, True, False),
        ("conditional", "conditional", True, False, False),
        ("empirical", "empirical_only", False, False, True),
        ("open", "open", False, False, False),
    ],
)
def test_evidence_provenance_cannot_be_promoted(
    status: str,
    expected_class: str,
    conditional: bool,
    unconditional: bool,
    empirical: bool,
) -> None:
    certificate = _certificate(0.5, 0.75, 1.0, bias_evidence=status)

    assert certificate.evidence_class == expected_class
    assert certificate.conditional_complexity_authorized is conditional
    assert certificate.unconditional_complexity_authorized is unconditional
    assert certificate.empirical_only is empirical


def test_terminal_contract_cross_checks_existing_conservative_rate() -> None:
    existing = terminal_rate_contract(0.12, epsilon_margin=0.01)
    certificate = conservative_terminal_rbergomi_complexity(
        0.12,
        epsilon_margin=0.01,
    )

    assert certificate.weak_bias_exponent == pytest.approx(
        existing.weak_bias_exponent
    )
    assert certificate.correction_variance_exponent == pytest.approx(
        existing.dcs_second_moment_exponent
    )
    assert certificate.epsilon_polynomial_exponent == pytest.approx(
        existing.mlmc_epsilon_polynomial_exponent
    )
    assert certificate.epsilon_log_power == pytest.approx(1.0)
    assert certificate.evidence_class == "conditional"
    assert certificate.conditional_complexity_authorized
    assert not certificate.unconditional_complexity_authorized


def test_barrier_threshold_has_exact_three_term_signed_decomposition() -> None:
    log_barrier = math.log(0.5)
    fine_candidates = torch.tensor([[1.0, 2.0, 5.0, 4.0]], dtype=torch.float64)
    coarse_candidates = torch.tensor([[4.0, 3.0]], dtype=torch.float64)
    fine_intercept = torch.cat(
        (
            torch.zeros((1, 1), dtype=torch.float64),
            log_barrier - fine_candidates,
        ),
        dim=1,
    )
    coarse_intercept = torch.cat(
        (
            torch.zeros((1, 1), dtype=torch.float64),
            log_barrier - coarse_candidates,
        ),
        dim=1,
    )
    diagnostics = evaluate_rbergomi_threshold_coupling(
        fine_intercept,
        torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0]], dtype=torch.float64),
        coarse_intercept,
        torch.tensor([[0.0, 1.0, 1.0]], dtype=torch.float64),
        fine_step_dt=0.25,
        coarse_step_dt=0.5,
        task=DiscreteBarrierHitTask(0.5),
        denominator_floor=0.5,
    )

    assert diagnostics.fine_threshold.item() == pytest.approx(5.0)
    assert diagnostics.coarse_threshold.item() == pytest.approx(4.0)
    assert diagnostics.coarse_active_coefficient_defect.item() == pytest.approx(
        -2.0
    )
    assert diagnostics.common_active_switch_defect.item() == pytest.approx(2.0)
    assert diagnostics.mesh_enrichment_defect.item() == pytest.approx(1.0)
    assert diagnostics.signed_threshold_difference.item() == pytest.approx(1.0)
    assert diagnostics.maximum_signed_decomposition_residual == pytest.approx(0.0)
    assert diagnostics.maximum_exact_decomposition_violation == pytest.approx(0.0)


@pytest.mark.parametrize("hurst", [0.05, 0.12, 0.30])
def test_terminal_wrapper_never_promotes_the_unreviewed_rate(hurst: float) -> None:
    certificate = conservative_terminal_rbergomi_complexity(
        hurst,
        epsilon_margin=0.01,
    )

    assert certificate.evidence_class == "conditional"
    assert certificate.conditional_complexity_authorized
    assert not certificate.unconditional_complexity_authorized


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("weak_bias_exponent", 0.0, "strictly positive"),
        ("correction_variance_exponent", -1.0, "strictly positive"),
        ("sample_cost_exponent", float("inf"), "finite"),
        ("sample_cost_log_power", -1.0, "nonnegative"),
        ("weak_bias_evidence", "fitted_and_proved", "must be one of"),
    ],
)
def test_invalid_complexity_inputs_fail_closed(
    field: str,
    value: float | str,
    match: str,
) -> None:
    arguments: dict[str, object] = {
        "weak_bias_exponent": 0.5,
        "correction_variance_exponent": 0.75,
        "sample_cost_exponent": 1.0,
        "sample_cost_log_power": 0.0,
        "weak_bias_evidence": "proved_internal",
        "correction_variance_evidence": "proved_internal",
        "sample_cost_evidence": "proved_internal",
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=match):
        mlmc_complexity_certificate(**arguments)  # type: ignore[arg-type]
