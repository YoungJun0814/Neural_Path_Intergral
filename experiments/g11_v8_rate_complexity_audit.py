"""Fail-closed audit for the G11 V8 P3 rate and complexity ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from src.path_integral.mlmc_complexity import (
    conservative_terminal_rbergomi_complexity,
)

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SCHEMA = "npi.g11.v8-rate-complexity-ledger.v1"
REPORT_SCHEMA = "npi.g11.v8-rate-complexity-ledger-audit.v1"

_ROOT_KEYS = {
    "schema",
    "protocol_id",
    "date",
    "phase",
    "outcome_data_used",
    "theorem_ledger_sha256",
    "proof_document",
    "proof_document_sha256",
    "finite_grid_decomposition",
    "terminal_rate_chain",
    "barrier_rate",
    "evidence_policy",
    "external_proof_obligations",
    "claim_levels",
    "prohibited_claims",
    "decision",
}
_DECOMPOSITION_TERMS = [
    "coarse_active_coefficient_defect",
    "common_active_switch_defect",
    "mesh_enrichment_defect",
]
_BARRIER_OBLIGATIONS = [
    "coarse_active_coefficient_rate",
    "early_active_time_small_slope_control",
    "common_active_index_switch_rate",
    "fine_only_mesh_enrichment_rate",
    "continuous_monitoring_crossing_bias_if_claimed",
]
_EXTERNAL_OBLIGATIONS = [
    "common_probability_space_and_filtration",
    "implementation_specific_BLP_volterra_estimate",
    "lognormal_moments_and_BDG_price_transfer",
    "measurable_affine_coefficient_decomposition",
    "continuous_terminal_limit_with_common_L2_direction",
]
_PROHIBITED = [
    "unconditional_terminal_weak_bias",
    "unconditional_terminal_mlmc_complexity",
    "barrier_rate_inherited_from_terminal",
    "continuous_barrier_exactness",
    "empirical_slope_proves_asymptotic_rate",
    "rough_regime_epsilon_minus_two_complexity",
    "training_inclusive_superiority_from_rate_algebra",
]
_PROOF_FRAGMENTS = (
    "D_{\\mathrm{coefficient}}",
    "D_{\\mathrm{active}}",
    "D_{\\mathrm{mesh}}",
    "\\alpha=r",
    "\\beta=2r",
    "\\gamma=1",
    "\\kappa=1",
    "\\epsilon^{-1/r}\\log(\\epsilon^{-1})",
    "empirical",
    "barrier remains a finite-grid experiment",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_ledger(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("rate-complexity ledger must be a mapping")
    if payload.get("schema") != EXPECTED_SCHEMA:
        raise ValueError("unexpected rate-complexity ledger schema")
    return payload, hashlib.sha256(raw).hexdigest()


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _paths(value: object) -> list[str]:
    return (
        [item for item in value if isinstance(item, str)]
        if isinstance(value, list)
        else []
    )


def audit_rate_complexity_ledger(
    ledger: dict[str, Any],
    ledger_sha256: str,
) -> dict[str, Any]:
    """Return an exhaustive check map without trusting ledger claim labels."""

    proof_relative = ledger.get("proof_document")
    proof_path = ROOT / proof_relative if isinstance(proof_relative, str) else ROOT
    theorem_path = ROOT / "configs/g11_v8/theorem_ledger_v1.yaml"
    proof_text = (
        proof_path.read_text(encoding="utf-8")
        if proof_path.is_file()
        else ""
    )
    decomposition = _mapping(ledger.get("finite_grid_decomposition"))
    terminal = _mapping(ledger.get("terminal_rate_chain"))
    weak_bias = _mapping(terminal.get("weak_bias"))
    variance = _mapping(terminal.get("correction_variance"))
    cost = _mapping(terminal.get("sample_cost"))
    complexity = _mapping(terminal.get("conditional_complexity"))
    barrier = _mapping(ledger.get("barrier_rate"))
    evidence = _mapping(ledger.get("evidence_policy"))
    claims = _mapping(ledger.get("claim_levels"))
    decision = _mapping(ledger.get("decision"))

    evidence_paths = (
        _paths(decomposition.get("code_evidence"))
        + _paths(decomposition.get("test_evidence"))
        + _paths(terminal.get("code_evidence"))
        + _paths(terminal.get("test_evidence"))
    )
    all_paths_well_formed = bool(evidence_paths) and all(
        not Path(item).is_absolute() and ".." not in Path(item).parts
        for item in evidence_paths
    )
    all_paths_exist = all((ROOT / item).is_file() for item in evidence_paths)

    exemplar_checks = []
    for hurst in (0.05, 0.12, 0.30):
        certificate = conservative_terminal_rbergomi_complexity(
            hurst,
            epsilon_margin=0.01,
        )
        rate = hurst - 0.01
        exemplar_checks.append(
            certificate.regime == "beta_less_gamma"
            and certificate.weak_bias_exponent == rate
            and certificate.correction_variance_exponent == 2.0 * rate
            and certificate.epsilon_polynomial_exponent is not None
            and abs(certificate.epsilon_polynomial_exponent - 1.0 / rate)
            <= 1e-12
            and certificate.epsilon_log_power == 1.0
            and certificate.evidence_class == "conditional"
            and certificate.conditional_complexity_authorized
            and not certificate.unconditional_complexity_authorized
        )

    checks = {
        "schema_exact": ledger.get("schema") == EXPECTED_SCHEMA,
        "root_keys_exact": set(ledger) == _ROOT_KEYS,
        "protocol_exact": ledger.get("protocol_id")
        == "g11-v8-rate-complexity-ledger-v1",
        "date_exact": ledger.get("date") == "2026-07-25",
        "phase_exact": ledger.get("phase") == "p3_development",
        "outcome_blind": ledger.get("outcome_data_used") is False,
        "theorem_ledger_exists": theorem_path.is_file(),
        "theorem_ledger_hash_bound": theorem_path.is_file()
        and ledger.get("theorem_ledger_sha256") == _sha256(theorem_path),
        "proof_document_exists": proof_path.is_file(),
        "proof_document_hash_bound": proof_path.is_file()
        and ledger.get("proof_document_sha256") == _sha256(proof_path),
        "proof_fragments_complete": all(
            fragment in proof_text for fragment in _PROOF_FRAGMENTS
        ),
        "decomposition_status_exact": decomposition.get("status")
        == "proved_pathwise",
        "decomposition_tasks_exact": decomposition.get("task_scope")
        == ["terminal_threshold", "discrete_barrier_hit"],
        "decomposition_terms_exact": decomposition.get("signed_terms")
        == _DECOMPOSITION_TERMS,
        "initial_hit_convention_exact": decomposition.get(
            "initially_hit_convention"
        )
        == "all_terms_zero_and_threshold_positive_infinity",
        "terminal_status_conditional": terminal.get("status")
        == "conditional_pending_external_proof_review",
        "terminal_rate_is_strictly_below_H": terminal.get("public_rate")
        == "r_equals_H_minus_epsilon"
        and "0_less_epsilon_less_H" in terminal.get("constraints", []),
        "fixed_cost_dimension_explicit": (
            "fixed_finite_mixture_control_bank_and_integration_rank"
            in terminal.get("constraints", [])
        ),
        "alpha_conditional": weak_bias
        == {"symbol": "alpha", "value": "r", "evidence": "conditional"},
        "beta_conditional": variance
        == {"symbol": "beta", "value": "2r", "evidence": "conditional"},
        "fft_cost_exact": cost
        == {
            "symbol": "gamma",
            "value": 1,
            "log_power": 1,
            "evidence": "proved_internal",
        },
        "compatibility_condition_explicit": terminal.get(
            "compatibility_condition"
        )
        == "alpha_at_least_half_min_beta_gamma",
        "rough_regime_exact": terminal.get("regime") == "beta_less_gamma",
        "conditional_complexity_exact": complexity
        == {
            "epsilon_polynomial_exponent": "1_over_r",
            "epsilon_log_power": 1,
        },
        "terminal_unconditional_claim_refused": terminal.get(
            "unconditional_claim_authorized"
        )
        is False,
        "executable_algebra_exact": all(exemplar_checks),
        "barrier_scope_downgraded": barrier.get("status")
        == "open_finite_grid_experiment_only",
        "barrier_identity_retained": barrier.get(
            "exact_finite_grid_identity_available"
        )
        is True,
        "barrier_rates_open": barrier.get("weak_bias_exponent") == "open"
        and barrier.get("correction_variance_exponent") == "open",
        "barrier_complexity_refused": barrier.get(
            "complexity_claim_authorized"
        )
        is False,
        "barrier_obligations_exact": barrier.get("unresolved_obligations")
        == _BARRIER_OBLIGATIONS,
        "evidence_statuses_exact": evidence.get("statuses")
        == [
            "proved_internal",
            "proved_external",
            "conditional",
            "empirical",
            "open",
        ],
        "empirical_not_proof": evidence.get("empirical_slopes_are_proof")
        is False,
        "incompatible_triplet_refused": evidence.get(
            "missing_compatibility_returns_exponent"
        )
        is False,
        "rate_promotion_requires_bound_review": evidence.get(
            "rate_promotion_requires_new_bound_review_artifact"
        )
        is True,
        "external_obligations_exact": ledger.get("external_proof_obligations")
        == _EXTERNAL_OBLIGATIONS,
        "claim_levels_exact": claims
        == {
            "C4_terminal_model_rate": "conditional",
            "C4_barrier_model_rate": "open",
            "C5_terminal_sampling_complexity": "conditional",
            "C5_barrier_sampling_complexity": "prohibited",
            "C5_training_inclusive_end_to_end_complexity": "open",
            "training_inclusive_superiority": "open_experimental",
        },
        "prohibited_claims_exact": ledger.get("prohibited_claims")
        == _PROHIBITED,
        "decision_conditional_scope_downgrade": decision.get("status")
        == "conditional_pass_with_scope_downgrade",
        "p4_only_authorized": decision.get("p4_baseline_framework_authorized")
        is True,
        "terminal_theory_not_complete": decision.get(
            "terminal_math_theory_complete"
        )
        is False,
        "barrier_theory_not_complete": decision.get(
            "barrier_rate_theory_complete"
        )
        is False,
        "submission_complexity_not_authorized": decision.get(
            "submission_complexity_claim_authorized"
        )
        is False,
        "external_review_required": decision.get(
            "external_proof_review_required"
        )
        is True,
        "evidence_paths_well_formed": all_paths_well_formed,
        "all_evidence_paths_exist": all_paths_exist,
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "schema": REPORT_SCHEMA,
        "ledger_schema": ledger.get("schema"),
        "ledger_sha256": ledger_sha256,
        "check_count": len(checks),
        "checks": checks,
        "failure_count": len(failures),
        "failures": failures,
        "evidence_path_count": len(evidence_paths),
        "passed": not failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {arguments.output}")
    ledger, digest = _load_ledger(arguments.ledger)
    report = audit_rate_complexity_ledger(ledger, digest)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
