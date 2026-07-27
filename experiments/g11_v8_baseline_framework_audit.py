"""Fail-closed audit for the V8 P4 strong-baseline framework ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from src.path_integral.baseline_framework import BASELINE_METHODS

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SCHEMA = "npi.g11.v8-baseline-framework-ledger.v1"
REPORT_SCHEMA = "npi.g11.v8-baseline-framework-audit.v1"

_ROOT_KEYS = {
    "schema",
    "protocol_id",
    "date",
    "phase",
    "outcome_data_used",
    "rate_complexity_ledger_sha256",
    "framework_document",
    "framework_document_sha256",
    "lifecycle",
    "methods",
    "flow_constraint",
    "cost_categories",
    "evidence",
    "decision",
}
_FAMILIES = (
    "target_gaussian",
    "target_gaussian",
    "target_gaussian",
    "gaussian_shift",
    "gaussian_mixture_shift",
    "rqmc_target_gaussian",
    "gaussian_mixture_shift",
    "coupling_flow",
)
_UNITS = (
    "iid_path",
    "antithetic_pair",
    "iid_path",
    "iid_path",
    "iid_path",
    "rqmc_randomization",
    "iid_path",
    "iid_path",
)
_COSTS = [
    "training_samples",
    "optimizer_steps",
    "hyperparameter_trials",
    "failed_restarts",
    "screening_samples",
    "planning_samples",
    "final_samples",
    "likelihood_evaluations",
    "cdf_calls",
    "quadrature_calls",
    "algorithmic_work_units",
    "wall_seconds",
    "cpu_seconds",
    "gpu_seconds",
    "peak_memory_bytes",
    "compute_cost_usd",
    "energy_kwh",
    "measurement_mode",
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _load_ledger(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("baseline framework ledger must be a mapping")
    if payload.get("schema") != EXPECTED_SCHEMA:
        raise ValueError("unexpected baseline framework schema")
    return payload, hashlib.sha256(raw).hexdigest()


def audit_baseline_framework(
    ledger: dict[str, Any],
    digest: str,
) -> dict[str, Any]:
    document_value = ledger.get("framework_document")
    document = ROOT / document_value if isinstance(document_value, str) else ROOT
    upstream = ROOT / "configs/g11_v8/rate_complexity_ledger_v1.yaml"
    lifecycle = _mapping(ledger.get("lifecycle"))
    flow = _mapping(ledger.get("flow_constraint"))
    evidence = _mapping(ledger.get("evidence"))
    decision = _mapping(ledger.get("decision"))
    methods = ledger.get("methods") if isinstance(ledger.get("methods"), list) else []
    method_maps = [item for item in methods if isinstance(item, dict)]
    ids = [item.get("id") for item in method_maps]
    families = [item.get("family") for item in method_maps]
    units = [item.get("inferential_unit") for item in method_maps]
    paths: list[str] = []
    for group in ("code", "tests"):
        values = evidence.get(group)
        if isinstance(values, list):
            paths.extend(item for item in values if isinstance(item, str))
    document_text = document.read_text(encoding="utf-8") if document.is_file() else ""

    checks = {
        "schema_exact": ledger.get("schema") == EXPECTED_SCHEMA,
        "root_keys_exact": set(ledger) == _ROOT_KEYS,
        "protocol_exact": ledger.get("protocol_id")
        == "g11-v8-baseline-framework-ledger-v1",
        "date_exact": ledger.get("date") == "2026-07-25",
        "phase_exact": ledger.get("phase") == "p4_development",
        "outcome_blind": ledger.get("outcome_data_used") is False,
        "upstream_exists": upstream.is_file(),
        "upstream_hash_bound": upstream.is_file()
        and ledger.get("rate_complexity_ledger_sha256") == _sha(upstream),
        "document_exists": document.is_file(),
        "document_hash_bound": document.is_file()
        and ledger.get("framework_document_sha256") == _sha(document),
        "document_boundaries_explicit": all(
            fragment in document_text
            for fragment in (
                "antithetic pair mean",
                "independent randomization",
                "Clipping and self-normalization are prohibited",
                "baseline_only",
                "no performance",
            )
        ),
        "lifecycle_stages_exact": lifecycle.get("stages")
        == ["train", "plan", "estimate", "audit"],
        "proposal_frozen_before_pilot": lifecycle.get(
            "proposal_frozen_before_pilot"
        )
        is True,
        "allocation_frozen_before_final": lifecycle.get(
            "integer_allocation_frozen_before_final"
        )
        is True,
        "seeds_disjoint": lifecycle.get("training_pilot_final_seeds_disjoint")
        is True,
        "self_normalization_refused": lifecycle.get("self_normalization_allowed")
        is False,
        "clipping_refused": lifecycle.get("likelihood_clipping_allowed") is False,
        "method_count_exact": len(method_maps) == len(BASELINE_METHODS),
        "method_ids_exact": ids == list(BASELINE_METHODS),
        "method_ids_unique": len(set(ids)) == len(ids),
        "families_exact": families == list(_FAMILIES),
        "inferential_units_exact": units == list(_UNITS),
        "fresh_trained_methods_exact": [
            item.get("id")
            for item in method_maps
            if item.get("training") == "fresh_task_tuned"
        ]
        == ["pure_cem", "defensive_cem", "ld_subspace_is", "flow_is"],
        "flow_nonlinear_bounded": flow.get(
            "bounded_nonlinear_triangular_coupling"
        )
        is True,
        "flow_inverse_exact": flow.get("exact_inverse") is True,
        "flow_jacobian_exact": flow.get("exact_log_jacobian") is True,
        "flow_baseline_only": flow.get("role") == "baseline_only",
        "flow_dcs_claim_refused": flow.get("dcs_extension_claim_authorized")
        is False,
        "cost_categories_exact": ledger.get("cost_categories") == _COSTS,
        "evidence_paths_relative": bool(paths)
        and all(
            not Path(path).is_absolute() and ".." not in Path(path).parts
            for path in paths
        ),
        "evidence_paths_exist": all((ROOT / path).is_file() for path in paths),
        "framework_oracle_pass": decision.get("status") == "framework_oracle_pass",
        "all_likelihood_oracles_required": decision.get(
            "all_method_likelihood_oracles_required"
        )
        is True,
        "all_cost_oracles_required": decision.get(
            "all_method_cost_oracles_required"
        )
        is True,
        "performance_not_qualified": decision.get("numerical_performance_qualified")
        is False,
        "fresh_ld_flow_runs_open": decision.get(
            "fresh_ld_and_flow_training_runs_complete"
        )
        is False,
        "p5_authorized": decision.get("p5_reference_matrix_authorized") is True,
        "superiority_refused": decision.get("superiority_claim_authorized") is False,
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "schema": REPORT_SCHEMA,
        "ledger_schema": ledger.get("schema"),
        "ledger_sha256": digest,
        "check_count": len(checks),
        "checks": checks,
        "failure_count": len(failures),
        "failures": failures,
        "method_count": len(method_maps),
        "evidence_path_count": len(paths),
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
    report = audit_baseline_framework(ledger, digest)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
