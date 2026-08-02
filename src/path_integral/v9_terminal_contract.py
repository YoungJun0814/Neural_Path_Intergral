"""Semantic audit for the frozen V9 terminal-only claim contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class V9ContractAudit:
    checks: tuple[tuple[str, bool], ...]
    failures: tuple[str, ...]
    passed: bool


def audit_v9_contract(path: Path) -> V9ContractAudit:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("V9 claim contract must be a mapping")
    cells = payload.get("cells")
    if not isinstance(cells, list):
        raise ValueError("V9 claim contract cells must be a list")
    identifiers = [str(cell.get("cell_id")) for cell in cells]
    hursts = {float(cell.get("hurst")) for cell in cells}
    probabilities = {float(cell.get("nominal_probability")) for cell in cells}
    tasks = {str(cell.get("task")) for cell in cells}
    bindings = str(payload.get("artifact_policy", {}))
    claim = payload.get("claim", {})
    gate = payload.get("gate", {})
    checks = (
        ("schema", payload.get("schema") == "npi.g11.v9-terminal-claim-contract.v1"),
        ("v8_disclosed_as_design_data", payload.get("design_status") == "v8-informed-new-development"),
        ("terminal_only", tasks == {"terminal_left_tail"}),
        ("complete_roster", len(cells) == 12 and len(set(identifiers)) == 12),
        ("hurst_roster", hursts == {0.05, 0.12, 0.2}),
        ("rarity_roster", probabilities == {1e-2, 1e-3, 1e-4, 1e-5}),
        ("finite_grid_estimand", claim.get("estimand") == "finite_grid_terminal_probability"),
        ("ordinary_mean", claim.get("self_normalization_allowed") is False),
        ("exact_likelihood", claim.get("exact_likelihood_required") is True),
        ("training_inclusive", claim.get("training_inclusive_total_work") is True),
        ("primary_k_frozen", payload.get("primary_query_count") == 100),
        ("cluster_inference", claim.get("inferential_unit") == "independent_cluster"),
        ("no_barrier_rate_claim", claim.get("barrier_claim_allowed") is False),
        ("no_uniform_superiority_claim", claim.get("uniform_superiority_claim_allowed") is False),
        ("no_v8_confirmation_binding", "results/g11_v8" not in bindings),
        ("fresh_reference_required", payload.get("artifact_policy", {}).get("fresh_v9_reference") is True),
        ("fresh_seeds_required", payload.get("artifact_policy", {}).get("fresh_v9_seed_namespaces") is True),
        ("correctness_gate", float(gate.get("maximum_exactness_error", 1.0)) <= 1e-10),
        ("submission_locked", claim.get("submission_authorized") is False),
    )
    failures = tuple(name for name, passed in checks if not passed)
    return V9ContractAudit(checks=checks, failures=failures, passed=not failures)
