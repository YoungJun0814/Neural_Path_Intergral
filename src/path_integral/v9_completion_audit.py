"""Independent final closure audit for the gated V9 terminal program."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class V9CompletionAudit:
    checks: tuple[tuple[str, bool], ...]
    failures: tuple[str, ...]
    scientific_status: str
    passed: bool


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_v9_completion(*, ledger_path: Path, root: Path) -> V9CompletionAudit:
    ledger = yaml.safe_load(ledger_path.read_text(encoding="utf-8"))
    if not isinstance(ledger, dict):
        raise ValueError("V9 completion ledger must be a mapping")
    artifacts: dict[str, dict[str, Any]] = {}
    bindings = True
    for name, binding in ledger["bindings"].items():
        path = root / str(binding["path"])
        if not path.is_file() or _sha256(path) != str(binding["sha256"]):
            bindings = False
            continue
        if path.suffix == ".json":
            artifacts[name] = json.loads(path.read_text(encoding="utf-8"))
    expected = ledger["expected"]
    claim_audit = artifacts.get("claim_contract_audit", {})
    reference_v1 = artifacts.get("reference_v1", {})
    reference_v1_audit = artifacts.get("reference_v1_audit", {})
    reference_v2 = artifacts.get("reference_v2", {})
    reference_v2_audit = artifacts.get("reference_v2_audit", {})
    bank = artifacts.get("proposal_bank", {})
    bank_audit = artifacts.get("proposal_bank_audit", {})
    development = artifacts.get("development", {})
    development_audit = artifacts.get("development_audit", {})
    aggregate = development.get("aggregate", {})
    groups = {float(item["hurst"]): item for item in aggregate.get("group_summaries", [])}
    forbidden_absent = all(not (root / str(path)).exists() for path in ledger["forbidden_outputs"])
    decisions = development.get("decision", {})
    numeric = (
        math.isclose(
            float(aggregate.get("maximum_exactness_error", math.inf)),
            float(expected["maximum_exactness_error"]),
            rel_tol=1e-12,
        )
        and math.isclose(
            float(aggregate.get("dcs_accuracy_maximum_combined_z", math.inf)),
            float(expected["maximum_dcs_combined_z"]),
            rel_tol=1e-12,
        )
        and math.isclose(
            float(aggregate.get("external_accuracy_maximum_combined_z", math.inf)),
            float(expected["maximum_external_combined_z"]),
            rel_tol=1e-12,
        )
        and math.isclose(
            float(aggregate.get("paired_difference_maximum_z", math.inf)),
            float(expected["maximum_paired_difference_z"]),
            rel_tol=1e-12,
        )
    )
    group_ratios = len(groups) == 3 and all(
        math.isclose(
            float(groups[hurst]["dcs_vs_raw"]["geometric_ratio"]),
            float(expected[f"h{hurst:.2f}_raw_geometric_ratio"]),
            rel_tol=1e-12,
        )
        for hurst in (0.05, 0.12, 0.2)
    )
    checks = (
        ("schema", ledger.get("schema") == "npi.g11.v9-completion-status-ledger.v1"),
        ("binding_hashes", bindings),
        ("claim_contract_audit", claim_audit.get("passed") is True),
        (
            "reference_v1_preserved_failure",
            reference_v1.get("decision", {}).get("reference_complete")
            is expected["reference_v1_complete"]
            and reference_v1_audit.get("passed") is True,
        ),
        (
            "reference_v2_pass",
            reference_v2.get("decision", {}).get("reference_complete")
            is expected["reference_v2_complete"]
            and reference_v2_audit.get("passed") is True,
        ),
        (
            "proposal_bank",
            bank.get("entry_count") == expected["proposal_bank_entries"]
            and bank.get("seed_count") == expected["proposal_bank_seeds"]
            and bank_audit.get("passed") is True,
        ),
        (
            "development_audit",
            development_audit.get("passed") is True
            and aggregate.get("paired_record_count")
            == expected["development_paired_records"]
            and aggregate.get("external_record_count")
            == expected["development_external_records"]
            and development.get("seed_count") == expected["development_seeds"],
        ),
        ("numeric_reconstruction", numeric),
        ("mechanism_group_ratios", group_ratios),
        (
            "falsification_decision",
            aggregate.get("stage_pass") is expected["development_stage_pass"]
            and aggregate.get("selected_hurst_groups") == expected["selected_hurst_groups"]
            and aggregate.get("blockers") == expected["blockers"]
            and aggregate.get("primary_resource_censoring_count")
            == expected["resource_censoring_count"],
        ),
        (
            "claim_locks",
            decisions.get("qualification_authorized")
            is expected["qualification_authorized"]
            and decisions.get("regime_conditional_empirical_claim_authorized")
            is expected["regime_conditional_empirical_claim_authorized"]
            and decisions.get("top_journal_claim_authorized")
            is expected["top_journal_claim_authorized"]
            and decisions.get("submission_authorized")
            is expected["submission_authorized"],
        ),
        ("forbidden_downstream_outputs_absent", forbidden_absent),
    )
    failures = tuple(name for name, passed in checks if not passed)
    return V9CompletionAudit(
        checks=checks,
        failures=failures,
        scientific_status="closed_by_development_falsification",
        passed=not failures,
    )
