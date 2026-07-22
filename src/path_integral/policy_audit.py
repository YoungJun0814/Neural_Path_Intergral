"""Independent arithmetic audit for V6 policy preparations and results."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

from src.path_integral.policy_allocation import V6PolicyPreparedRun, V6PolicyResult
from src.path_integral.seed_ledger import SeedLedger


@dataclass(frozen=True)
class V6PolicyAudit:
    policy_hash_valid: bool
    result_hash_valid: bool
    preparation_hash_matches: bool
    estimate_recomputed: bool
    variance_recomputed: bool
    allocation_counts_match: bool
    work_recomputed: bool
    seed_ledger_valid: bool
    final_seeds_absent_from_preparation: bool
    new_seed_roles_are_final: bool
    passed: bool


def _hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def audit_v6_policy(
    prepared: V6PolicyPreparedRun,
    result: V6PolicyResult,
) -> V6PolicyAudit:
    """Recompute invariants without calling preparation or execution helpers."""

    policy_payload: dict[str, object] = {
        "schema": "npi.g11.v6-policy-preparation.v1",
        "policy_name": prepared.policy_name,
        "cell_id": prepared.cell_id,
        "execution_method": prepared.execution_method,
        "route": None if prepared.route is None else asdict(prepared.route),
        "core_preparation_hash": prepared.core.preparation_hash,
        "preprocessing_work_sha256": prepared.preprocessing_work.sha256,
    }
    policy_hash_valid = _hash(policy_payload) == prepared.policy_hash == result.policy_hash
    result_payload: dict[str, object] = {
        "schema": "npi.g11.v6-policy-result.v1",
        "policy_hash": result.policy_hash,
        "core_preparation_hash": result.core.preparation_hash,
        "complete": result.core.complete,
        "resource_censored": result.core.resource_censored,
        "estimate": result.core.estimate,
        "empirical_sampling_variance": result.core.empirical_sampling_variance,
        "seed_ledger_hash": result.core.seed_ledger_hash,
        "total_work_sha256": result.total_work.sha256,
    }
    result_hash_valid = _hash(result_payload) == result.result_hash
    preparation_hash_matches = result.core.preparation_hash == prepared.core.preparation_hash

    if result.core.complete:
        estimate_recomputed = result.core.estimate is not None and math.isclose(
            result.core.estimate,
            math.fsum(term.mean for term in result.core.terms),
            rel_tol=1e-15,
            abs_tol=1e-15,
        )
        variance_recomputed = result.core.empirical_sampling_variance is not None and math.isclose(
            result.core.empirical_sampling_variance,
            math.fsum(term.variance / term.count for term in result.core.terms),
            rel_tol=1e-15,
            abs_tol=1e-15,
        )
        allocation_counts_match = tuple(term.count for term in result.core.terms) == tuple(
            allocation.final_count for allocation in result.core.allocations
        )
    else:
        estimate_recomputed = result.core.estimate is None
        variance_recomputed = result.core.empirical_sampling_variance is None
        allocation_counts_match = not result.core.terms

    work_recomputed = math.isclose(
        result.total_work.total_work_units,
        result.core.work.total_work_units,
        rel_tol=1e-15,
        abs_tol=1e-12,
    )
    seed_ledger_valid = False
    final_seeds_absent = False
    new_seed_roles_are_final = False
    try:
        final_ledger = SeedLedger.from_dict(result.core.seed_ledger_payload)
        seed_ledger_valid = final_ledger.sha256 == result.core.seed_ledger_hash
        final_seeds_absent = all(
            not record.key.role.startswith("final") for record in prepared.core.ledger.records
        )
        prepared_keys = {record.key for record in prepared.core.ledger.records}
        final_keys = {record.key for record in final_ledger.records}
        seed_ledger_valid = seed_ledger_valid and prepared_keys.issubset(final_keys)
        new_seed_roles_are_final = all(
            record.key.role == "final"
            for record in final_ledger.records
            if record.key not in prepared_keys
        )
    except (KeyError, TypeError, ValueError):
        pass
    checks = (
        policy_hash_valid,
        result_hash_valid,
        preparation_hash_matches,
        estimate_recomputed,
        variance_recomputed,
        allocation_counts_match,
        work_recomputed,
        seed_ledger_valid,
        final_seeds_absent,
        new_seed_roles_are_final,
    )
    return V6PolicyAudit(
        policy_hash_valid=policy_hash_valid,
        result_hash_valid=result_hash_valid,
        preparation_hash_matches=preparation_hash_matches,
        estimate_recomputed=estimate_recomputed,
        variance_recomputed=variance_recomputed,
        allocation_counts_match=allocation_counts_match,
        work_recomputed=work_recomputed,
        seed_ledger_valid=seed_ledger_valid,
        final_seeds_absent_from_preparation=final_seeds_absent,
        new_seed_roles_are_final=new_seed_roles_are_final,
        passed=all(checks),
    )
