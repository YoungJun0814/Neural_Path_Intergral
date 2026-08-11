"""Independent structural and arithmetic audit for V12 ECRPT results."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import torch

from src.path_integral.baseline_framework import BaselineCostLedger
from src.path_integral.ecrpt_protocol import aggregate_ecrpt_microstudy
from src.path_integral.residual_transport import (
    ResidualGaussianMixtureSpec,
    freeze_residual_transport,
)
from src.path_integral.seed_ledger import SeedLedger


def semantic_equal(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            semantic_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            semantic_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-11, abs_tol=1e-14)
    return left == right


def recompute_summary(summary: dict[str, Any]) -> dict[str, float | int]:
    """Reconstruct ordinary-mean moments from stored sufficient statistics."""

    count = int(summary["count"])
    total = float(summary["sum"])
    sum_squares = float(summary["sum_squares"])
    if count < 2 or not all(math.isfinite(x) for x in (total, sum_squares)):
        raise ValueError("invalid sufficient statistics")
    mean = total / count
    centered = sum_squares - count * mean * mean
    tolerance = 1e-10 * max(1.0, abs(sum_squares))
    if centered < -tolerance:
        raise ValueError("sufficient statistics imply negative variance")
    variance = max(0.0, centered) / (count - 1)
    return {
        "count": count,
        "estimate": mean,
        "variance": variance,
        "standard_error": math.sqrt(variance / count),
    }


def _summary_matches(summary: dict[str, Any]) -> bool:
    rebuilt = recompute_summary(summary)
    return all(
        semantic_equal(summary[key], value)
        for key, value in rebuilt.items()
    )


def _proposal_hash_matches(payload: dict[str, Any]) -> bool:
    spec = ResidualGaussianMixtureSpec(
        direction=torch.tensor(payload["direction"], dtype=torch.float64),
        means=torch.tensor(payload["component_means"], dtype=torch.float64),
        weights=torch.tensor(payload["component_weights"], dtype=torch.float64),
    )
    rebuilt = freeze_residual_transport(
        task_id=str(payload["task_id"]),
        spec=spec,
        training_seed=int(payload["training_seed"]),
        training_objective=str(payload["training_objective"]),
        training_cost=BaselineCostLedger(**payload["training_cost"]),
    )
    return rebuilt.sha256 == payload.get("sha256")


@dataclass(frozen=True)
class ECRPTMicrostudyAudit:
    passed: bool
    result_sha256: str
    config_hash_match: bool
    seed_ledger_match: bool
    roster_match: bool
    summaries_match: bool
    proposal_hashes_match: bool
    aggregate_match: bool
    decision_match: bool
    failures: tuple[str, ...]


def audit_ecrpt_microstudy(
    *,
    config: dict[str, Any],
    config_sha256: str,
    result: dict[str, Any],
    result_bytes: bytes | None = None,
) -> ECRPTMicrostudyAudit:
    """Audit without simulator replay; exact-path identities are result diagnostics.

    This audit deliberately does not turn development evidence into qualification
    evidence.  A later qualification run must additionally replay selected cells.
    """

    failures: list[str] = []
    config_match = result.get("config_sha256") == config_sha256
    if not config_match:
        failures.append("config_hash")
    try:
        ledger = SeedLedger.from_dict(result["seed_ledger"])
        seed_match = ledger.sha256 == result.get("seed_ledger_sha256")
    except (KeyError, TypeError, ValueError):
        seed_match = False
    if not seed_match:
        failures.append("seed_ledger")
    records = result.get("records")
    record_list: list[dict[str, Any]] = (
        records
        if isinstance(records, list)
        and all(isinstance(record, dict) for record in records)
        else []
    )
    expected = {
        (str(cell["cell_id"]), cluster)
        for cell in config["cells"]
        for cluster in range(int(config["clusters"]))
    }
    roster_match = bool(record_list) and {
        (str(record.get("cell_id")), int(record.get("cluster", -1)))
        for record in record_list
    } == expected and len(record_list) == len(expected)
    if not roster_match:
        failures.append("roster")
    summaries_match = roster_match
    proposal_match = roster_match
    if roster_match:
        for record in record_list:
            candidate = record["candidate"]
            summaries = [
                candidate["estimate"],
                candidate["raw_estimate"],
                candidate["difference"],
                candidate["likelihood"],
                *(
                    comparator["estimate"]
                    for comparator in record["comparators"].values()
                ),
            ]
            summaries_match = summaries_match and all(
                _summary_matches(summary) for summary in summaries
            )
            proposal_match = proposal_match and _proposal_hash_matches(
                candidate["proposal"]
            )
    if not summaries_match:
        failures.append("sufficient_statistics")
    if not proposal_match:
        failures.append("proposal_hash")
    try:
        rebuilt_aggregate = aggregate_ecrpt_microstudy(
            config=config, records=record_list
        )
        aggregate_match = semantic_equal(rebuilt_aggregate, result.get("aggregate"))
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        aggregate_match = False
    if not aggregate_match:
        failures.append("aggregate")
    aggregate = result.get("aggregate", {})
    expected_decision = {
        "software_correctness_evidence_pass": bool(aggregate.get("correctness_pass")),
        "development_performance_gate_pass": bool(aggregate.get("performance_pass")),
        "qualification_authorized": bool(aggregate.get("correctness_pass"))
        and bool(aggregate.get("performance_pass")),
        "broad_performance_claim_authorized": False,
        "top_journal_claim_authorized": False,
        "submission_authorized": False,
    }
    decision_match = semantic_equal(expected_decision, result.get("decision"))
    if not decision_match:
        failures.append("decision")
    digest = hashlib.sha256(result_bytes or b"").hexdigest()
    return ECRPTMicrostudyAudit(
        passed=not failures,
        result_sha256=digest,
        config_hash_match=config_match,
        seed_ledger_match=seed_match,
        roster_match=roster_match,
        summaries_match=summaries_match,
        proposal_hashes_match=proposal_match,
        aggregate_match=aggregate_match,
        decision_match=decision_match,
        failures=tuple(failures),
    )
