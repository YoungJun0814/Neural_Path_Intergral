"""Independent structural and arithmetic audit for V14 local-Volterra results."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, cast

from src.path_integral.baseline_framework import (
    BaselineCostLedger,
    BaselineMethod,
    freeze_baseline_proposal,
)
from src.path_integral.ecrpt_microstudy_audit import recompute_summary, semantic_equal
from src.path_integral.ecrpt_protocol import aggregate_local_volterra_development
from src.path_integral.seed_ledger import SeedLedger
from src.path_integral.v13_development_audit import (
    _smc_contract_matches,
    _tail_certificate_matches,
)


def _summary_matches(summary: dict[str, Any]) -> bool:
    rebuilt = recompute_summary(summary)
    return all(semantic_equal(summary[key], value) for key, value in rebuilt.items())


def proposal_hash_matches(payload: dict[str, Any]) -> bool:
    proposal = freeze_baseline_proposal(
        method=cast(BaselineMethod, str(payload["method"])),
        task_id=str(payload["task_id"]),
        dimension=int(payload["dimension"]),
        training_seed=int(payload["training_seed"]),
        training_cost=BaselineCostLedger(**payload["training_cost"]),
        training_budget_work_units=float(payload["training_budget_work_units"]),
        location=tuple(float(x) for x in payload["location"]),
        component_means=tuple(tuple(float(x) for x in row) for row in payload["component_means"]),
        component_weights=tuple(float(x) for x in payload["component_weights"]),
        flow_split=int(payload["flow_split"]),
        flow_scale_matrix=tuple(
            tuple(float(x) for x in row) for row in payload["flow_scale_matrix"]
        ),
        flow_scale_bias=tuple(float(x) for x in payload["flow_scale_bias"]),
        flow_shift_matrix=tuple(
            tuple(float(x) for x in row) for row in payload["flow_shift_matrix"]
        ),
        flow_shift_bias=tuple(float(x) for x in payload["flow_shift_bias"]),
        flow_max_log_scale=float(payload["flow_max_log_scale"]),
        conditional_integral=str(payload["conditional_integral"]),
    )
    return proposal.sha256 == payload.get("sha256")


@dataclass(frozen=True)
class V14LocalVolterraAudit:
    passed: bool
    result_sha256: str
    config_hash_match: bool
    seed_ledger_match: bool
    roster_match: bool
    summaries_match: bool
    proposal_hashes_match: bool
    smc_contracts_match: bool
    tail_certificates_match: bool
    aggregate_match: bool
    decision_match: bool
    failures: tuple[str, ...]


def audit_v14_local_volterra(
    *,
    config: dict[str, Any],
    config_sha256: str,
    result: dict[str, Any],
    result_bytes: bytes | None = None,
    expected_stage: str = "development",
) -> V14LocalVolterraAudit:
    failures: list[str] = []
    config_match = (
        result.get("config_sha256") == config_sha256 and result.get("stage") == expected_stage
    )
    if not config_match:
        failures.append("config_or_stage")
    try:
        ledger = SeedLedger.from_dict(result["seed_ledger"])
        seed_match = ledger.sha256 == result.get("seed_ledger_sha256")
    except (KeyError, TypeError, ValueError):
        seed_match = False
    if not seed_match:
        failures.append("seed_ledger")
    records = result.get("records")
    record_list = records if isinstance(records, list) else []
    expected = {
        (str(cell["cell_id"]), cluster)
        for cell in config["cells"]
        for cluster in range(int(config["clusters"]))
    }
    roster = (
        len(record_list) == len(expected)
        and {
            (str(record.get("cell_id")), int(record.get("cluster", -1)))
            for record in record_list
            if isinstance(record, dict)
        }
        == expected
    )
    if not roster:
        failures.append("roster")
    summaries = proposal_hashes = smc_contracts = tails = roster
    if roster:
        for record in record_list:
            candidate = record["candidate"]
            stored = [
                candidate["estimate"],
                candidate["raw_estimate"],
                candidate["difference"],
                candidate["likelihood"],
                *(item["estimate"] for item in record["comparators"].values()),
            ]
            summaries = summaries and all(_summary_matches(item) for item in stored)
            proposal_hashes = proposal_hashes and proposal_hash_matches(candidate["proposal"])
            smc_contracts = smc_contracts and all(
                _smc_contract_matches(item) for item in candidate["smc"]
            )
            tails = tails and _tail_certificate_matches(
                candidate["tail_safe_forecast"], candidate["estimate"]
            )
            tails = tails and all(
                _tail_certificate_matches(item["tail_safe_forecast"], item["estimate"])
                for item in record["comparators"].values()
            )
    for name, valid in (
        ("sufficient_statistics", summaries),
        ("proposal_hash", proposal_hashes),
        ("smc_contract", smc_contracts),
        ("tail_certificate", tails),
    ):
        if not valid:
            failures.append(name)
    try:
        rebuilt = aggregate_local_volterra_development(config=config, records=record_list)
        aggregate_match = semantic_equal(rebuilt, result.get("aggregate"))
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        aggregate_match = False
    if not aggregate_match:
        failures.append("aggregate")
    aggregate = result.get("aggregate", {})
    expected_decision = {
        "software_correctness_evidence_pass": bool(aggregate.get("correctness_pass")),
        "development_performance_gate_pass": bool(aggregate.get("performance_pass")),
        "qualification_authorized": bool(aggregate.get("stage_pass"))
        if expected_stage == "development"
        else False,
        "qualification_executed": expected_stage == "qualification",
        "distribution_free_tail_claim_authorized": bool(
            aggregate.get("distribution_free_tail_certificate_pass")
        ),
        "broad_performance_claim_authorized": False,
        "top_journal_claim_authorized": False,
        "submission_authorized": False,
    }
    decision_match = semantic_equal(expected_decision, result.get("decision"))
    if not decision_match:
        failures.append("decision")
    return V14LocalVolterraAudit(
        passed=not failures,
        result_sha256=hashlib.sha256(result_bytes or b"").hexdigest(),
        config_hash_match=config_match,
        seed_ledger_match=seed_match,
        roster_match=roster,
        summaries_match=summaries,
        proposal_hashes_match=proposal_hashes,
        smc_contracts_match=smc_contracts,
        tail_certificates_match=tails,
        aggregate_match=aggregate_match,
        decision_match=decision_match,
        failures=tuple(failures),
    )
