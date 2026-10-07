"""Independent arithmetic and structural audit for V13 development artifacts."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any, cast

import torch

from src.path_integral.baseline_framework import BaselineCostLedger
from src.path_integral.ecrpt_microstudy_audit import recompute_summary, semantic_equal
from src.path_integral.ecrpt_protocol import aggregate_structured_ecrpt_development
from src.path_integral.low_rank_residual_flow import (
    FrozenLowRankCouplingLayer,
    PartitionStyle,
    freeze_low_rank_residual_flow,
)
from src.path_integral.seed_ledger import SeedLedger
from src.path_integral.tail_safe_allocation import (
    BoundedRange,
    StreamingMoments,
    empirical_bernstein_variance_certificate,
)


def _summary_matches(summary: dict[str, Any]) -> bool:
    rebuilt = recompute_summary(summary)
    return all(semantic_equal(summary[key], value) for key, value in rebuilt.items())


def _layer(payload: dict[str, Any]) -> FrozenLowRankCouplingLayer:
    return FrozenLowRankCouplingLayer(
        active_indices=tuple(int(x) for x in payload["active_indices"]),
        transformed_indices=tuple(int(x) for x in payload["transformed_indices"]),
        scale_left=tuple(tuple(float(x) for x in row) for row in payload["scale_left"]),
        scale_right=tuple(tuple(float(x) for x in row) for row in payload["scale_right"]),
        scale_bias=tuple(float(x) for x in payload["scale_bias"]),
        shift_left=tuple(tuple(float(x) for x in row) for row in payload["shift_left"]),
        shift_right=tuple(tuple(float(x) for x in row) for row in payload["shift_right"]),
        shift_bias=tuple(float(x) for x in payload["shift_bias"]),
    )


def proposal_hash_matches(payload: dict[str, Any]) -> bool:
    rebuilt = freeze_low_rank_residual_flow(
        task_id=str(payload["task_id"]),
        direction=torch.tensor(payload["direction"], dtype=torch.float64),
        defensive_weight=float(payload["defensive_weight"]),
        maximum_log_scale=float(payload["maximum_log_scale"]),
        partition_style=cast(PartitionStyle, str(payload["partition_style"])),
        layers=tuple(_layer(layer) for layer in payload["layers"]),
        training_seed=int(payload["training_seed"]),
        training_cost=BaselineCostLedger(**payload["training_cost"]),
    )
    return rebuilt.sha256 == payload.get("sha256")


def _smc_contract_matches(payload: dict[str, Any]) -> bool:
    stages = payload.get("stages", [])
    if not isinstance(stages, list) or not stages:
        return False
    previous = 0.0
    for index, stage in enumerate(stages):
        if int(stage["stage"]) != index:
            return False
        if not math.isclose(float(stage["beta_previous"]), previous, abs_tol=1e-14):
            return False
        next_beta = float(stage["beta_next"])
        if not previous < next_beta <= 1.0 or not bool(stage["ess_target_met"]):
            return False
        if not 0.0 <= float(stage["pcn_acceptance_rate"]) <= 1.0:
            return False
        previous = next_beta
    seeds = [int(seed) for seed in payload.get("used_seeds", [])]
    return (
        previous == 1.0
        and float(payload.get("final_beta", -1.0)) == 1.0
        and bool(payload.get("final_particles_equally_weighted"))
        and not bool(payload.get("particles_are_final_inferential_units"))
        and len(seeds) == len(set(seeds))
    )


def _tail_certificate_matches(forecast: dict[str, Any], summary: dict[str, Any]) -> bool:
    certificate = forecast.get("pilot_certificate", {})
    try:
        alpha = float(certificate["per_bound_failure_probability"])
        confidence = float(certificate["confidence_level"])
        mean_upper = float(certificate["mean_square_upper"])
        h_upper = float(certificate["hoeffding_mean_square_upper"])
        eb_upper = float(certificate["empirical_bernstein_mean_square_upper"])
        variance_upper = float(certificate["variance_upper"])
        popoviciu = float(certificate["popoviciu_variance_upper"])
        certified = int(forecast["certified_required_units"])
        planned = int(forecast["planned_units"])
        moments = StreamingMoments(
            count=int(summary["count"]),
            mean=float(summary["estimate"]),
            m2=float(summary["variance"]) * (int(summary["count"]) - 1),
            sum_squares=float(summary["sum_squares"]),
            sum_fourth_powers=float(summary["sum_fourth_powers"]),
            minimum=float(summary["minimum"]),
            maximum=float(summary["maximum"]),
            nonzero_count=int(summary["nonzero_count"]),
        )
        rebuilt = asdict(
            empirical_bernstein_variance_certificate(
                moments,
                bounds=BoundedRange(**certificate["bounds"]),
                confidence_level=confidence,
            )
        )
    except (KeyError, TypeError, ValueError):
        return False
    return (
        math.isclose(alpha, (1.0 - confidence) / 2.0, rel_tol=1e-12)
        and math.isclose(mean_upper, min(h_upper, eb_upper), rel_tol=1e-12)
        and math.isclose(variance_upper, min(mean_upper, popoviciu), rel_tol=1e-12)
        and planned <= certified
        and forecast.get("status") == "v13_development_forecast_not_executed"
        and semantic_equal(rebuilt, certificate)
    )


@dataclass(frozen=True)
class V13DevelopmentAudit:
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


def audit_v13_development(
    *,
    config: dict[str, Any],
    config_sha256: str,
    result: dict[str, Any],
    result_bytes: bytes | None = None,
) -> V13DevelopmentAudit:
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
            smc_contracts = smc_contracts and _smc_contract_matches(candidate["smc"])
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
        rebuilt = aggregate_structured_ecrpt_development(config=config, records=record_list)
        aggregate_match = semantic_equal(rebuilt, result.get("aggregate"))
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        aggregate_match = False
    if not aggregate_match:
        failures.append("aggregate")
    aggregate = result.get("aggregate", {})
    expected_decision = {
        "software_correctness_evidence_pass": bool(aggregate.get("correctness_pass")),
        "development_performance_gate_pass": bool(aggregate.get("performance_pass")),
        "qualification_authorized": bool(aggregate.get("stage_pass")),
        "qualification_executed": False,
        "broad_performance_claim_authorized": False,
        "top_journal_claim_authorized": False,
        "submission_authorized": False,
    }
    decision_match = semantic_equal(expected_decision, result.get("decision"))
    if not decision_match:
        failures.append("decision")
    return V13DevelopmentAudit(
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
