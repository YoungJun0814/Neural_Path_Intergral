"""Strict verification and deterministic reduction of V6 proposal training."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from src.path_integral import SeedLedger, V6WorkLedger


def task_conditioned_training_source_summary(
    source_path: Path,
) -> dict[str, Any]:
    """Derive the deterministic V3 proposal bank from a pure-CEM artifact."""

    raw = source_path.read_bytes()
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    source = json.loads(raw)
    if not isinstance(source, dict) or source.get("schema") not in {
        "npi.g11.v6-baseline-qualification.v1",
        "npi.g11.v6-proposal-training.v1",
    }:
        raise ValueError("unsupported V6 proposal-training source artifact")
    source_schema = str(source["schema"])
    dedicated_source = source_schema == "npi.g11.v6-proposal-training.v1"
    records = source.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("proposal training source must contain records")

    source_contract_verified = False
    if dedicated_source:
        gates = source.get("gates")
        formal = source.get("formal_readiness")
        if (
            source.get("proposal_training_qualified") is not True
            or not isinstance(gates, dict)
            or not gates
            or not all(value is True for value in gates.values())
            or not isinstance(formal, dict)
            or not formal
            or not all(value is True for value in formal.values())
        ):
            raise ValueError("dedicated proposal-training source is not qualified")
        seed_payload = source.get("seed_ledger")
        work_payload = source.get("work_ledger")
        if not isinstance(seed_payload, dict) or not isinstance(work_payload, dict):
            raise ValueError("proposal-training source lacks a strict ledger")
        seed_ledger = SeedLedger.from_dict(seed_payload)
        work_ledger = V6WorkLedger.from_dict(work_payload)
        if (
            seed_ledger.sha256 != source.get("seed_ledger_sha256")
            or work_ledger.sha256 != source.get("work_ledger_sha256")
            or len(seed_ledger) != len(records)
            or len(work_ledger.records) != len(records)
        ):
            raise ValueError("proposal-training source ledger hash or count mismatch")
        record_seed_pairs = {
            (
                json.dumps(
                    record.get("seed_key"),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ),
                record.get("seed"),
            )
            for record in records
        }
        ledger_seed_pairs = {
            (
                json.dumps(
                    asdict(seed_record.key),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ),
                seed_record.seed,
            )
            for seed_record in seed_ledger.records
        }
        if record_seed_pairs != ledger_seed_pairs:
            raise ValueError("proposal-training record seeds do not match the ledger")
        if [record.get("training_work_record") for record in records] != list(
            work_payload["records"]
        ):
            raise ValueError("proposal-training record work does not match the ledger")
        source_contract_verified = True

    grouped_controls: dict[str, list[list[list[float]]]] = {}
    total_samples = 0
    total_work = 0.0
    total_wall = 0.0
    total_cpu = 0.0
    for record in records:
        if record.get("method") != "pure_cem":
            raise ValueError("proposal training source may contain only pure-CEM records")
        if dedicated_source:
            if record.get("cem_fit", {}).get("converged") is not True:
                raise ValueError(
                    "proposal training source contains a nonconverged CEM fit"
                )
            task = str(record.get("task"))
            entry = record.get("training_work_record")
        else:
            if not record["result"]["core"]["complete"]:
                raise ValueError("proposal training source contains an incomplete record")
            task = str(record["preparation"]["core"]["task"])
            training_records = [
                entry
                for entry in record["result"]["total_work"]["records"]
                if entry["category"] == "proposal_training"
            ]
            if len(training_records) != 1:
                raise ValueError(
                    "each proposal training record must have one training ledger entry"
                )
            entry = training_records[0]
        control = record["cem_fit"]["control"]
        if (
            task not in {"terminal_left_tail", "discrete_lower_barrier"}
            or not isinstance(control, list)
            or not control
            or any(
                not isinstance(segment, list)
                or len(segment) != 2
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in segment
                )
                for segment in control
            )
        ):
            raise ValueError("proposal training source contains a malformed CEM control")
        if (
            not isinstance(entry, dict)
            or entry.get("category") != "proposal_training"
            or entry.get("successful", True) is not True
        ):
            raise ValueError("proposal training source contains invalid charged work")
        grouped_controls.setdefault(task, []).append(control)
        total_samples += int(entry["samples"])
        total_work += float(entry["work_units"])
        total_wall += float(entry["wall_seconds"])
        total_cpu += float(entry["cpu_seconds"])

    expected_tasks = {"terminal_left_tail", "discrete_lower_barrier"}
    if set(grouped_controls) != expected_tasks:
        raise ValueError("proposal training source must cover both task families")
    derived: dict[str, list[list[list[float]]]] = {}
    for task, controls in grouped_controls.items():
        segments = len(controls[0])
        if any(len(control) != segments for control in controls):
            raise ValueError(
                "proposal training controls have inconsistent segment counts"
            )
        median_control = [
            [
                statistics.median(
                    control[segment][driver] for control in controls
                )
                for driver in range(2)
            ]
            for segment in range(segments)
        ]
        zero = [[0.0, 0.0] for _ in range(segments)]
        half = [[0.5 * value for value in segment] for segment in median_control]
        derived[task] = [zero, half, median_control]

    return {
        "verified": True,
        "source_artifact_sha256": raw_sha256,
        "source_commit": source.get("source_commit"),
        "source_dirty_worktree": source.get("dirty_worktree"),
        "source_smoke": source.get("smoke"),
        "source_schema": source_schema,
        "source_contract_verified": source_contract_verified,
        "formal_training_source_readiness": (
            dedicated_source
            and source_contract_verified
            and source.get("dirty_worktree") is False
            and source.get("smoke") is False
            and isinstance(source.get("source_commit"), str)
            and len(source["source_commit"]) == 40
            and all(
                character in "0123456789abcdef"
                for character in source["source_commit"]
            )
            and source["source_commit"] != "uncommitted"
        ),
        "source_record_count": len(records),
        "derivation": "componentwise_median_pure_cem_then_zero_half_full_bank",
        "task_controls": derived,
        "total_samples": total_samples,
        "total_work_units": total_work,
        "total_wall_seconds": total_wall,
        "total_cpu_seconds": total_cpu,
    }


def task_conditioned_training_source_audit(
    proposal: dict[str, Any], source_path: Path
) -> dict[str, Any]:
    """Verify a task-conditioned proposal bank against its training artifact."""

    summary = task_conditioned_training_source_summary(source_path)
    if (
        summary["source_artifact_sha256"]
        != proposal["training_source_artifact_sha256"]
    ):
        raise ValueError("proposal training source hash does not match the config")
    configured = proposal["task_controls"]
    derived = summary["task_controls"]
    expected_tasks = {"terminal_left_tail", "discrete_lower_barrier"}
    for task in sorted(expected_tasks):
        configured_tensor = torch.tensor(configured[task], dtype=torch.float64)
        derived_tensor = torch.tensor(derived[task], dtype=torch.float64)
        if configured_tensor.shape != derived_tensor.shape or not torch.allclose(
            configured_tensor, derived_tensor, rtol=0.0, atol=1e-9
        ):
            raise ValueError(
                "configured proposal controls do not match the declared derivation"
            )

    expected_totals = {
        "training_source_record_count": summary["source_record_count"],
        "training_total_samples": summary["total_samples"],
        "training_total_work_units": summary["total_work_units"],
        "training_total_wall_seconds": summary["total_wall_seconds"],
        "training_total_cpu_seconds": summary["total_cpu_seconds"],
    }
    for key, observed in expected_totals.items():
        declared = proposal[key]
        if isinstance(observed, int):
            if int(declared) != observed:
                raise ValueError(f"proposal {key} does not match the source ledger")
        elif not math.isclose(
            float(declared), observed, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(f"proposal {key} does not match the source ledger")
    return {
        key: value for key, value in summary.items() if key != "task_controls"
    }
