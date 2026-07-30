"""Build and audit the complete 48-entry reference-proposal manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml

from experiments.g11_v8_p5_sharded_reference_common import ROOT, load_context
from src.path_integral import (
    REFERENCE_METHODS,
    TimePiecewiseTwoDriverControl,
    rank_one_price_control_span,
)
from src.path_integral.provenance import source_provenance
from src.path_integral.reference_protocol import canonical_sha256
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting

SCHEMA = "npi.g11.v8-p5-reference-proposal-manifest-build.v1"
MANIFEST_SCHEMA = "npi.g11.v8-p5-reference-proposal-manifest.v1"
AUDIT_SCHEMA = "npi.g11.v8-p5-reference-proposal-manifest-audit.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bound_path(record: Any) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError("proposal-manifest binding is malformed")
    path = (ROOT / str(record["path"])).resolve()
    if ROOT not in path.parents or not path.is_file():
        raise ValueError("proposal-manifest bound path is invalid")
    if record["sha256"] != _sha256(path):
        raise ValueError("proposal-manifest artifact hash mismatch")
    return path


def load_manifest_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(config, dict) or config.get("schema") != SCHEMA:
        raise ValueError("unexpected proposal-manifest build schema")
    for field in (
        "threshold_binding",
        "threshold_proposal",
        "all_failed_cells_result",
        "all_failed_cells_audit",
        "dense_barrier_result",
        "dense_barrier_audit",
        "dense_terminal_result",
        "dense_terminal_audit",
    ):
        _bound_path(config.get(field))
    protocol = config.get("reference_protocol")
    overrides = config.get("override_sources")
    decision = config.get("decision")
    if (
        config.get("design_informed_by_prior_development_outcomes") is not True
        or config.get("current_namespace_outcomes_inspected_before_freeze") is not False
        or not isinstance(protocol, dict)
        or protocol.get("id")
        != "g11-v8-p5-sharded-reference-cell-tuned-v1"
        or protocol.get("pilot_namespace")
        != "v8-r2-reference-cell-tuned-pilot-v1"
        or protocol.get("final_namespace")
        != "v8-r2-reference-cell-tuned-final-v1"
        or protocol.get("methods") != list(REFERENCE_METHODS)
        or protocol.get("estimand") != "fixed_finest_grid"
        or protocol.get("dtype") != "float64"
        or protocol.get("device") != "cpu"
        or not isinstance(overrides, dict)
        or overrides.get("v3_selected_requirement_count") != 9
        or overrides.get("total_override_count") != 11
        or not isinstance(decision, dict)
        or decision.get("proposal_manifest_build_authorized") is not True
        or decision.get("new_full_pilot_authorized") is not False
        or decision.get("final_execution_authorized") is not False
        or decision.get("performance_claim_authorized") is not False
    ):
        raise ValueError("proposal-manifest build contract is invalid")
    return config, hashlib.sha256(raw).hexdigest()


def _load_json_binding(config: dict[str, Any], field: str) -> dict[str, Any]:
    value = json.loads(_bound_path(config[field]).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{field} must bind a JSON mapping")
    return value


def _proposal(
    *,
    weights: list[Any],
    schedules: list[Any],
    source_kind: str,
    source_id: str,
) -> dict[str, Any]:
    resolved_weights = [float(weight) for weight in weights]
    resolved_schedules = [
        [[float(pair[0]), float(pair[1])] for pair in schedule]
        for schedule in schedules
    ]
    if (
        len(resolved_weights) != len(resolved_schedules)
        or any(weight <= 0.0 for weight in resolved_weights)
        or not math.isclose(sum(resolved_weights), 1.0)
        or not resolved_schedules
        or any(not schedule for schedule in resolved_schedules)
    ):
        raise ValueError("resolved proposal is malformed")
    if any(abs(value) > 0.0 for pair in resolved_schedules[0] for value in pair):
        raise ValueError("every defensive proposal must start with the natural expert")
    return {
        "source_kind": source_kind,
        "source_id": source_id,
        "weights": resolved_weights,
        "schedules": resolved_schedules,
    }


def _assert_rank_one(proposal: dict[str, Any], *, maturity: float, steps: int) -> None:
    controls = [
        TimePiecewiseTwoDriverControl(
            tuple((float(pair[0]), float(pair[1])) for pair in schedule),
            maturity=maturity,
        )
        for schedule in proposal["schedules"]
    ]
    times = torch.arange(steps, dtype=torch.float64) * (maturity / steps)
    expanded = torch.stack(
        [control.deterministic_schedule(times) for control in controls]
    )
    span = rank_one_price_control_span(expanded, step_dt=maturity / steps)
    if span.maximum_span_residual > 1e-10:
        raise ValueError("proposal failed the exact rank-one structural check")


def build_proposal_manifest(config_path: Path) -> dict[str, Any]:
    config, config_sha256 = load_manifest_config(config_path)
    provenance = source_provenance()
    if provenance["dirty_worktree"]:
        raise RuntimeError("proposal manifest requires a clean Git worktree")
    context = load_context(
        ROOT / "configs/g11_v8/p5_sharded_reference_execution_v3.yaml"
    )
    if context.binding_sha256 != config["threshold_binding"]["sha256"]:
        raise ValueError("proposal manifest binds a different threshold manifest")
    threshold = yaml.safe_load(
        _bound_path(config["threshold_proposal"]).read_text(encoding="utf-8")
    )
    if not isinstance(threshold, dict):
        raise ValueError("threshold proposal must be a mapping")
    v3 = _load_json_binding(config, "all_failed_cells_result")
    v3_audit = _load_json_binding(config, "all_failed_cells_audit")
    barrier = _load_json_binding(config, "dense_barrier_result")
    barrier_audit = _load_json_binding(config, "dense_barrier_audit")
    terminal = _load_json_binding(config, "dense_terminal_result")
    terminal_audit = _load_json_binding(config, "dense_terminal_audit")
    if (
        v3_audit.get("passed") is not True
        or barrier_audit.get("passed") is not True
        or terminal_audit.get("passed") is not True
        or terminal_audit.get("decision", {}).get(
            "proposal_manifest_freeze_authorized"
        )
        is not True
    ):
        raise ValueError("proposal source audits do not authorize manifest construction")
    overrides: dict[tuple[str, str], dict[str, Any]] = {}
    for cell_id, methods in v3["selected_proposals"].items():
        for method, selected in methods.items():
            overrides[(cell_id, method)] = _proposal(
                weights=selected["weights"],
                schedules=selected["schedules"],
                source_kind="all_failed_cells_v3",
                source_id=selected["candidate_id"],
            )
    barrier_cell = config["override_sources"]["dense_barrier_cell_id"]
    barrier_method = config["override_sources"]["dense_barrier_method"]
    barrier_selected = barrier["selected_proposals"][barrier_cell]
    overrides[(barrier_cell, barrier_method)] = _proposal(
        weights=barrier_selected["weights"],
        schedules=barrier_selected["schedules"],
        source_kind="dense_barrier_v1",
        source_id=barrier_selected["candidate_id"],
    )
    terminal_cell = config["override_sources"]["dense_terminal_cell_id"]
    terminal_method = config["override_sources"]["dense_terminal_method"]
    terminal_selected = terminal["selected_proposals"][terminal_cell]
    overrides[(terminal_cell, terminal_method)] = _proposal(
        weights=terminal_selected["weights"],
        schedules=terminal_selected["schedules"],
        source_kind="dense_terminal_v2",
        source_id=terminal_selected["candidate_id"],
    )
    if len(overrides) != int(config["override_sources"]["total_override_count"]):
        raise ValueError("proposal override roster is incomplete")
    entries = []
    for cell_id, cell in context.cells_by_id.items():
        for method in REFERENCE_METHODS:
            key = (cell_id, method)
            if key in overrides:
                proposal = overrides[key]
                overridden = True
            else:
                task_name = str(cell["task"])
                proposal = _proposal(
                    weights=threshold["proposal"]["weights"],
                    schedules=threshold["proposal"]["task_controls"][task_name],
                    source_kind="threshold_default",
                    source_id=task_name,
                )
                overridden = False
            _assert_rank_one(
                proposal,
                maturity=float(cell["maturity"]),
                steps=int(cell["finest_steps"]),
            )
            entries.append(
                {
                    "entry_id": f"{cell_id}/{method}",
                    "cell_id": cell_id,
                    "method": method,
                    "overrides_failed_pilot_entry": overridden,
                    **proposal,
                }
            )
    protocol = config["reference_protocol"]
    return {
        "schema": MANIFEST_SCHEMA,
        "protocol_id": protocol["id"],
        "build_config_sha256": config_sha256,
        "threshold_binding_sha256": context.binding_sha256,
        "threshold_manifest_sha256": context.binding[
            "threshold_manifest_sha256"
        ],
        "pilot_namespace": protocol["pilot_namespace"],
        "final_namespace": protocol["final_namespace"],
        "methods": protocol["methods"],
        "estimand": protocol["estimand"],
        "dtype": protocol["dtype"],
        "device": protocol["device"],
        "entry_count": len(entries),
        "override_count": sum(
            bool(entry["overrides_failed_pilot_entry"]) for entry in entries
        ),
        "entries": entries,
        "development_selection_only": True,
        "fresh_pilot_required": True,
        "final_execution_authorized": False,
        "performance_claim_authorized": False,
        **provenance,
    }


def audit_proposal_manifest(
    config_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    config, config_sha256 = load_manifest_config(config_path)
    manifest = _load_json_binding(
        {"manifest": {"path": str(manifest_path.relative_to(ROOT)), "sha256": _sha256(manifest_path)}},
        "manifest",
    )
    entries = manifest.get("entries", [])
    matrix = {
        (entry.get("cell_id"), entry.get("method"))
        for entry in entries
        if isinstance(entry, dict)
    }
    context = load_context(
        ROOT / "configs/g11_v8/p5_sharded_reference_execution_v3.yaml"
    )
    expected_matrix = {
        (cell_id, method)
        for cell_id in context.cells_by_id
        for method in REFERENCE_METHODS
    }
    structural_pass = True
    for entry in entries:
        try:
            _assert_rank_one(
                entry,
                maturity=float(context.cells_by_id[entry["cell_id"]]["maturity"]),
                steps=int(context.cells_by_id[entry["cell_id"]]["finest_steps"]),
            )
        except (KeyError, TypeError, ValueError):
            structural_pass = False
    source_counts = {
        source: sum(entry.get("source_kind") == source for entry in entries)
        for source in (
            "threshold_default",
            "all_failed_cells_v3",
            "dense_barrier_v1",
            "dense_terminal_v2",
        )
    }
    checks = {
        "schema_protocol_config_exact": manifest.get("schema") == MANIFEST_SCHEMA
        and manifest.get("protocol_id") == config["reference_protocol"]["id"]
        and manifest.get("build_config_sha256") == config_sha256,
        "threshold_binding_exact": manifest.get("threshold_binding_sha256")
        == config["threshold_binding"]["sha256"],
        "complete_unique_matrix": len(entries) == 48
        and len(matrix) == 48
        and matrix == expected_matrix,
        "override_roster_exact": manifest.get("override_count") == 11
        and sum(
            bool(entry.get("overrides_failed_pilot_entry")) for entry in entries
        )
        == 11
        and source_counts
        == {
            "threshold_default": 37,
            "all_failed_cells_v3": 9,
            "dense_barrier_v1": 1,
            "dense_terminal_v2": 1,
        },
        "weights_and_natural_component_exact": all(
            len(entry["weights"]) == len(entry["schedules"])
            and all(float(weight) > 0.0 for weight in entry["weights"])
            and math.isclose(sum(float(weight) for weight in entry["weights"]), 1.0)
            and all(
                abs(float(value)) == 0.0
                for pair in entry["schedules"][0]
                for value in pair
            )
            for entry in entries
        ),
        "rank_one_structure_exact": structural_pass,
        "clean_frozen_source": manifest.get("dirty_worktree") is False
        and isinstance(manifest.get("source_commit"), str)
        and len(manifest["source_commit"]) == 40,
        "decision_fail_closed": manifest.get("development_selection_only") is True
        and manifest.get("fresh_pilot_required") is True
        and manifest.get("final_execution_authorized") is False
        and manifest.get("performance_claim_authorized") is False,
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    return {
        "schema": AUDIT_SCHEMA,
        "manifest_file_sha256": _sha256(manifest_path),
        "manifest_canonical_sha256": canonical_sha256(manifest),
        "checks": checks,
        "source_counts": source_counts,
        "failures": failures,
        "passed": not failures,
        "decision": {
            "status": (
                "reference_proposal_manifest_audit_pass"
                if not failures
                else "reference_proposal_manifest_audit_fail"
            ),
            "new_execution_config_authorized": not failures,
            "new_full_pilot_authorized": False,
            "final_execution_authorized": False,
            "performance_claim_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--config", type=Path, required=True)
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "build":
        manifest = build_proposal_manifest(arguments.config)
        digest = write_json_atomic_nonoverwriting(arguments.output, manifest)
        print(
            json.dumps(
                {
                    "manifest_sha256": digest,
                    "entry_count": manifest["entry_count"],
                    "override_count": manifest["override_count"],
                    "new_full_pilot_authorized": False,
                },
                sort_keys=True,
            )
        )
        return
    report = audit_proposal_manifest(arguments.config, arguments.manifest)
    digest = write_json_atomic_nonoverwriting(arguments.output, report)
    print(json.dumps({"audit_sha256": digest, **report["decision"]}, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
