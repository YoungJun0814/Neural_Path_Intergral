"""Fail-closed audit of the final V16 finite-grid policy and claim boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from src.path_integral.v15_result_audit import file_sha256, git_source_provenance

ROOT = Path(__file__).resolve().parents[1]


def _load_bound(binding: dict[str, str]) -> dict[str, Any]:
    path = ROOT / binding["path"]
    if not path.is_file() or file_sha256(path) != binding["sha256"]:
        raise RuntimeError(f"artifact binding failed: {binding['path']}")
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    canonical = _load_bound(config["artifacts"]["canonical_confirmation"])
    ood = _load_bound(config["artifacts"]["ood_confirmation"])
    theorems = _load_bound(config["artifacts"]["theorem_ledger"])
    claims = _load_bound(config["artifacts"]["claim_contract"])
    novelty = _load_bound(config["artifacts"]["novelty_ledger"])
    internal_failures = []
    if canonical.get("passed") is not True or canonical.get("failures"):
        internal_failures.append("canonical confirmation failed")
    if ood.get("passed") is not True or ood.get("failures"):
        internal_failures.append("OOD confirmation failed")
    for name, artifact in (("canonical", canonical), ("OOD", ood)):
        provenance = artifact.get("source_provenance", {})
        if provenance.get("source_dirty_before_run") is not False or provenance.get(
            "source_changes_before_run"
        ):
            internal_failures.append(f"{name} confirmation source was not clean")

    minimum_ratio = float(config["gates"]["minimum_strongest_comparator_ratio"])
    canonical_records = {
        str(item["reference_cell"]): item for item in canonical["variants"]
    }
    for cell in config["required_cells"]["canonical_dominance"]:
        record = canonical_records.get(cell)
        if record is None or record.get("passed") is not True:
            internal_failures.append(f"{cell}: canonical dominance record missing")
        elif float(record["strongest_comparator_over_candidate_work_ratio"]) < minimum_ratio:
            internal_failures.append(f"{cell}: canonical dominance ratio failed")

    ood_records = {str(item["reference_cell"]): item for item in ood["variants"]}
    dominance_summary = {}
    for cell in config["required_cells"]["ood_dominance"]:
        record = ood_records.get(cell)
        if record is None:
            internal_failures.append(f"{cell}: OOD dominance record missing")
            continue
        ratio = float(record["strongest_comparator_over_candidate_work_ratio"])
        dominance_summary[cell] = ratio
        if (
            record.get("passed") is not True
            or record.get("claim_role") != "dominance_candidate"
            or record.get("require_comparator_dominance") is not True
            or ratio < minimum_ratio
        ):
            internal_failures.append(f"{cell}: OOD dominance contract failed")

    fallback_summary = {}
    for cell in config["required_cells"]["correctness_fallback"]:
        record = ood_records.get(cell)
        if record is None:
            internal_failures.append(f"{cell}: fallback record missing")
            continue
        fallback_summary[cell] = {
            "accuracy_z": float(record["external_accuracy_z"]),
            "robust_relative_standard_error": float(
                record["robust_relative_standard_error"]
            ),
            "strongest_comparator_ratio_descriptive_only": float(
                record["strongest_comparator_over_candidate_work_ratio"]
            ),
        }
        if (
            record.get("passed") is not True
            or record.get("claim_role") != "correctness_fallback"
            or record.get("require_comparator_dominance") is not False
            or float(record["maximum_likelihood_bound_violation"]) > 0.0
        ):
            internal_failures.append(f"{cell}: correctness fallback contract failed")

    theorem_records = theorems["theorems"]
    for theorem in config["gates"]["required_theorems"]:
        if theorem_records.get(theorem, {}).get("status") != "proved":
            internal_failures.append(f"{theorem}: required theorem is not proved")
    continuum_gate = str(config["gates"]["continuous_complexity_gate"])
    if theorems["gates"].get(continuum_gate, {}).get("pass") is not False:
        internal_failures.append("continuous complexity lock was unexpectedly opened")

    novelty_review = novelty["external_review"]
    external_result = ROOT / str(config["external_reproduction_result_path"])
    submission_blockers = []
    if int(novelty_review["completed_reviewers"]) < int(
        novelty_review["required_reviewers"]
    ):
        submission_blockers.append("external novelty review incomplete")
    if not external_result.is_file():
        submission_blockers.append("independent person/hardware reproduction absent")
    if claims["external_novelty_review"]["submission_lock"] is not True:
        internal_failures.append("claim contract unexpectedly removed submission lock")
    if theorems["gates"][continuum_gate]["pass"] is not True:
        submission_blockers.append(
            "quantitative continuous relative-bias/end-to-end complexity open"
        )

    return {
        "schema": "npi.g11.v16-final-policy-audit.v1",
        "passed": not internal_failures,
        "internal_failures": internal_failures,
        "finite_grid_empirical_claim_authorized": not internal_failures,
        "uniform_ood_dominance_authorized": False,
        "top_journal_submission_authorized": False,
        "submission_blockers": submission_blockers,
        "dominance_cells": dominance_summary,
        "correctness_fallback_cells": fallback_summary,
        "artifact_bindings": config["artifacts"],
        "source_provenance": git_source_provenance(ROOT),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config: dict[str, Any] = yaml.safe_load(
        config_path.read_text(encoding="utf-8")
    )
    payload = build_audit(config)
    payload["config_binding"] = {
        "path": config_path.relative_to(ROOT).as_posix(),
        "sha256": file_sha256(config_path),
    }
    output = ROOT / str(config["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "passed": payload["passed"]}, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
