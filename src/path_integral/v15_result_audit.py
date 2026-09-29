"""Fail-closed audit of frozen V15 development and qualification artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.path_integral.v15_baseline_protocol import REQUIRED_PRIMARY_COMPARATORS


@dataclass(frozen=True)
class V15ResultAudit:
    passed_integrity: bool
    passed_numerical: bool
    passed_theory: bool
    passed_top_journal_gate: bool
    failures: tuple[str, ...]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_source_provenance(root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    source_changes = []
    for line in status:
        path = line[3:].replace("\\", "/")
        if not path.startswith("results/"):
            source_changes.append(line)
    return {
        "git_commit": commit,
        "source_dirty_before_run": bool(source_changes),
        "source_changes_before_run": source_changes,
    }


def audit_v15_result(payload: dict[str, Any], *, root: Path) -> V15ResultAudit:
    failures: list[str] = []
    if payload.get("schema") != "npi.g11.v15-experiment-result.v1":
        failures.append("wrong result schema")
    provenance = payload.get("source_provenance", {})
    commit = str(provenance.get("git_commit", ""))
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        failures.append("source commit is missing or malformed")
    if provenance.get("source_dirty_before_run") is not False:
        failures.append("experiment was executed from dirty source")
    config_binding = payload.get("config_binding", {})
    config_path = root / str(config_binding.get("path", ""))
    if not config_path.is_file() or file_sha256(config_path) != config_binding.get("sha256"):
        failures.append("configuration hash binding failed")
    elif config_path.suffix in {".yaml", ".yml"}:
        bound_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if isinstance(bound_config, dict) and "gates" in bound_config:
            if payload.get("gates") != bound_config["gates"]:
                failures.append("serialized gates do not match bound configuration")
    seeds = payload.get("used_seeds", [])
    if not seeds or len(seeds) != len(set(seeds)):
        failures.append("seed ledger is empty or contains collisions")
    cells = payload.get("cells", [])
    if not cells:
        failures.append("result contains no cells")
    numerical_failures: list[str] = []
    for cell in cells:
        reference = cell.get("reference", {})
        minimum_reference_snr = float(
            payload.get("gates", {}).get("minimum_reference_signal_to_noise", 0.0)
        )
        reference_estimate = abs(float(reference.get("estimate", 0.0)))
        reference_se = float(reference.get("standard_error", math.inf))
        reference_snr = reference_estimate / max(
            reference_se,
            float.fromhex("0x1.0p-1022"),
        )
        if reference_snr < minimum_reference_snr:
            numerical_failures.append(
                f"{cell.get('cell_id')}: reference signal-to-noise gate failed"
            )
        records = {record.get("method"): record for record in cell.get("methods", [])}
        missing = REQUIRED_PRIMARY_COMPARATORS - set(records)
        if missing:
            failures.append(f"{cell.get('cell_id')}: missing comparators {sorted(missing)}")
        candidate = records.get("v15_cm_transport")
        if candidate is None:
            failures.append(f"{cell.get('cell_id')}: missing V15 candidate")
            continue
        exactness = candidate.get("exactness", {})
        if exactness.get("maximum_likelihood_bound_violation", math.inf) > 1e-10:
            failures.append(f"{cell.get('cell_id')}: defensive likelihood bound failed")
        if not exactness.get("proposal_hash_unchanged", False):
            failures.append(f"{cell.get('cell_id')}: proposal hash changed")
        if not math.isfinite(float(candidate.get("estimate", math.nan))):
            failures.append(f"{cell.get('cell_id')}: candidate estimate is nonfinite")
        if float(candidate.get("accuracy_z", math.inf)) > float(
            payload.get("gates", {}).get("maximum_accuracy_z", 4.0)
        ):
            numerical_failures.append(f"{cell.get('cell_id')}: candidate accuracy gate failed")
        ratio = float(cell.get("best_primary_over_v15_work_ratio", 0.0))
        required = float(payload.get("gates", {}).get("minimum_work_ratio", 1.0))
        if ratio < required:
            numerical_failures.append(f"{cell.get('cell_id')}: work-efficiency gate failed")
        minimum_comparators = int(
            payload.get("gates", {}).get("minimum_accuracy_qualified_comparators", 0)
        )
        qualified = set(cell.get("accuracy_qualified_primary", []))
        if len(qualified) < minimum_comparators:
            numerical_failures.append(
                f"{cell.get('cell_id')}: too few accuracy-qualified comparators"
            )
    ledger_path = root / "configs/g11_v15/theorem_ledger_v1.yaml"
    ledger = yaml.safe_load(ledger_path.read_text(encoding="utf-8"))
    passed_theory = bool(ledger["gates"]["G5"]["pass"])
    submission_unlocked = True
    claim_path = root / "configs/g11_v15/claim_contract_v1.yaml"
    if claim_path.is_file():
        claim_contract = yaml.safe_load(claim_path.read_text(encoding="utf-8"))
        external = claim_contract.get("external_novelty_review", {})
        completed = int(external.get("completed", 0))
        required = int(external.get("required", 0))
        submission_unlocked = (
            completed >= required
            and external.get("submission_lock") is False
            and claim_contract.get("gates", {}).get("p8_baselines") == "pass"
            and claim_contract.get("gates", {}).get("qualification") == "pass"
        )
    integrity = not failures
    numerical = integrity and not numerical_failures
    all_failures = failures + numerical_failures
    if not passed_theory:
        all_failures.append("G5 theory gate is open: T15-5 is not proved")
    if not submission_unlocked:
        all_failures.append("top-journal submission locks remain open")
    return V15ResultAudit(
        passed_integrity=integrity,
        passed_numerical=numerical,
        passed_theory=passed_theory,
        passed_top_journal_gate=(
            integrity and numerical and passed_theory and submission_unlocked
        ),
        failures=tuple(all_failures),
    )


def load_and_audit_v15_result(path: Path, *, root: Path) -> V15ResultAudit:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return audit_v15_result(payload, root=root)
