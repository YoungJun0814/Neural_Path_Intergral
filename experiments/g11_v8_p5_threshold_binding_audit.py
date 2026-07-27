"""Fail-closed binding audit for the V8 P5 threshold manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "npi.g11.v8-p5-threshold-manifest-binding.v1"
REPORT = "npi.g11.v8-p5-threshold-binding-audit.v1"
ROOT_KEYS = {
    "schema",
    "protocol_id",
    "date",
    "phase",
    "outcome_data_used",
    "p5_reference_matrix_design_sha256",
    "p6_statistical_design_sha256",
    "threshold_calibration_config",
    "threshold_calibration_config_sha256",
    "threshold_calibration_result",
    "threshold_calibration_result_sha256",
    "threshold_manifest_sha256",
    "calibration_source_commit",
    "reference_seed_namespace",
    "final_method_seed_namespace",
    "decision",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_binding(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    value = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("unexpected P5 threshold-binding schema")
    return value, hashlib.sha256(raw).hexdigest()


def _result_path(value: object) -> Path:
    if not isinstance(value, str):
        raise ValueError("threshold result path must be a string")
    candidate = (ROOT / value).resolve()
    if ROOT not in candidate.parents or candidate.suffix != ".json":
        raise ValueError("threshold result path must remain in the repository")
    return candidate


def audit_binding(binding: dict[str, Any], digest: str) -> dict[str, Any]:
    p5_matrix = ROOT / "configs/g11_v8/p5_reference_matrix_design_v1.yaml"
    p6_design = ROOT / "configs/g11_v8/p6_statistical_design_v1.yaml"
    calibration_config = ROOT / str(binding.get("threshold_calibration_config", ""))
    try:
        result_path = _result_path(binding.get("threshold_calibration_result"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        result_path = None
        result = {}
    manifest = result.get("candidate_manifest", {}) if isinstance(result, dict) else {}
    cells = result.get("cells", []) if isinstance(result, dict) else []
    seed_records = result.get("seed_ledger", {}).get("records", []) if isinstance(result, dict) else []
    seed_roles = {
        item.get("key", {}).get("role")
        for item in seed_records
        if isinstance(item, dict) and isinstance(item.get("key"), dict)
    }
    decision = binding.get("decision", {})
    checks = {
        "schema_exact": binding.get("schema") == SCHEMA,
        "root_keys_exact": set(binding) == ROOT_KEYS,
        "phase_exact": binding.get("phase") == "p5_threshold_bound",
        "outcome_blind": binding.get("outcome_data_used") is False,
        "p5_matrix_hash_bound": p5_matrix.is_file()
        and binding.get("p5_reference_matrix_design_sha256") == _sha(p5_matrix),
        "p6_design_hash_bound": p6_design.is_file()
        and binding.get("p6_statistical_design_sha256") == _sha(p6_design),
        "calibration_config_hash_bound": calibration_config.is_file()
        and binding.get("threshold_calibration_config_sha256") == _sha(calibration_config),
        "calibration_result_hash_bound": result_path is not None
        and binding.get("threshold_calibration_result_sha256") == _sha(result_path),
        "clean_p5_calibration_result": result.get("schema")
        == "npi.g11.v8-p5-threshold-calibration.v1"
        and result.get("passed") is True
        and result.get("smoke") is False
        and result.get("dirty_worktree") is False,
        "source_commit_bound": result.get("source_commit") == binding.get("calibration_source_commit"),
        "complete_24_cell_manifest": isinstance(cells, list)
        and len(cells) == 24
        and isinstance(manifest, dict)
        and len(manifest.get("cells", [])) == 24,
        "manifest_hash_bound": isinstance(manifest, dict)
        and result.get("candidate_manifest_sha256")
        == hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
                "utf-8"
            )
        ).hexdigest()
        and result.get("candidate_manifest_sha256") == binding.get("threshold_manifest_sha256"),
        "all_calibration_gates_pass": isinstance(result.get("gates"), dict)
        and bool(result["gates"])
        and all(result["gates"].values()),
        "p5_seed_roles_exact": bool(seed_roles)
        and all(isinstance(role, str) and role.startswith("p5-threshold-") for role in seed_roles),
        "reference_and_final_namespaces_fresh": binding.get("reference_seed_namespace")
        == "p5-reference"
        and binding.get("final_method_seed_namespace") == "p5-final-method"
        and not any(
            name in role
            for role in seed_roles
            for name in ("p5-reference", "p5-final-method")
        ),
        "thresholds_bound": decision.get("calibrated_thresholds_hash_bound") is True,
        "references_open": decision.get("independent_reference_execution_complete") is False,
        "performance_refused": decision.get("performance_claim_authorized") is False,
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "schema": REPORT,
        "binding_sha256": digest,
        "checks": checks,
        "failure_count": len(failures),
        "failures": failures,
        "passed": not failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {args.output}")
    binding, digest = load_binding(args.binding)
    report = audit_binding(binding, digest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
