"""Run the complete laptop-safe V6 development pipeline without scientific promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from experiments.g11_v6_baseline_qualification import run as run_baselines
from experiments.g11_v6_rarity_calibration import run as run_calibration
from experiments.g11_v6_reference import run as run_reference
from experiments.g11_v6_result_audit import run as run_audit
from experiments.g11_v6_routed_policy import run as run_policy
from experiments.g11_v6_theory_diagnostics import run as run_theory
from src.path_integral.provenance import runtime_provenance, source_provenance

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "g11_v6"


def _write(path: Path, payload: dict[str, Any]) -> str:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output_directory: Path) -> dict[str, Any]:
    """Execute all laptop-safe stages once and preserve their dependency hashes."""

    started = time.perf_counter()
    output_directory.mkdir(parents=True, exist_ok=True)
    names = {
        "calibration": output_directory / "01_calibration.json",
        "manifest": output_directory / "02_manifest.json",
        "reference": output_directory / "03_reference.json",
        "baseline": output_directory / "04_baseline.json",
        "policy": output_directory / "05_policy.json",
        "baseline_audit": output_directory / "06_baseline_audit.json",
        "policy_audit": output_directory / "07_policy_audit.json",
        "theory": output_directory / "08_theory_diagnostics.json",
        "summary": output_directory / "09_summary.json",
    }
    if any(path.exists() for path in names.values()):
        raise FileExistsError("laptop smoke refuses to overwrite an existing artifact")

    calibration = run_calibration(
        CONFIG / "rarity_calibration_development.yaml", smoke=True
    )
    hashes = {"calibration": _write(names["calibration"], calibration)}
    manifest = calibration["candidate_manifest"]
    hashes["manifest"] = _write(names["manifest"], manifest)

    reference = run_reference(
        CONFIG / "reference_development.yaml", names["manifest"], smoke=True
    )
    hashes["reference"] = _write(names["reference"], reference)
    baseline = run_baselines(
        CONFIG / "baseline_qualification_development.yaml",
        names["manifest"],
        names["reference"],
        smoke=True,
    )
    hashes["baseline"] = _write(names["baseline"], baseline)
    policy = run_policy(
        CONFIG / "routed_policy_development.yaml",
        names["manifest"],
        names["reference"],
        smoke=True,
    )
    hashes["policy"] = _write(names["policy"], policy)
    baseline_audit = run_audit(
        CONFIG / "result_audit_development.yaml", names["baseline"]
    )
    hashes["baseline_audit"] = _write(names["baseline_audit"], baseline_audit)
    policy_audit = run_audit(
        CONFIG / "result_audit_development.yaml", names["policy"]
    )
    hashes["policy_audit"] = _write(names["policy_audit"], policy_audit)
    theory = run_theory(
        CONFIG / "theory_diagnostics_development.yaml", names["manifest"], smoke=True
    )
    hashes["theory"] = _write(names["theory"], theory)

    gates = {
        "calibration_structure": bool(calibration["gates"]["complete_matrix"]),
        "reference_structure": bool(reference["gates"]["complete_reference_matrix"]),
        "baseline_complete": bool(baseline["gates"]["complete_matrix"])
        and bool(baseline["gates"]["all_runs_complete"]),
        "policy_complete": bool(policy["gates"]["complete_matrix"])
        and bool(policy["gates"]["all_runs_complete"]),
        "baseline_offline_audit": bool(baseline_audit["gates"]["all_records_pass"]),
        "policy_offline_audit": bool(policy_audit["gates"]["all_records_pass"]),
        "theory_pathwise_exactness": bool(theory["gates"]["pathwise_exactness"]),
        "terminal_inverse_moment_bound": bool(
            theory["gates"]["terminal_analytic_inverse_moment_bounds_finite"]
        ),
        "scientific_promotion_prohibited": not any(
            (
                baseline["baseline_qualified"],
                policy["policy_qualified"],
                baseline_audit["qualification_audit_passed"],
                policy_audit["qualification_audit_passed"],
                theory["diagnostics_qualified"],
            )
        ),
    }
    provenance = source_provenance()
    summary = {
        "schema": "npi.g11.v6-laptop-smoke.v1",
        "smoke": True,
        "artifact_sha256": hashes,
        "gates": gates,
        "passed": all(gates.values()),
        "elapsed_seconds": time.perf_counter() - started,
        "scientific_claim": False,
        "environment": runtime_provenance(dtype="torch.float64"),
        **provenance,
    }
    _write(names["summary"], summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.output_directory)
    print(json.dumps({"passed": result["passed"], **result["gates"]}, sort_keys=True))


if __name__ == "__main__":
    main()
