"""Freeze a cross-platform V6 reproduction audit before Linux outcomes exist."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from experiments.g11_v6_hardware_reproduction import _load_config
from src.path_integral.provenance import source_provenance

_RECEIPT_SCHEMA = "npi.g11.v6-freeze-receipt.v2"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_frozen_hardware_config(
    template: dict[str, Any],
    *,
    canonical_confirmation_sha256: str,
    reproduction_hashes: dict[str, str],
) -> dict[str, Any]:
    """Return the deterministic, outcome-blind hardware-audit config."""

    if template.get("frozen") is not False:
        raise ValueError("hardware freeze requires an unfrozen template")
    expected_reproduction = {
        "baseline_config",
        "policy_config",
        "manifest",
        "reference",
        "power",
        "audit_config",
    }
    if set(reproduction_hashes) != expected_reproduction:
        raise ValueError("reproduction protocol hashes are incomplete")
    values = (canonical_confirmation_sha256, *reproduction_hashes.values())
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in values
    ):
        raise ValueError("hardware freeze received a malformed SHA-256")
    frozen = json.loads(json.dumps(template))
    frozen["protocol_id"] = "g11-v6-hardware-reproduction-confirmation-v1"
    frozen["frozen"] = True
    frozen["expected_sha256"] = {
        "canonical_confirmation": canonical_confirmation_sha256,
        "reproduction_baseline_config": reproduction_hashes["baseline_config"],
        "reproduction_policy_config": reproduction_hashes["policy_config"],
        "manifest": reproduction_hashes["manifest"],
        "reference": reproduction_hashes["reference"],
        "power": reproduction_hashes["power"],
        "audit_config": reproduction_hashes["audit_config"],
    }
    return frozen


def run(
    *,
    template_path: Path,
    canonical_confirmation_path: Path,
    canonical_baseline_path: Path,
    reproduction_freeze_receipt_path: Path,
    reproduction_freeze_directory: Path,
    output_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    provenance = source_provenance()
    if provenance["dirty_worktree"]:
        raise RuntimeError("formal hardware freeze requires a clean source tree")
    template, template_hash = _load_config(template_path)
    canonical_raw = canonical_confirmation_path.read_bytes()
    canonical = json.loads(canonical_raw)
    if (
        not isinstance(canonical, dict)
        or canonical.get("schema") != "npi.g11.v6-confirmatory.v1"
        or not canonical.get("confirmation_passed")
    ):
        raise ValueError("hardware freeze requires a passing canonical confirmation")
    canonical_baseline_raw = canonical_baseline_path.read_bytes()
    canonical_baseline = json.loads(canonical_baseline_raw)
    result_hashes = canonical.get("result_artifact_sha256", {})
    if result_hashes.get("baseline") != hashlib.sha256(
        canonical_baseline_raw
    ).hexdigest():
        raise ValueError("canonical confirmation does not authenticate its baseline")
    freeze_receipt_raw = reproduction_freeze_receipt_path.read_bytes()
    freeze_receipt = json.loads(freeze_receipt_raw)
    if (
        not isinstance(freeze_receipt, dict)
        or freeze_receipt.get("schema") != _RECEIPT_SCHEMA
        or freeze_receipt.get("protocol_version") is None
    ):
        raise ValueError("unsupported reproduction freeze receipt")
    if canonical_baseline.get("source_commit") != freeze_receipt.get("source_commit"):
        raise ValueError(
            "reproduction must execute the same source commit as the canonical run"
        )
    reproduction_hashes = freeze_receipt.get("frozen_protocol_sha256")
    if not isinstance(reproduction_hashes, dict):
        raise ValueError("reproduction freeze receipt lacks protocol hashes")
    file_map = {
        "baseline_config": "baseline_confirmation.yaml",
        "policy_config": "routed_policy_confirmation.yaml",
        "audit_config": "result_audit_confirmation.yaml",
    }
    for key, name in file_map.items():
        if _sha256(reproduction_freeze_directory / name) != reproduction_hashes.get(
            key
        ):
            raise ValueError(f"reproduction {key} file does not match its receipt")
    frozen = build_frozen_hardware_config(
        template,
        canonical_confirmation_sha256=hashlib.sha256(canonical_raw).hexdigest(),
        reproduction_hashes=reproduction_hashes,
    )
    serialized = yaml.safe_dump(
        frozen,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    ).encode("utf-8")
    if output_path.exists() or receipt_path.exists():
        raise FileExistsError("hardware freeze refuses to overwrite an artifact")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(serialized)
    receipt = {
        "schema": "npi.g11.v6-hardware-freeze-receipt.v1",
        "source_commit": provenance["source_commit"],
        "canonical_execution_source_commit": canonical_baseline["source_commit"],
        "reproduction_execution_source_commit": freeze_receipt["source_commit"],
        "reproduction_protocol_version": freeze_receipt["protocol_version"],
        "config_sha256": hashlib.sha256(serialized).hexdigest(),
        "input_sha256": {
            "template": template_hash,
            "canonical_confirmation": hashlib.sha256(canonical_raw).hexdigest(),
            "canonical_baseline": hashlib.sha256(canonical_baseline_raw).hexdigest(),
            "reproduction_freeze_receipt": hashlib.sha256(
                freeze_receipt_raw
            ).hexdigest(),
        },
        "expected_sha256": frozen["expected_sha256"],
    }
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--canonical-confirmation", type=Path, required=True)
    parser.add_argument("--canonical-baseline", type=Path, required=True)
    parser.add_argument("--reproduction-freeze-receipt", type=Path, required=True)
    parser.add_argument("--reproduction-freeze-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args()
    receipt = run(
        template_path=arguments.template,
        canonical_confirmation_path=arguments.canonical_confirmation,
        canonical_baseline_path=arguments.canonical_baseline,
        reproduction_freeze_receipt_path=arguments.reproduction_freeze_receipt,
        reproduction_freeze_directory=arguments.reproduction_freeze_directory,
        output_path=arguments.output,
        receipt_path=arguments.receipt,
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
