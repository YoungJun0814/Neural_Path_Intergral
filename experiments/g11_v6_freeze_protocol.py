"""Create outcome-blind V6 confirmation configs after a passing power gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from experiments.g11_v6_baseline_qualification import _load_references
from experiments.g11_v6_reference import _load_manifest
from experiments.g11_v6_routed_policy import _task_conditioned_training_source_audit
from src.path_integral.provenance import source_provenance


def _yaml_bytes(payload: dict[str, Any]) -> bytes:
    return yaml.safe_dump(
        payload,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_yaml(path: Path, schemas: str | tuple[str, ...]) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_bytes())
    accepted = (schemas,) if isinstance(schemas, str) else schemas
    if not isinstance(payload, dict) or payload.get("schema") not in accepted:
        raise ValueError(f"expected one of {accepted} templates")
    return payload


def build_frozen_configs(
    baseline_template: dict[str, Any],
    policy_template: dict[str, Any],
    audit_template: dict[str, Any],
    confirmatory_template: dict[str, Any],
    *,
    planned_clusters: int,
    manifest_cell_count: int,
    manifest_sha256: str,
    reference_sha256: str,
    power_sha256: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Return deterministic same-design confirmation configs and their hashes."""

    if isinstance(planned_clusters, bool) or planned_clusters < 3:
        raise ValueError("a powered confirmation requires at least three clusters")
    if (
        isinstance(manifest_cell_count, bool)
        or not isinstance(manifest_cell_count, int)
        or manifest_cell_count < 1
    ):
        raise ValueError("confirmation manifest must contain at least one cell")
    for name, value in (
        ("manifest", manifest_sha256),
        ("reference", reference_sha256),
        ("power", power_sha256),
    ):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{name} SHA-256 is malformed")
    templates = (
        baseline_template,
        policy_template,
        audit_template,
        confirmatory_template,
    )
    if any(item.get("frozen") is not False for item in templates):
        raise ValueError("freeze input must be an unfrozen development template")

    baseline = json.loads(json.dumps(baseline_template))
    baseline["protocol_id"] = "g11-v6-baseline-confirmation-v1"
    baseline["phase"] = "confirmation"
    baseline["frozen"] = True
    baseline["sampling"]["clusters"] = planned_clusters

    policy = json.loads(json.dumps(policy_template))
    policy["protocol_id"] = "g11-v6-routed-policy-confirmation-v1"
    policy["phase"] = "confirmation"
    policy["frozen"] = True
    policy["sampling"]["clusters"] = planned_clusters
    if policy.get("schema") in {
        "npi.g11.v6-routed-policy.config.v3",
        "npi.g11.v6-routed-policy.config.v4",
    }:
        policy["proposal"]["training_amortization_record_count"] = (
            manifest_cell_count * planned_clusters
        )

    audit = json.loads(json.dumps(audit_template))
    audit["protocol_id"] = "g11-v6-independent-audit-confirmation-v1"
    audit["frozen"] = True

    hashes = {
        "baseline_config": _sha256(_yaml_bytes(baseline)),
        "policy_config": _sha256(_yaml_bytes(policy)),
        "manifest": manifest_sha256,
        "reference": reference_sha256,
        "power": power_sha256,
        "audit_config": _sha256(_yaml_bytes(audit)),
    }
    confirmatory = json.loads(json.dumps(confirmatory_template))
    confirmatory["protocol_id"] = "g11-v6-confirmatory-v1"
    confirmatory["phase"] = "confirmation"
    confirmatory["frozen"] = True
    confirmatory["expected_sha256"] = hashes

    payloads = {
        "baseline_confirmation.yaml": baseline,
        "routed_policy_confirmation.yaml": policy,
        "result_audit_confirmation.yaml": audit,
        "confirmatory.yaml": confirmatory,
    }
    # Recompute after assembly to catch accidental serialization drift.
    if _sha256(_yaml_bytes(payloads["baseline_confirmation.yaml"])) != hashes[
        "baseline_config"
    ]:
        raise AssertionError("baseline confirmation serialization is unstable")
    if _sha256(_yaml_bytes(payloads["routed_policy_confirmation.yaml"])) != hashes[
        "policy_config"
    ]:
        raise AssertionError("policy confirmation serialization is unstable")
    if _sha256(_yaml_bytes(payloads["result_audit_confirmation.yaml"])) != hashes[
        "audit_config"
    ]:
        raise AssertionError("audit confirmation serialization is unstable")
    return payloads, hashes


def run(
    *,
    baseline_template_path: Path,
    policy_template_path: Path,
    audit_template_path: Path,
    confirmatory_template_path: Path,
    confirmation_manifest_path: Path,
    reference_path: Path,
    power_path: Path,
    output_directory: Path,
    proposal_training_source_path: Path | None = None,
) -> dict[str, Any]:
    provenance = source_provenance()
    if provenance["dirty_worktree"]:
        raise RuntimeError("formal freeze requires a clean source tree")
    manifest = _load_manifest(confirmation_manifest_path)
    if manifest.phase != "confirmation" or not manifest.frozen or manifest.smoke:
        raise ValueError("freeze requires a frozen non-smoke confirmation manifest")
    reference_raw = reference_path.read_bytes()
    reference = json.loads(reference_raw)
    if (
        not isinstance(reference, dict)
        or reference.get("schema") != "npi.g11.v6-reference.v1"
        or not reference.get("reference_qualified")
        or reference.get("smoke")
    ):
        raise ValueError("freeze requires a qualified non-smoke reference artifact")
    references, reference_hash = _load_references(reference_path)
    if set(references) != {cell.cell_id for cell in manifest.cells}:
        raise ValueError("reference and confirmation manifest have different cell sets")
    for cell in manifest.cells:
        if references[cell.cell_id][2] != cell.to_dict():
            raise ValueError(f"reference estimand drift for cell {cell.cell_id}")

    power_raw = power_path.read_bytes()
    power = json.loads(power_raw)
    if (
        not isinstance(power, dict)
        or power.get("schema") != "npi.g11.v6-power-analysis.v1"
        or not power.get("freeze_power_ready")
    ):
        raise ValueError("freeze requires a passing qualification power artifact")
    planned_clusters_raw = power.get("planned_clusters")
    if (
        isinstance(planned_clusters_raw, bool)
        or not isinstance(planned_clusters_raw, int)
        or planned_clusters_raw < 3
    ):
        raise ValueError(
            "passing power artifact must declare an integer cluster count of at least three"
        )
    planned_clusters = planned_clusters_raw
    policy_template = _load_yaml(
        policy_template_path,
        (
            "npi.g11.v6-routed-policy.config.v1",
            "npi.g11.v6-routed-policy.config.v2",
            "npi.g11.v6-routed-policy.config.v3",
            "npi.g11.v6-routed-policy.config.v4",
        ),
    )
    proposal_training_audit = None
    if policy_template["schema"] in {
        "npi.g11.v6-routed-policy.config.v3",
        "npi.g11.v6-routed-policy.config.v4",
    }:
        if proposal_training_source_path is None:
            raise ValueError(
                "freezing a V3/V4 policy requires its proposal training source"
            )
        proposal_training_audit = _task_conditioned_training_source_audit(
            policy_template["proposal"], proposal_training_source_path
        )
        if not proposal_training_audit["formal_training_source_readiness"]:
            raise ValueError(
                "formal freeze requires a clean, committed, non-smoke proposal training source"
            )
    payloads, hashes = build_frozen_configs(
        _load_yaml(
            baseline_template_path,
            (
                "npi.g11.v6-baseline-qualification.config.v1",
                "npi.g11.v6-baseline-qualification.config.v2",
            ),
        ),
        policy_template,
        _load_yaml(audit_template_path, "npi.g11.v6-independent-audit.config.v1"),
        _load_yaml(confirmatory_template_path, "npi.g11.v6-confirmatory.config.v1"),
        planned_clusters=planned_clusters,
        manifest_cell_count=len(manifest.cells),
        manifest_sha256=manifest.sha256,
        reference_sha256=reference_hash,
        power_sha256=_sha256(power_raw),
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    targets = {name: output_directory / name for name in payloads}
    targets["freeze_receipt.json"] = output_directory / "freeze_receipt.json"
    if any(path.exists() for path in targets.values()):
        raise FileExistsError("freeze refuses to overwrite an existing protocol file")
    for name, payload in payloads.items():
        targets[name].write_bytes(_yaml_bytes(payload))
    receipt = {
        "schema": "npi.g11.v6-freeze-receipt.v1",
        "source_commit": provenance["source_commit"],
        "planned_clusters": planned_clusters,
        "manifest_cell_count": len(manifest.cells),
        "frozen_protocol_sha256": hashes,
        "confirmation_config_sha256": _sha256(_yaml_bytes(payloads["confirmatory.yaml"])),
        "proposal_training_audit": proposal_training_audit,
        "input_file_sha256": {
            "baseline_template": _sha256(baseline_template_path.read_bytes()),
            "policy_template": _sha256(policy_template_path.read_bytes()),
            "audit_template": _sha256(audit_template_path.read_bytes()),
            "confirmatory_template": _sha256(confirmatory_template_path.read_bytes()),
            "manifest": _sha256(confirmation_manifest_path.read_bytes()),
            "reference": _sha256(reference_raw),
            "power": _sha256(power_raw),
            **(
                {
                    "proposal_training_source": _sha256(
                        proposal_training_source_path.read_bytes()
                    )
                }
                if proposal_training_source_path is not None
                else {}
            ),
        },
    }
    targets["freeze_receipt.json"].write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-template", type=Path, required=True)
    parser.add_argument("--policy-template", type=Path, required=True)
    parser.add_argument("--audit-template", type=Path, required=True)
    parser.add_argument("--confirmatory-template", type=Path, required=True)
    parser.add_argument("--confirmation-manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--power", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--proposal-training-source", type=Path)
    arguments = parser.parse_args()
    receipt = run(
        baseline_template_path=arguments.baseline_template,
        policy_template_path=arguments.policy_template,
        audit_template_path=arguments.audit_template,
        confirmatory_template_path=arguments.confirmatory_template,
        confirmation_manifest_path=arguments.confirmation_manifest,
        reference_path=arguments.reference,
        power_path=arguments.power,
        output_directory=arguments.output_directory,
        proposal_training_source_path=arguments.proposal_training_source,
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
