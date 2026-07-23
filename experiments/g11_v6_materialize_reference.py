"""Materialize a task-conditioned V6 reference config from the frozen bank."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from experiments.g11_v6_proposal_source import (
    task_conditioned_training_source_audit,
)
from experiments.g11_v6_routed_policy import _load_config as _load_policy_config
from src.path_integral.provenance import source_provenance


def _yaml_bytes(payload: dict[str, Any]) -> bytes:
    return yaml.safe_dump(
        payload,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    ).encode("utf-8")


def materialize_reference_config(
    base_config_path: Path,
    policy_config_path: Path,
    training_source_path: Path,
    *,
    protocol_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reuse the frozen task bank without changing the reference SE contract."""

    base_raw = base_config_path.read_bytes()
    base = yaml.safe_load(base_raw)
    if (
        not isinstance(base, dict)
        or base.get("schema") != "npi.g11.v6-reference.config.v2"
        or base.get("phase") != "qualification"
        or base.get("frozen") is not True
    ):
        raise ValueError("reference materialization requires a frozen V2 base config")
    policy, policy_hash = _load_policy_config(policy_config_path)
    if (
        policy.get("schema") != "npi.g11.v6-routed-policy.config.v3"
        or policy.get("phase") != "qualification"
        or policy.get("frozen") is not True
    ):
        raise ValueError("reference materialization requires a frozen V3 policy")
    if not isinstance(protocol_id, str) or not protocol_id.strip():
        raise ValueError("reference protocol id must be nonempty")
    proposal = json.loads(json.dumps(policy["proposal"]))
    audit = task_conditioned_training_source_audit(
        proposal, training_source_path
    )
    if not audit["formal_training_source_readiness"]:
        raise ValueError("reference materialization requires formal proposal training")

    output = json.loads(json.dumps(base))
    output["schema"] = "npi.g11.v6-reference.config.v3"
    output["protocol_id"] = protocol_id
    output["proposal"] = proposal
    output_bytes = _yaml_bytes(output)
    provenance = source_provenance()
    receipt = {
        "schema": "npi.g11.v6-reference-materialization-receipt.v1",
        "protocol_id": protocol_id,
        "base_config_sha256": hashlib.sha256(base_raw).hexdigest(),
        "policy_config_sha256": policy_hash,
        "output_config_sha256": hashlib.sha256(output_bytes).hexdigest(),
        "reference_contract_unchanged": (
            output["reference_contract"] == base["reference_contract"]
        ),
        "sampling_contract_unchanged": output["sampling"] == base["sampling"],
        "proposal_training_audit": audit,
        "formal_readiness": {
            "clean_source": not bool(provenance["dirty_worktree"]),
            "formal_training_source": bool(
                audit["formal_training_source_readiness"]
            ),
            "frozen_qualification_output": (
                output["phase"] == "qualification" and output["frozen"] is True
            ),
        },
        **provenance,
    }
    if (
        not receipt["reference_contract_unchanged"]
        or not receipt["sampling_contract_unchanged"]
    ):
        raise AssertionError("reference materialization changed its accuracy contract")
    return output, receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--policy-config", type=Path, required=True)
    parser.add_argument("--proposal-training-source", type=Path, required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    arguments = parser.parse_args()
    receipt_path = arguments.receipt or arguments.output.with_suffix(
        arguments.output.suffix + ".receipt.json"
    )
    if arguments.output.exists() or receipt_path.exists():
        raise FileExistsError("reference materialization refuses existing outputs")
    config, receipt = materialize_reference_config(
        arguments.base_config,
        arguments.policy_config,
        arguments.proposal_training_source,
        protocol_id=arguments.protocol_id,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(_yaml_bytes(config))
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
