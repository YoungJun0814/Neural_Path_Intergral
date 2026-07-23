"""Build a deterministic, hash-verified V6 publication archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import yaml

from experiments.g11_v6_routed_policy import _task_conditioned_training_source_audit
from src.path_integral.provenance import source_provenance

_FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=_FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def build_publication_archive(
    output_path: Path,
    *,
    policy_config_path: Path,
    proposal_training_source_path: Path,
    additional_files: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Archive the V3 proposal bank, raw source, and declared evidence by raw hash."""

    if output_path.exists():
        raise FileExistsError("publication archive refuses to overwrite an existing file")
    policy_raw = policy_config_path.read_bytes()
    policy = yaml.safe_load(policy_raw)
    if (
        not isinstance(policy, dict)
        or policy.get("schema") != "npi.g11.v6-routed-policy.config.v3"
    ):
        raise ValueError("publication archive requires a V3 routed-policy config")
    training_audit = _task_conditioned_training_source_audit(
        policy["proposal"], proposal_training_source_path
    )
    files = {
        "policy_config": policy_config_path,
        "proposal_training_source": proposal_training_source_path,
        **(additional_files or {}),
    }
    if len(files) != 2 + len(additional_files or {}):
        raise ValueError("publication archive roles must be unique")
    if any(
        not role
        or role.strip() != role
        or "/" in role
        or "\\" in role
        for role in files
    ):
        raise ValueError("publication archive roles must be safe stripped names")
    resolved = [path.resolve() for path in files.values()]
    if len(set(resolved)) != len(resolved):
        raise ValueError("one source file cannot be archived under multiple roles")

    entries = []
    raw_by_entry: dict[str, bytes] = {}
    for role, path in sorted(files.items()):
        raw = path.read_bytes()
        archive_name = f"files/{role}/{path.name}"
        raw_by_entry[archive_name] = raw
        entries.append(
            {
                "role": role,
                "archive_path": archive_name,
                "source_basename": path.name,
                "sha256": _sha256(raw),
                "bytes": len(raw),
            }
        )
    provenance = source_provenance()
    manifest = {
        "schema": "npi.g11.v6-publication-archive-manifest.v1",
        "source_commit": provenance["source_commit"],
        "dirty_worktree": provenance["dirty_worktree"],
        "policy_schema": policy["schema"],
        "policy_protocol_id": policy["protocol_id"],
        "proposal_training_audit": training_audit,
        "entries": entries,
    }
    manifest_raw = json.dumps(
        manifest, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "x") as archive:
        archive.writestr(_zip_info("MANIFEST.json"), manifest_raw)
        for archive_name in sorted(raw_by_entry):
            archive.writestr(_zip_info(archive_name), raw_by_entry[archive_name])

    with zipfile.ZipFile(output_path, "r") as archive:
        if archive.read("MANIFEST.json") != manifest_raw:
            raise AssertionError("publication archive manifest replay failed")
        for entry in entries:
            if _sha256(archive.read(entry["archive_path"])) != entry["sha256"]:
                raise AssertionError(
                    f"publication archive replay failed for {entry['role']}"
                )
    return {
        "schema": "npi.g11.v6-publication-archive-receipt.v1",
        "archive_sha256": _sha256(output_path.read_bytes()),
        "manifest_sha256": _sha256(manifest_raw),
        "entry_count": len(entries),
        "proposal_training_source_verified": training_audit["verified"],
        "proposal_training_source_formal": training_audit[
            "formal_training_source_readiness"
        ],
        "source_commit": provenance["source_commit"],
        "dirty_worktree": provenance["dirty_worktree"],
        "output": str(output_path),
    }


def _parse_file(value: str) -> tuple[str, Path]:
    role, separator, path = value.partition("=")
    if not separator or not role or not path:
        raise argparse.ArgumentTypeError("file must be ROLE=PATH")
    return role, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-config", type=Path, required=True)
    parser.add_argument("--proposal-training-source", type=Path, required=True)
    parser.add_argument("--file", action="append", type=_parse_file, default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    arguments = parser.parse_args()
    additional = dict(arguments.file)
    if len(additional) != len(arguments.file):
        raise ValueError("additional publication archive roles must be unique")
    receipt_path = arguments.receipt or arguments.output.with_suffix(
        arguments.output.suffix + ".receipt.json"
    )
    if arguments.output.exists() or receipt_path.exists():
        raise FileExistsError(
            "publication archive refuses to overwrite an archive or receipt"
        )
    receipt = build_publication_archive(
        arguments.output,
        policy_config_path=arguments.policy_config,
        proposal_training_source_path=arguments.proposal_training_source,
        additional_files=additional,
    )
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
