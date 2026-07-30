"""Atomic non-overwriting storage for reference shards."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from .reference_protocol import (
    ReferenceShardIdentity,
    canonical_json_bytes,
    canonical_sha256,
    validate_shard_artifact,
)


def shard_path(directory: Path, identity: ReferenceShardIdentity) -> Path:
    return directory / f"{identity.shard_id}.json"


def write_json_atomic_nonoverwriting(path: Path, payload: object) -> str:
    """Atomically link a complete canonical file into place without overwriting."""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite completed artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    data = canonical_json_bytes(payload) + b"\n"
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite completed artifact: {path}"
            ) from error
    finally:
        if temporary.exists():
            temporary.unlink()
    return canonical_sha256(payload)


def write_shard_atomic(directory: Path, payload: dict[str, Any]) -> tuple[Path, str]:
    validate_shard_artifact(payload)
    identity = ReferenceShardIdentity.from_dict(payload["identity"])
    path = shard_path(directory, identity)
    digest = write_json_atomic_nonoverwriting(path, payload)
    return path, digest


def load_shard(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("ascii"))
    validate_shard_artifact(payload)
    return payload, canonical_sha256(payload)


def find_completed_shards(directory: Path) -> dict[str, tuple[Path, dict[str, Any], str]]:
    completed: dict[str, tuple[Path, dict[str, Any], str]] = {}
    if not directory.exists():
        return completed
    for path in sorted(directory.glob("*.json")):
        payload, digest = load_shard(path)
        shard_id = payload["shard_id"]
        if shard_id in completed:
            raise ValueError(f"duplicate completed shard id: {shard_id}")
        completed[shard_id] = (path, payload, digest)
    return completed
