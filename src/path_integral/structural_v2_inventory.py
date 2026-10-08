"""Bounded-memory historical evidence inventory, not a numerical replay."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, TextIO

from src.path_integral.research_result_contract import canonical_digest
from src.path_integral.seed_ledger import SeedLedger


def file_sha256(path: Path, *, decompressed: bool = False) -> tuple[str, int]:
    digest, size = hashlib.sha256(), 0
    opener = gzip.open if decompressed else open
    with opener(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def workspace_file(root: Path, name: str) -> Path:
    path = (root/name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"missing or out-of-workspace evidence: {name}")
    return path


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _nonfinite(token: str) -> None:
    raise ValueError(f"nonfinite JSON number: {token}")


class _Reader:
    def __init__(self, handle: TextIO, maximum_value_chars: int, chunk_size: int) -> None:
        self.handle, self.cap, self.chunk = handle, maximum_value_chars, chunk_size
        self.buffer, self.pos, self.eof = "", 0, False
        self.decoder = json.JSONDecoder(object_pairs_hook=_unique_pairs, parse_constant=_nonfinite)

    def fill(self) -> None:
        self.buffer, self.pos = self.buffer[self.pos:], 0
        if len(self.buffer) >= self.cap:
            raise ValueError("JSON value exceeds bounded-memory limit")
        extra = self.handle.read(min(self.cap-len(self.buffer), max(self.chunk, len(self.buffer))))
        self.buffer += extra
        self.eof = not extra

    def peek(self) -> str:
        while True:
            while self.pos < len(self.buffer) and self.buffer[self.pos].isspace():
                self.pos += 1
            if self.pos < len(self.buffer):
                return self.buffer[self.pos]
            if self.eof:
                return ""
            self.fill()

    def take(self, token: str) -> None:
        if self.peek() != token:
            raise ValueError(f"expected JSON token {token!r}")
        self.pos += 1

    def value(self) -> Any:
        self.peek()
        self.buffer, self.pos = self.buffer[self.pos:], 0
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer)
            except json.JSONDecodeError as error:
                if self.eof:
                    raise ValueError("truncated/invalid JSON value") from error
                self.fill()
                continue
            if end == len(self.buffer) and not self.eof:
                self.fill()
                continue
            if end < len(self.buffer) and self.buffer[end] not in ":,]} \t\r\n":
                if isinstance(value, (int, float)) and not isinstance(value, bool) and not self.eof:
                    self.fill()
                    continue
                raise ValueError("invalid JSON value delimiter")
            self.pos = end
            return value


def stream_evidence(path: Path, *, maximum_value_chars: int = 64 * 1024 * 1024,
                    chunk_size: int = 65536) -> Iterator[tuple[str, Any]]:
    """Yield metadata and individual records; never load the full records array."""
    if maximum_value_chars < 1 or chunk_size < 1:
        raise ValueError("invalid streaming limits")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        reader = _Reader(handle, maximum_value_chars, chunk_size)
        reader.take("{")
        seen: set[str] = set()
        if reader.peek() != "}":
            while True:
                key = reader.value()
                if not isinstance(key, str) or key in seen:
                    raise ValueError("invalid/duplicate top-level field")
                seen.add(key)
                reader.take(":")
                if key == "records":
                    reader.take("[")
                    if reader.peek() != "]":
                        while True:
                            record = reader.value()
                            if not isinstance(record, dict):
                                raise ValueError("record must be an object")
                            yield "record", record
                            if reader.peek() == "]":
                                break
                            reader.take(",")
                    reader.take("]")
                else:
                    yield key, reader.value()
                if reader.peek() == "}":
                    break
                reader.take(",")
        reader.take("}")
        if reader.peek():
            raise ValueError("trailing JSON content")


def audit_snapshot(root: Path, source: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    if canonical_digest(config) != source["config_digest"]:
        raise ValueError("historical configuration binding mismatch")
    path = workspace_file(root, source["snapshot_path"])
    sha, _ = file_sha256(path)
    if sha != source["snapshot_sha256"]:
        raise ValueError("historical source archive changed")
    expected = source["snapshot_file_hashes"]
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(expected) | {"RUN_CONFIG.json"}:
            raise ValueError("source archive completion/duplicate mismatch")
        if json.loads(archive.read("RUN_CONFIG.json")) != config:
            raise ValueError("archived configuration changed")
        for name, digest in expected.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError("archived source entry changed")
    return {"path": source["snapshot_path"], "sha256": sha, "entries": len(expected)}


def _proposal_binding(parameters: dict[str, Any], digest: str) -> None:
    if canonical_digest(parameters) != digest:
        raise ValueError("historical proposal binding mismatch")
    components, weights = parameters["components"], parameters["weights"]
    if (not components or len(components) != len(weights)
            or any(isinstance(w, bool) or not isinstance(w, (int, float)) or w <= 0 for w in weights)
            or not math.isclose(sum(weights), 1., abs_tol=1e-12, rel_tol=0)):
        raise ValueError("invalid saved mixture weights")
    natural = sum(w for w, c in zip(weights, components, strict=True)
                  if not c["eigenvalues"] and c["mean"] and all(x == 0 for x in c["mean"]))
    if natural < .1 - 1e-12:
        raise ValueError("historical exact natural floor lost")


def scan_legacy(root: Path, relative: str, *, current_source_digest: str,
                check_limits: Callable[[], None] | None = None) -> dict[str, Any]:
    """Refresh hash/snapshot/parameter/completion bindings, not all numerical audits."""
    path = workspace_file(root, relative)
    metadata: dict[str, Any] = {}
    identities: set[tuple[Any, ...]] = set()
    parent_bindings: dict[tuple[str, int], str] = {}
    record_count = estimator_count = candidate_count = 0
    work_sum = 0
    statuses: dict[str, int] = {}
    for name, value in stream_evidence(path):
        if check_limits is not None:
            check_limits()
        if name != "record":
            metadata[name] = value
            continue
        record_count += 1
        identity = (value["cell"]["id"], value["parent_training_rep"],
                    value.get("auxiliary_training_rep", 0), value.get("method", ""))
        if identity in identities:
            raise ValueError("duplicate historical record identity")
        identities.add(identity)
        statuses[value.get("status", "not_declared")] = statuses.get(value.get("status", "not_declared"), 0)+1
        work_sum += value.get("potential_evaluations", 0)
        for item in value.get("candidates", []):
            candidate_count += 1
            if "proposal_parameters" in item:
                _proposal_binding(item["proposal_parameters"], item["proposal_digest"])
        for item in value.get("estimators", []):
            estimator_count += 1
            _proposal_binding(item["r_parameters"], item["r_digest"])
            if item["id"] == "direct-q" and item["r_digest"] != value["q_digest"]:
                raise ValueError("direct-q binding mismatch")
        if "q_digest" in value:
            parent = (identity[0], identity[1])
            if parent in parent_bindings and parent_bindings[parent] != value["q_digest"]:
                raise ValueError("fixed parent q changed between auxiliary fits")
            parent_bindings[parent] = value["q_digest"]
    if not record_count or "source" not in metadata or "config" not in metadata:
        raise ValueError("missing historical records/source/config")
    config, source = metadata["config"], metadata["source"]
    snapshot = audit_snapshot(root, source, config)
    ledger = SeedLedger.from_dict(metadata["seed_ledger"])
    dependencies = []
    for field, sha_field in (("reference_path", "reference_sha256"),
                             ("guide_artifact_path", "guide_artifact_sha256")):
        if config.get(field):
            dependency = workspace_file(root, config[field])
            dependency_sha, dependency_size = file_sha256(dependency)
            if dependency_sha != metadata[sha_field]:
                raise ValueError("historical reference/guide dependency changed")
            dependencies.append({"path": config[field], "sha256": dependency_sha,
                                 "bytes": dependency_size})
    if "proposal_artifact_path" in config:
        parent_path = workspace_file(root, config["proposal_artifact_path"])
        parent_sha, _ = file_sha256(parent_path)
        if parent_sha != metadata["proposal_artifact_sha256"]:
            raise ValueError("parent source artifact changed")
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        declared = {(r["cell"]["id"], r["parent_training_rep"]): next(
            c["proposal_digest"] for c in r["candidates"] if c["method"] == config["method"])
                    for r in parent["records"]}
        if parent_bindings and parent_bindings != declared:
            raise ValueError("parent q completion/digest mismatch")
        repetitions = config.get("training_repetitions", 1)
        expected = {(cell, rep, aux, "") for cell, rep in declared for aux in range(repetitions)}
        if metadata["schema"] == "npi.post-audit.r2-aux-risk.v1" and identities != expected:
            raise ValueError("auxiliary fit completion mismatch")
    if metadata["schema"] == "npi.post-audit.r2-family-diagnosis.v1":
        expected = {(c["id"], rep, 0, "") for c in config["cells"]
                    for rep in range(config["parent_training_replicates"])}
        if identities != expected:
            raise ValueError("whole parent completion mismatch")
    if work_sum and work_sum != metadata["potential_evaluations"]:
        raise ValueError("lost historical row work")
    if metadata.get("model_q_changed") or metadata.get("performance_claim_authorized"):
        raise ValueError("historical auxiliary evidence has unauthorized performance flags")
    sha, size = file_sha256(path)
    raw_sha, raw_size = file_sha256(path, decompressed=True) if path.suffix == ".gz" else (sha, size)
    raw_copy = path.with_suffix("") if path.suffix == ".gz" else None
    raw_copy_checked = bool(raw_copy and raw_copy.exists())
    if raw_copy is not None and raw_copy_checked and file_sha256(raw_copy) != (raw_sha, raw_size):
        raise ValueError("gzip/raw evidence mismatch")
    return {"path": relative, "schema": metadata["schema"], "sha256": sha, "bytes": size,
            "raw_sha256": raw_sha, "raw_bytes": raw_size, "raw_copy_roundtrip_checked": raw_copy_checked,
            "source_commit": source["source_commit"], "historical_source_digest": source["source_tree_digest"],
            "current_runtime_source_matches": source["source_tree_digest"] == current_source_digest,
            "snapshot": snapshot, "config_digest": source["config_digest"], "records": record_count,
            "dependencies": dependencies,
            "candidates": candidate_count, "estimators": estimator_count, "record_statuses": statuses,
            "seed_streams": len(ledger), "potential_evaluations": metadata["potential_evaluations"],
            "audit_scope": "hash_snapshot_config_proposal_floor_seed_and_completion_binding",
            "full_numerical_replay_performed": False, "statistical_certificate": False}
