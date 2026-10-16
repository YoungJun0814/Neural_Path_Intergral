"""Append-only complete-unit checkpoints with conservative interrupted budgets."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .reference_protocol import canonical_sha256
from .reference_shards import write_json_atomic_nonoverwriting


class RecoveryJournal:
    """No seed replacement, partial-unit inference, or changed-source resume."""

    def __init__(self, directory: Path, manifest: dict[str, Any]) -> None:
        for limit in manifest["limits"].values():
            if (isinstance(limit["work"], bool) or not isinstance(limit["work"], int) or limit["work"] < 0
                    or isinstance(limit["wall"], bool) or not math.isfinite(limit["wall"]) or limit["wall"] <= 0):
                raise ValueError("invalid checkpoint phase limits")
        self.directory = directory
        self.digest = canonical_sha256(manifest)
        path = directory / "manifest.json"
        if path.exists():
            if json.loads(path.read_text()) != manifest:
                raise ValueError("resume manifest/source/config mismatch")
        else:
            write_json_atomic_nonoverwriting(path, manifest)
        self.limits = manifest["limits"]
        self._budget_cache: dict[Path, tuple[int, int, str, int, float]] = {}

    def _read(self, path: Path) -> dict[str, Any]:
        envelope = json.loads(path.read_text())
        value = envelope["payload"]
        if envelope["sha256"] != canonical_sha256(value) or value["manifest"] != self.digest:
            raise ValueError("corrupt checkpoint or identity mismatch")
        return value

    def _write(self, path: Path, value: dict[str, Any]) -> None:
        write_json_atomic_nonoverwriting(path, {"payload": value, "sha256": canonical_sha256(value)})

    def accounting(self, phase: str) -> tuple[int, float]:
        work, wall = 0, 0.
        for start in sorted(self.directory.glob("*.start.json")):
            reservation_phase, reserved_work, reserved_wall = self._budget_fields(start)
            if reservation_phase != phase:
                continue
            finish = start.with_name(start.name.replace(".start.json", ".finish.json"))
            if finish.exists():
                _, actual_work, actual_wall = self._budget_fields(finish)
                work += actual_work
                wall += actual_wall
            else:
                work += reserved_work
                wall += reserved_wall
        return work, wall

    def _budget_fields(self, path: Path) -> tuple[str, int, float]:
        """Cache only small validated metadata, never mutable sample payloads."""
        stat = path.stat()
        old = self._budget_cache.get(path)
        if old is not None and old[:2] == (stat.st_mtime_ns, stat.st_size):
            return old[2:]
        record = self._read(path)
        work = record["work"] if "work" in record else record["reserved_work"]
        wall = float(record["wall"] if "wall" in record else record["reserved_wall"])
        if isinstance(work, bool) or not isinstance(work, int) or work < 0 or not math.isfinite(wall) or wall < 0:
            raise ValueError("invalid saved budget metadata")
        fields = (str(record["phase"]), work, wall)
        self._budget_cache[path] = (stat.st_mtime_ns, stat.st_size, *fields)
        return fields

    def unit(self, unit_id: str, *, phase: str, work: int, maximum_wall: float,
             operation: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        if not unit_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in unit_id):
            raise ValueError("unsafe checkpoint unit ID")
        violation_path = self.directory/f"phase-{phase}.violation.json"
        if violation_path.exists():
            self._read(violation_path)
            raise RuntimeError("phase failed; completed prefix cannot be qualified")
        start, finish = (self.directory / f"{unit_id}.{suffix}.json" for suffix in ("start", "finish"))
        if finish.exists():
            value = self._read(finish)
            if value["phase"] != phase or value.get("reserved_work", value["work"]) != work:
                raise ValueError("checkpoint unit specification changed")
            if value["status"] != "complete":
                raise RuntimeError("failed unit cannot be selectively replaced")
            return value["result"]
        if start.exists():
            self._read(start)
            raise RuntimeError("interrupted unit: preserve reservation; new protocol required")
        if (isinstance(work, bool) or not isinstance(work, int) or work < 0
                or isinstance(maximum_wall, bool) or not math.isfinite(maximum_wall) or maximum_wall <= 0):
            raise ValueError("invalid unit reservation")
        used, elapsed = self.accounting(phase)
        limit = self.limits[phase]
        clock_path = self.directory/f"phase-{phase}.clock.json"
        if not clock_path.exists():
            self._write(clock_path, {"manifest": self.digest, "started_unix": time.time()})
        phase_elapsed = time.time()-self._read(clock_path)["started_unix"]
        # Includes checkpoint parsing/writes and pauses; conservative on restart.
        if phase_elapsed + maximum_wall > limit["wall"]:
            raise TimeoutError("whole phase wall budget exhausted including I/O")
        if used + work > limit["work"] or elapsed + maximum_wall > limit["wall"]:
            raise TimeoutError("cumulative phase budget exhausted")
        self._write(start, {"manifest": self.digest, "phase": phase,
                            "reserved_work": work, "reserved_wall": maximum_wall})
        timer = time.perf_counter()
        try:
            result = operation()
            actual_work = result.get("actual_work", work)
            if isinstance(actual_work, bool) or not isinstance(actual_work, int) or not 0 <= actual_work <= work:
                raise ValueError("actual work exceeds immutable unit reservation")
            elapsed = time.perf_counter() - timer
            if elapsed > maximum_wall:
                raise TimeoutError("complete unit exceeded wall reservation")
            if time.time()-self._read(clock_path)["started_unix"] > limit["wall"]:
                raise TimeoutError("whole phase wall exceeded during complete unit")
        except Exception as error:
            self._write(finish, {"manifest": self.digest, "phase": phase, "status": "protocol_failure",
                                 "work": work, "wall": time.perf_counter() - timer,
                                 "error": type(error).__name__ + ": " + str(error)})
            raise
        self._write(finish, {"manifest": self.digest, "phase": phase, "status": "complete",
                             "work": actual_work, "reserved_work": work, "wall": elapsed, "result": result})
        if time.time()-self._read(clock_path)["started_unix"] > limit["wall"]:
            self._write(violation_path, {"manifest": self.digest, "phase": phase,
                                        "status": "protocol_failure_after_checkpoint_io"})
            raise TimeoutError("whole phase wall exceeded in checkpoint I/O")
        return result
