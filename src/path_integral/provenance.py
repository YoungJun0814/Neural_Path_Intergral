"""Shared source and runtime provenance for reproducible G11 artifacts."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
from typing import Any

import psutil
import torch


def source_provenance() -> dict[str, Any]:
    """Return the exact Git source state without mutating the repository."""

    try:
        commit = subprocess.check_output(
            ("git", "rev-parse", "HEAD"), text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ("git", "status", "--porcelain"), text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unavailable", True
    return {"source_commit": commit, "dirty_worktree": dirty}


def runtime_provenance(*, dtype: str) -> dict[str, Any]:
    """Return the environment fields required by the frozen result schema."""

    packages = {}
    for name in ("numpy", "scipy", "PyYAML", "psutil"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "unavailable"
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "os": platform.platform(),
        "processor": platform.processor(),
        "torch_threads": torch.get_num_threads(),
        "dtype": dtype,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "packages": packages,
    }


def process_peak_resident_memory_bytes() -> int:
    """Return the OS-reported lifetime peak resident set for this process."""

    memory = psutil.Process().memory_info()
    peak = getattr(memory, "peak_wset", None)
    if peak is None:
        # Linux psutil does not expose a lifetime peak.  ru_maxrss is KiB on
        # Linux and bytes on macOS.
        try:
            import resource

            getrusage = getattr(resource, "getrusage", None)
            process_usage = getattr(resource, "RUSAGE_SELF", None)
            if not callable(getrusage) or process_usage is None:
                raise AttributeError("resource peak-RSS API is unavailable")
            maximum = int(getrusage(process_usage).ru_maxrss)
            peak = maximum if platform.system() == "Darwin" else maximum * 1024
        except (AttributeError, ImportError, OSError, ValueError):
            peak = int(memory.rss)
    result = int(peak)
    if result < 0:
        raise RuntimeError("operating system returned a negative peak resident set")
    return result
