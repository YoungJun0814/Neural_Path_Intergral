"""Shared source and runtime provenance for reproducible G11 artifacts."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
from typing import Any

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
    for name in ("numpy", "scipy", "PyYAML"):
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
