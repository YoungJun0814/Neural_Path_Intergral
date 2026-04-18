"""Thin MLflow integration wrapper.

MLflow is an optional dependency: if it is not installed, these helpers
no-op and still return a usable object.  This keeps the core codebase
portable while letting users opt in to experiment tracking.
"""
from __future__ import annotations

import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

try:  # optional
    import mlflow  # type: ignore
    _HAS_MLFLOW = True
except ImportError:  # pragma: no cover
    mlflow = None
    _HAS_MLFLOW = False


class RunLogger:
    """Unified logger: writes to MLflow if available, else to a JSONL file.

    Example::

        with RunLogger(experiment="phase2-heston").run(name="mc_baseline") as run:
            run.log_params({"S0": 100, "K": 90})
            for step, loss in enumerate(losses):
                run.log_metric("loss", loss, step=step)
            run.log_artifact_text("summary.txt", "OK")
    """

    def __init__(self, experiment: str, out_dir: str | Path = "mlruns_fallback"):
        self.experiment = experiment
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if _HAS_MLFLOW:
            mlflow.set_experiment(experiment)

    @contextlib.contextmanager
    def run(self, name: str) -> Iterator["RunHandle"]:
        if _HAS_MLFLOW:
            with mlflow.start_run(run_name=name):
                yield MLflowHandle()
        else:
            run_dir = self.out_dir / f"{name}_{int(time.time())}"
            run_dir.mkdir(parents=True, exist_ok=True)
            yield FileHandle(run_dir)


class RunHandle:
    def log_params(self, params: Dict[str, Any]) -> None: ...
    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None: ...
    def log_artifact_text(self, filename: str, content: str) -> None: ...


class MLflowHandle(RunHandle):
    def log_params(self, params: Dict[str, Any]) -> None:
        for k, v in params.items():
            mlflow.log_param(k, v)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        mlflow.log_metric(key, float(value), step=step)

    def log_artifact_text(self, filename: str, content: str) -> None:
        tmp = Path(mlflow.get_artifact_uri().replace("file://", "")) / filename
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(content, encoding="utf-8")


class FileHandle(RunHandle):
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.metrics_path = run_dir / "metrics.jsonl"
        self.params_path = run_dir / "params.json"

    def log_params(self, params: Dict[str, Any]) -> None:
        self.params_path.write_text(json.dumps(params, indent=2), encoding="utf-8")

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "value": float(value), "step": step, "t": time.time()}) + "\n")

    def log_artifact_text(self, filename: str, content: str) -> None:
        (self.run_dir / filename).write_text(content, encoding="utf-8")


__all__ = ["RunLogger", "RunHandle"]
