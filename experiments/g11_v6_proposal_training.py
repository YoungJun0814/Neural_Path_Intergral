"""Reference-free, development-only CEM training for the V6 proposal bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from experiments.g11_v6_reference import _load_manifest
from src.path_integral import (
    DiscreteBarrierHitTask,
    SeedKey,
    SeedLedger,
    TerminalThresholdTask,
    V6WorkLedger,
    V6WorkRecord,
)
from src.path_integral.provenance import runtime_provenance, source_provenance
from src.physics_engine import RBergomiSimulator
from src.training import fit_rbergomi_piecewise_cem

_SCHEMA = "npi.g11.v6-proposal-training.config.v1"


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _finite_real(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite real number")
    return result


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw)
    expected = {
        "schema",
        "protocol_id",
        "phase",
        "frozen",
        "estimand",
        "selected_cell_ids",
        "clusters",
        "training",
    }
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != _SCHEMA
        or set(payload) != expected
    ):
        raise ValueError("malformed V6 proposal-training config")
    if payload["phase"] != "development" or payload["frozen"] is not False:
        raise ValueError("proposal training must be unfrozen development work")
    if payload["estimand"] != "fixed_finest_grid":
        raise ValueError("proposal training must declare the fixed-grid estimand")
    protocol = payload["protocol_id"]
    if (
        not isinstance(protocol, str)
        or not protocol.startswith("g11-v6-")
        or protocol.strip() != protocol
    ):
        raise ValueError("proposal-training protocol id is invalid")
    selected = payload["selected_cell_ids"]
    if (
        not isinstance(selected, list)
        or len(selected) != 2
        or len(set(selected)) != 2
        or any(
            not isinstance(cell_id, str)
            or not cell_id
            or cell_id.strip() != cell_id
            for cell_id in selected
        )
    ):
        raise ValueError("proposal training requires two unique selected cell ids")
    _positive_integer(payload["clusters"], "clusters")

    training = payload["training"]
    training_fields = {
        "segments",
        "initial_control",
        "paths_per_iteration",
        "maximum_iterations",
        "elite_quantile",
        "smoothing",
        "minimum_elite_paths",
        "control_bound",
        "target_level_repetitions",
    }
    if not isinstance(training, dict) or set(training) != training_fields:
        raise ValueError("malformed proposal-training hyperparameters")
    segments = _positive_integer(training["segments"], "training.segments")
    initial = training["initial_control"]
    if (
        not isinstance(initial, list)
        or len(initial) != segments
        or any(
            not isinstance(segment, list)
            or len(segment) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in segment
            )
            for segment in initial
        )
    ):
        raise ValueError("initial control must contain finite two-driver segments")
    paths = _positive_integer(
        training["paths_per_iteration"], "training.paths_per_iteration"
    )
    _positive_integer(training["maximum_iterations"], "training.maximum_iterations")
    elite_quantile = _finite_real(
        training["elite_quantile"], "training.elite_quantile"
    )
    if not 0.5 < elite_quantile < 1.0:
        raise ValueError("elite quantile must select an upper-tail minority")
    smoothing = _finite_real(training["smoothing"], "training.smoothing")
    if not 0.0 < smoothing <= 1.0:
        raise ValueError("training smoothing must lie in (0, 1]")
    minimum_elite = _positive_integer(
        training["minimum_elite_paths"], "training.minimum_elite_paths"
    )
    if minimum_elite > paths:
        raise ValueError("minimum elite paths exceeds paths per iteration")
    if _finite_real(training["control_bound"], "training.control_bound") <= 0.0:
        raise ValueError("control bound must be positive")
    _positive_integer(
        training["target_level_repetitions"],
        "training.target_level_repetitions",
    )
    return payload, hashlib.sha256(raw).hexdigest()


def _task(cell):
    if cell.task == "terminal_left_tail":
        return TerminalThresholdTask(cell.event_threshold)
    return DiscreteBarrierHitTask(cell.event_threshold)


def _fit(config, cell, seed: int, *, smoke: bool):
    training = config["training"]
    paths = 256 if smoke else int(training["paths_per_iteration"])
    iterations = 2 if smoke else int(training["maximum_iterations"])
    simulator = RBergomiSimulator(
        H=cell.hurst,
        eta=cell.eta,
        xi=cell.xi,
        rho=cell.rho,
        device="cpu",
    )
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    fit = fit_rbergomi_piecewise_cem(
        simulator,
        _task(cell),
        spot=cell.spot,
        maturity=cell.maturity,
        dt=cell.maturity / cell.finest_steps,
        initial_control=tuple(
            tuple(float(value) for value in segment)
            for segment in training["initial_control"]
        ),
        num_paths=paths,
        seed=seed,
        max_iterations=iterations,
        elite_quantile=float(training["elite_quantile"]),
        smoothing=float(training["smoothing"]),
        min_elite_paths=min(int(training["minimum_elite_paths"]), paths),
        control_bound=float(training["control_bound"]),
        target_level_repetitions=int(training["target_level_repetitions"]),
    )
    wall_seconds = time.perf_counter() - wall_started
    cpu_seconds = time.process_time() - cpu_started
    operation_scale = cell.finest_steps * max(1.0, math.log2(cell.finest_steps))
    work = V6WorkRecord(
        category="proposal_training",
        method="pure_cem",
        cell_id=cell.cell_id,
        attempt=0,
        samples=paths * len(fit.history),
        work_units=paths * len(fit.history) * operation_scale,
        wall_seconds=wall_seconds,
        cpu_seconds=cpu_seconds,
        peak_memory_bytes=0,
        successful=True,
    )
    return fit, work


def run(
    config_path: Path,
    manifest_path: Path,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    config, config_hash = _load_config(config_path)
    manifest = _load_manifest(manifest_path)
    if not smoke and (
        manifest.phase != "development"
        or manifest.frozen
        or manifest.dirty_tree
        or manifest.smoke
        or manifest.source_commit == "uncommitted"
    ):
        raise ValueError(
            "formal proposal training requires a clean committed development manifest"
        )
    selected_ids = tuple(str(value) for value in config["selected_cell_ids"])
    available = {cell.cell_id: cell for cell in manifest.cells}
    missing = sorted(set(selected_ids) - set(available))
    if missing:
        raise ValueError(f"proposal-training cells are absent from the manifest: {missing}")
    cells = tuple(available[cell_id] for cell_id in selected_ids)
    if {cell.task for cell in cells} != {
        "terminal_left_tail",
        "discrete_lower_barrier",
    }:
        raise ValueError("proposal-training cells must cover both task families exactly")

    clusters = 1 if smoke else int(config["clusters"])
    ledger = SeedLedger()
    work = V6WorkLedger()
    records: list[dict[str, Any]] = []
    for cell in cells:
        for cluster in range(clusters):
            key = SeedKey(
                str(config["protocol_id"]),
                "proposal-training",
                f"{cell.cell_id}:cluster-{cluster}",
                "pure_cem",
                0,
                0,
                "proposal",
            )
            seed = ledger.allocate(key)
            fit, work_record = _fit(config, cell, seed, smoke=smoke)
            work = work.append(work_record)
            records.append(
                {
                    "cell_id": cell.cell_id,
                    "task": cell.task,
                    "cluster": cluster,
                    "method": "pure_cem",
                    "seed_key": asdict(key),
                    "seed": seed,
                    "cem_fit": {
                        "converged": fit.converged,
                        "iterations": len(fit.history),
                        "control": fit.control,
                        "history": [asdict(item) for item in fit.history],
                    },
                    "training_work_record": asdict(work_record),
                }
            )

    bound = float(config["training"]["control_bound"])
    gates = {
        "complete_matrix": len(records) == len(cells) * clusters,
        "both_task_families_covered": {
            str(record["task"]) for record in records
        }
        == {"terminal_left_tail", "discrete_lower_barrier"},
        "all_cem_fits_converged": all(
            bool(record["cem_fit"]["converged"]) for record in records
        ),
        "all_controls_finite_and_bounded": all(
            math.isfinite(float(value)) and abs(float(value)) <= bound
            for record in records
            for segment in record["cem_fit"]["control"]
            for value in segment
        ),
        "all_histories_nonempty": all(
            int(record["cem_fit"]["iterations"]) > 0 for record in records
        ),
        "one_successful_training_record_per_fit": (
            len(work.records) == len(records)
            and all(
                record.category == "proposal_training" and record.successful
                for record in work.records
            )
        ),
        "canonical_seed_count_matches": len(ledger) == len(records),
    }
    provenance = source_provenance()
    formal = {
        "clean_source": not bool(provenance["dirty_worktree"]),
        "non_smoke": not smoke,
        "clean_committed_development_manifest": (
            manifest.phase == "development"
            and not manifest.frozen
            and not manifest.dirty_tree
            and not manifest.smoke
            and manifest.source_commit != "uncommitted"
        ),
        "reference_free_training_contract": True,
    }
    return {
        "schema": "npi.g11.v6-proposal-training.v1",
        "protocol_id": config["protocol_id"],
        "config_sha256": config_hash,
        "manifest_sha256": manifest.sha256,
        "manifest_source_commit": manifest.source_commit,
        "selected_cell_ids": list(selected_ids),
        "clusters": clusters,
        "smoke": smoke,
        "records": records,
        "work_ledger": work.to_dict(),
        "work_ledger_sha256": work.sha256,
        "seed_ledger": ledger.to_dict(),
        "seed_ledger_sha256": ledger.sha256,
        "gates": gates,
        "formal_readiness": formal,
        "proposal_training_qualified": all(gates.values()) and all(formal.values()),
        "environment": runtime_provenance(dtype="torch.float64"),
        **provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/g11_v6/proposal_training_development_v1.yaml"),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError("proposal training refuses to overwrite an artifact")
    result = run(arguments.config, arguments.manifest, smoke=arguments.smoke)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "qualified": result["proposal_training_qualified"],
                **result["gates"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
