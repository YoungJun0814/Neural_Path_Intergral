"""Training-inclusive, paired-seed development comparison for G11 full MLMC."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from statistics import geometric_mean
from typing import Any

import torch
import yaml

from src.path_integral import (
    DiscreteBarrierHitTask,
    DownsideExcursionTask,
    FixedFinestGridTarget,
    MLMCHierarchy,
    RBergomiMLMCSampler,
    RBergomiMLMCSamplerConfig,
    TerminalThresholdTask,
    TimePiecewiseTwoDriverControl,
    WorkLedgerEntry,
    execute_mlmc,
    prepare_mlmc,
)
from src.path_integral.provenance import runtime_provenance, source_provenance
from src.physics_engine import RBergomiSimulator


def _load(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("expected a G11 MLMC-development schema-version-1 config")
    if config.get("frozen") is not False or config.get("estimand") != "finite_grid":
        raise ValueError("development config must be unfrozen and finite-grid explicit")
    return config, hashlib.sha256(raw).hexdigest()


def _task(specification: dict[str, Any]):
    if specification["kind"] == "terminal":
        return TerminalThresholdTask(float(specification["level"]))
    if specification["kind"] == "barrier":
        return DiscreteBarrierHitTask(float(specification["barrier"]))
    if specification["kind"] == "hit_plus_occupation":
        return DownsideExcursionTask(
            hit_barrier=float(specification["hit_barrier"]),
            stress_level=float(specification["stress_level"]),
            minimum_occupation=float(specification["minimum_occupation"]),
            hit_scale=float(specification["hit_scale"]),
            occupation_scale=float(specification["occupation_scale"]),
        )
    raise ValueError("unsupported development task")


def _write_progress(
    path: Path,
    *,
    config_hash: str,
    smoke: bool,
    cells: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    payload = {
        "schema": "npi.g11.mlmc-development-progress.v1",
        "config_sha256": config_hash,
        "smoke": smoke,
        "cells": cells,
        "failures": failures,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def run(
    config_path: Path, *, smoke: bool, progress_path: Path | None = None
) -> dict[str, Any]:
    started = time.perf_counter()
    config, config_hash = _load(config_path)
    input_artifacts: list[dict[str, str]] = []
    for declaration in config.get("input_artifacts", []):
        path = Path(declaration["path"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != declaration["sha256"]:
            raise ValueError(f"input artifact hash mismatch: {path}")
        input_artifacts.append({"path": str(path), "sha256": digest})
    model = config["model"]
    hierarchy_config = config["hierarchy"]
    proposal = config["proposal"]
    sampling = config["sampling"]
    simulator = RBergomiSimulator(
        H=float(model["H"]),
        eta=float(model["eta"]),
        xi=float(model["xi"]),
        rho=float(model["rho"]),
        device="cpu",
    )
    maturity = float(model["maturity"])
    controls = tuple(
        TimePiecewiseTwoDriverControl(
            tuple(tuple(float(value) for value in segment) for segment in schedule),
            maturity=maturity,
        )
        for schedule in proposal["controls"]
    )
    hierarchy = MLMCHierarchy(
        int(hierarchy_config["coarsest_steps"]),
        int(hierarchy_config["refinement"]),
        FixedFinestGridTarget(int(hierarchy_config["finest_level"])),
    )
    task_items = list(config["tasks"].items())
    natural_weights = [float(value) for value in proposal["natural_weights"]]
    nonnatural_count = len(controls) - 1
    raw_ratios = proposal.get(
        "non_natural_weight_ratios", [1.0 / nonnatural_count] * nonnatural_count
    )
    nonnatural_ratios = [float(value) for value in raw_ratios]
    if (
        len(nonnatural_ratios) != nonnatural_count
        or any(value <= 0.0 for value in nonnatural_ratios)
        or abs(sum(nonnatural_ratios) - 1.0) > 1e-12
    ):
        raise ValueError("non-natural proposal weight ratios must be positive and normalized")
    absolute_rmse_targets = [
        float(value) for value in sampling.get("sampling_rmse_targets", [])
    ]
    relative_rmse_targets = [
        float(value) for value in sampling.get("relative_rmse_targets", [])
    ]
    if bool(absolute_rmse_targets) == bool(relative_rmse_targets):
        raise ValueError("declare exactly one of absolute or relative RMSE targets")
    repetitions = int(sampling["repetitions"])
    pilot_samples = int(sampling["pilot_samples"])
    if smoke:
        task_items = task_items[:1]
        natural_weights = natural_weights[:1]
        absolute_rmse_targets = absolute_rmse_targets[:1]
        relative_rmse_targets = relative_rmse_targets[:1]
        repetitions = 1
        pilot_samples = 128
    finest_steps = hierarchy.steps(hierarchy.finest_level)
    calibration_paths = int(proposal["declared_calibration_paths"])
    calibration_work = (
        calibration_paths * finest_steps * max(1.0, math.log2(finest_steps))
    )
    cells: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if progress_path is not None and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if (
            progress.get("schema") != "npi.g11.mlmc-development-progress.v1"
            or progress.get("config_sha256") != config_hash
            or progress.get("smoke") is not smoke
        ):
            raise ValueError("development progress file does not match this run")
        cells = list(progress["cells"])
        failures = list(progress["failures"])
    completed = {
        (
            float(cell["natural_weight"]),
            str(cell["task"]),
            float(cell["rmse_target"]),
            int(cell["replicate"]),
        )
        for cell in cells
    }
    for natural_weight in natural_weights:
        weights = torch.tensor(
            [natural_weight]
            + [
                (1.0 - natural_weight) * ratio
                for ratio in nonnatural_ratios
            ],
            dtype=torch.float64,
        )
        for task_name, task_specification in task_items:
            task = _task(task_specification)
            if relative_rmse_targets:
                target_probability = float(task_specification["target_probability"])
                rmse_targets = [
                    target_probability * relative for relative in relative_rmse_targets
                ]
            else:
                rmse_targets = absolute_rmse_targets
            for rmse in rmse_targets:
                for replicate in range(repetitions):
                    cell_key = (natural_weight, task_name, rmse, replicate)
                    if cell_key in completed:
                        continue
                    paired: dict[str, Any] = {}
                    protocol = (
                        f"{config['protocol_id']}:w={natural_weight}:task={task_name}:"
                        f"rmse={rmse}:rep={replicate}"
                    )
                    for method in ("raw_defensive", "dcs_mgi"):
                        try:
                            sampler = RBergomiMLMCSampler(
                                simulator,
                                controls,
                                weights,
                                task,
                                RBergomiMLMCSamplerConfig(
                                    spot=float(model["spot"]),
                                    maturity=maturity,
                                    coarsest_steps=hierarchy.coarsest_steps,
                                    method=method,
                                    engine=str(sampling["engine"]),
                                ),
                            )
                            prepared = prepare_mlmc(
                                hierarchy,
                                sampler,
                                protocol=protocol,
                                regime="development",
                                task=task_name,
                                sampling_variance_target=rmse**2,
                                pilot_samples=pilot_samples,
                                minimum_final_samples=int(
                                    sampling["minimum_final_samples"]
                                ),
                                chunk_size=int(sampling["chunk_size"]),
                                allocation_safety_factor=float(
                                    sampling["allocation_safety_factor"]
                                ),
                                initial_work_entries=(
                                    WorkLedgerEntry(
                                        "proposal_calibration",
                                        None,
                                        calibration_paths,
                                        float(calibration_work),
                                        0.0,
                                    ),
                                ),
                                minimum_pilot_nonzero=int(
                                    sampling.get("minimum_pilot_nonzero", 0)
                                ),
                                maximum_pilot_samples=int(
                                    sampling.get(
                                        "maximum_pilot_samples", pilot_samples
                                    )
                                ),
                            )
                            result = execute_mlmc(prepared, sampler)
                            paired[method] = {
                                "estimate": result.estimate,
                                "empirical_sampling_variance": (
                                    result.empirical_sampling_variance
                                ),
                                "design_sampling_variance": (
                                    result.design_sampling_variance
                                ),
                                "standard_error": result.standard_error,
                                "target_attained": bool(
                                    result.empirical_sampling_variance is not None
                                    and result.empirical_sampling_variance <= rmse**2
                                ),
                                "total_work_units": result.work.total_work_units,
                                "total_wall_seconds": result.work.total_wall_seconds,
                                "pilot": [item.__dict__ for item in result.pilot],
                                "allocations": [
                                    item.__dict__ for item in result.allocations
                                ],
                                "levels": [item.__dict__ for item in result.levels],
                                "seed_ledger_sha256": result.seed_ledger_hash,
                            }
                        except Exception as error:
                            failures.append(
                                {
                                    "protocol": protocol,
                                    "method": method,
                                    "type": type(error).__name__,
                                    "message": str(error),
                                }
                            )
                    if len(paired) == 2:
                        raw = paired["raw_defensive"]
                        dcs = paired["dcs_mgi"]
                        cells.append(
                            {
                                "natural_weight": natural_weight,
                                "task": task_name,
                                "rmse_target": rmse,
                                "replicate": replicate,
                                "methods": paired,
                                "work_ratio_raw_over_dcs": (
                                    raw["total_work_units"]
                                    / dcs["total_work_units"]
                                ),
                                "wall_ratio_raw_over_dcs": (
                                    raw["total_wall_seconds"]
                                    / dcs["total_wall_seconds"]
                                ),
                                "paired_estimate_difference": (
                                    dcs["estimate"] - raw["estimate"]
                                ),
                            }
                        )
                        completed.add(cell_key)
                    if progress_path is not None:
                        _write_progress(
                            progress_path,
                            config_hash=config_hash,
                            smoke=smoke,
                            cells=cells,
                            failures=failures,
                        )
    allocated_work_ratios = [cell["work_ratio_raw_over_dcs"] for cell in cells]
    matched_cells = [
        cell
        for cell in cells
        if cell["methods"]["raw_defensive"]["target_attained"]
        and cell["methods"]["dcs_mgi"]["target_attained"]
    ]
    matched_work_ratios = [
        cell["work_ratio_raw_over_dcs"] for cell in matched_cells
    ]
    allocated_geometric_ratio = (
        geometric_mean(allocated_work_ratios) if allocated_work_ratios else 0.0
    )
    geometric_ratio = (
        geometric_mean(matched_work_ratios) if matched_work_ratios else 0.0
    )
    method_attainment = {
        method: sum(
            cell["methods"][method]["target_attained"] for cell in cells
        )
        / len(cells)
        if cells
        else 0.0
        for method in ("raw_defensive", "dcs_mgi")
    }
    seed_manifest = [
        {
            "natural_weight": cell["natural_weight"],
            "task": cell["task"],
            "rmse_target": cell["rmse_target"],
            "replicate": cell["replicate"],
            "method": method,
            "seed_ledger_sha256": result["seed_ledger_sha256"],
        }
        for cell in cells
        for method, result in sorted(cell["methods"].items())
    ]
    aggregate_seed_hash = hashlib.sha256(
        json.dumps(
            seed_manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()
    gates = {
        "no_failures": not failures,
        "paired_cells_complete": bool(cells),
        "geometric_work_ratio": geometric_ratio,
        "geometric_work_ratio_at_least_1_25": geometric_ratio
        >= float(config["gates"]["minimum_geometric_work_ratio"]),
        "allocated_geometric_work_ratio": allocated_geometric_ratio,
        "matched_target_cell_count": len(matched_cells),
        "matched_target_cells_at_least_three": len(matched_cells) >= 3,
        "dcs_target_attainment_fraction": method_attainment["dcs_mgi"],
        "dcs_target_attainment_at_least_90_percent": method_attainment["dcs_mgi"]
        >= float(config["gates"]["minimum_target_attainment_fraction"]),
        "raw_target_attainment_fraction": method_attainment["raw_defensive"],
    }
    elapsed = time.perf_counter() - started
    method_wall_seconds = math.fsum(
        method["total_wall_seconds"]
        for cell in cells
        for method in cell["methods"].values()
    )
    return {
        "schema": "npi.g11.mlmc-development.v1",
        "protocol_id": config["protocol_id"],
        "config_sha256": config_hash,
        "input_artifacts": input_artifacts,
        "smoke": smoke,
        "estimand": "fixed finest finite-grid probability",
        "self_normalized": False,
        "calibration_work_included": True,
        "seed_ledger_sha256": aggregate_seed_hash,
        "seed_manifest_entries": len(seed_manifest),
        "cells": cells,
        "failures": failures,
        "gates": gates,
        "work_ledger": {
            "warmup_seconds": 0.0,
            "compilation_seconds": 0.0,
            "calibration_seconds": 0.0,
            "measured_method_wall_seconds": method_wall_seconds,
            "current_process_orchestration_seconds": elapsed,
            "resumable_progress_used": progress_path is not None,
            "audit_seconds": 0.0,
        },
        "environment": runtime_provenance(dtype="torch.float64"),
        **source_provenance(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/g11_mlmc_development.yaml")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    progress_path = arguments.progress or arguments.output.with_suffix(
        arguments.output.suffix + ".progress.json"
    )
    result = run(
        arguments.config, smoke=arguments.smoke, progress_path=progress_path
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(result["gates"], sort_keys=True))


if __name__ == "__main__":
    main()
