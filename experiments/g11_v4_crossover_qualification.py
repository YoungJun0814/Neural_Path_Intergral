"""Parameter-separated DCS/SLIS/MLMC crossover qualification for G11 V4."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from experiments.g11_m7_confirmatory import _atomic_json
from src.path_integral import (
    DiscreteBarrierHitTask,
    DownsideExcursionTask,
    SeedKey,
    SeedLedger,
    TerminalThresholdTask,
    TimePiecewiseTwoDriverControl,
    evaluate_total_work_crossover,
)
from src.path_integral.provenance import runtime_provenance, source_provenance
from src.path_integral.rbergomi_mlmc_sampler import (
    RBergomiMLMCSampler,
    RBergomiMLMCSamplerConfig,
)
from src.physics_engine import RBergomiSimulator
from src.training.rbergomi_piecewise_cem import fit_rbergomi_piecewise_cem


def _sha256(path: Path) -> str:
    normalized = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def _load(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("unsupported V4 crossover config")
    if config.get("run_class") != "qualification":
        raise ValueError("V4 crossover runner is restricted to qualification")
    if config.get("estimand") != "finite_grid":
        raise ValueError("V4 crossover qualification requires a finite-grid estimand")
    if config.get("frozen") is not True:
        raise ValueError("V4 crossover qualification config must be frozen")
    return config, hashlib.sha256(raw).hexdigest()


def _verify_freeze(config: dict[str, Any]) -> None:
    tag = str(config["required_git_tag"])
    head = subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()
    tag_commit = subprocess.check_output(("git", "rev-list", "-n", "1", tag), text=True).strip()
    if head != tag_commit:
        raise ValueError("qualification HEAD does not match its frozen Git tag")
    if source_provenance()["dirty_worktree"]:
        raise ValueError("frozen qualification requires a clean worktree")


def _event_task(name: str, specification: dict[str, Any], threshold: float):
    if name == "terminal":
        return TerminalThresholdTask(threshold)
    if name == "barrier":
        return DiscreteBarrierHitTask(threshold)
    if name == "excursion":
        return DownsideExcursionTask(
            hit_barrier=threshold,
            stress_level=float(specification["stress_level"]),
            minimum_occupation=float(specification["minimum_occupation"]),
            hit_scale=float(specification["hit_scale"]),
            occupation_scale=float(specification["occupation_scale"]),
        )
    raise ValueError(f"unsupported task {name}")


def _load_cells(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    cells: list[dict[str, Any]] = []
    inputs: list[dict[str, str]] = []
    for declaration in config["regimes"]:
        config_path = Path(declaration["calibration_config"]["path"])
        result_path = Path(declaration["calibration_result"]["path"])
        for path, expected in (
            (config_path, declaration["calibration_config"]["sha256"]),
            (result_path, declaration["calibration_result"]["sha256"]),
        ):
            actual = _sha256(path)
            if actual != str(expected):
                raise ValueError(f"frozen input hash mismatch: {path}")
            inputs.append({"path": str(path), "sha256": actual})
        calibration_config = yaml.safe_load(config_path.read_bytes())
        calibration_result = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(calibration_config, dict) or not isinstance(calibration_result, dict):
            raise ValueError("calibration inputs must be mappings")
        if calibration_result.get("config_sha256") != _sha256(config_path):
            raise ValueError("calibration result/config generation mismatch")
        if calibration_result.get("smoke") is not False:
            raise ValueError("V4 qualification cannot use smoke calibration")
        if calibration_result.get("gates", {}).get("likelihood_normalization") is not True:
            raise ValueError("selected calibration failed likelihood normalization")
        result_cells = calibration_result.get("cells")
        if not isinstance(result_cells, list):
            raise ValueError("calibration result has no cells")
        for selection in declaration["selections"]:
            task_name = str(selection["task"])
            probability = float(selection["target_probability"])
            matches = [
                item
                for item in result_cells
                if item["task"] == task_name
                and math.isclose(
                    float(item["target_probability"]),
                    probability,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            ]
            if len(matches) != 1:
                raise ValueError("selected calibration cell is missing or duplicated")
            matched = matches[0]
            if (
                matched.get("probability_band_passed") is not True
                or matched.get("precision_passed") is not True
            ):
                raise ValueError("selected calibration cell failed its declared gates")
            specification = calibration_config["tasks"][task_name]
            cells.append(
                {
                    "regime": str(declaration["name"]),
                    "changed_parameter": str(declaration["changed_parameter"]),
                    "model": calibration_config["model"],
                    "task_name": task_name,
                    "task_id": f"{task_name}_{probability:.0e}",
                    "target_probability": probability,
                    "reference_estimate": float(matched["validation_estimate"]),
                    "reference_standard_error": float(matched["validation_standard_error"]),
                    "task": _event_task(
                        task_name,
                        specification,
                        float(matched["calibrated_threshold"]),
                    ),
                    "cem_enabled": bool(selection.get("cem_enabled", False)),
                }
            )
    identities = [(cell["regime"], cell["task_id"]) for cell in cells]
    if len(identities) != len(set(identities)):
        raise ValueError("V4 crossover cells are duplicated")
    return cells, inputs


def _controls(config: dict[str, Any], maturity: float):
    return tuple(
        TimePiecewiseTwoDriverControl(
            tuple(tuple(float(value) for value in segment) for segment in schedule),
            maturity=maturity,
        )
        for schedule in config["proposal"]["controls"]
    )


def _batch_seeds(
    ledger: SeedLedger,
    *,
    protocol: str,
    regime: str,
    task: str,
    method: str,
    level: int,
    replicate: int,
) -> dict[str, int]:
    return {
        stream: ledger.allocate(
            SeedKey(
                protocol,
                "profile",
                regime,
                f"{task}:{method}",
                level,
                replicate,
                stream,
            )
        )
        for stream in ("proposal", "labels")
    }


def _profile(batch) -> dict[str, float | int]:
    count = int(batch.values.numel())
    variance = float(torch.var(batch.values, unbiased=True))
    return {
        "count": count,
        "nonzero_count": int(torch.count_nonzero(batch.values)),
        "mean": float(torch.mean(batch.values)),
        "variance": variance,
        "standard_error": math.sqrt(variance / count),
        "second_moment": float(torch.mean(batch.values.square())),
        "cost_per_sample": float(batch.work_units) / count,
        "wall_seconds": float(batch.wall_seconds),
        "work_units": float(batch.work_units),
    }


def _sampler(
    simulator: RBergomiSimulator,
    controls,
    weights: torch.Tensor,
    task,
    *,
    spot: float,
    maturity: float,
    coarsest_steps: int,
    method: str,
    require_natural_component: bool,
    engine: str,
) -> RBergomiMLMCSampler:
    return RBergomiMLMCSampler(
        simulator,
        controls,
        weights,
        task,
        RBergomiMLMCSamplerConfig(
            spot=spot,
            maturity=maturity,
            coarsest_steps=coarsest_steps,
            method=method,
            require_natural_component=require_natural_component,
            engine=engine,
        ),
    )


def _cell_result(config: dict[str, Any], cell: dict[str, Any]) -> dict[str, Any]:
    model = cell["model"]
    simulator = RBergomiSimulator(
        H=float(model["H"]),
        eta=float(model["eta"]),
        xi=float(model["xi"]),
        rho=float(model["rho"]),
        device="cpu",
    )
    spot = float(model["spot"])
    maturity = float(model["maturity"])
    hierarchy = config["hierarchy"]
    coarsest = int(hierarchy["coarsest_steps"])
    finest_level = int(hierarchy["finest_level"])
    finest_steps = coarsest * int(hierarchy["refinement"]) ** finest_level
    engine = str(config["sampling"]["engine"])
    paths = int(config["sampling"]["paths_per_profile"])
    repetitions = int(config["sampling"]["repetitions"])
    controls = _controls(config, maturity)
    weights = torch.tensor(config["proposal"]["weights"], dtype=torch.float64)
    dcs_correction_sampler = _sampler(
        simulator,
        controls,
        weights,
        cell["task"],
        spot=spot,
        maturity=maturity,
        coarsest_steps=coarsest,
        method="dcs_mgi",
        require_natural_component=True,
        engine=engine,
    )
    natural_control = TimePiecewiseTwoDriverControl(((0.0, 0.0),), maturity=maturity)
    natural_sampler = _sampler(
        simulator,
        (natural_control,),
        torch.ones(1, dtype=torch.float64),
        cell["task"],
        spot=spot,
        maturity=maturity,
        coarsest_steps=finest_steps,
        method="raw_defensive",
        require_natural_component=True,
        engine=engine,
    )
    ledger = SeedLedger()

    cem_payload: dict[str, Any] | None = None
    cem_sampler: RBergomiMLMCSampler | None = None
    cem_training_work = 0.0
    if cell["cem_enabled"]:
        cem = config["cem"]
        training_seed = ledger.allocate(
            SeedKey(
                config["protocol_id"],
                "training",
                cell["regime"],
                cell["task_id"],
                finest_level,
                0,
                "proposal",
            )
        )
        training_started = time.perf_counter()
        fit = fit_rbergomi_piecewise_cem(
            simulator,
            cell["task"],
            spot=spot,
            maturity=maturity,
            dt=maturity / finest_steps,
            initial_control=tuple(
                tuple(float(value) for value in segment) for segment in cem["initial_control"]
            ),
            num_paths=int(cem["paths_per_iteration"]),
            seed=training_seed,
            max_iterations=int(cem["maximum_iterations"]),
            elite_quantile=float(cem["elite_quantile"]),
            smoothing=float(cem["smoothing"]),
            min_elite_paths=int(cem["minimum_elite_paths"]),
            control_bound=float(cem["control_bound"]),
            target_level_repetitions=int(cem["target_level_repetitions"]),
        )
        cem_training_seconds = time.perf_counter() - training_started
        cem_training_work = (
            len(fit.history)
            * int(cem["paths_per_iteration"])
            * finest_steps
            * math.log2(finest_steps)
        )
        fitted_control = TimePiecewiseTwoDriverControl(fit.control, maturity=maturity)
        cem_sampler = _sampler(
            simulator,
            (fitted_control,),
            torch.ones(1, dtype=torch.float64),
            cell["task"],
            spot=spot,
            maturity=maturity,
            coarsest_steps=finest_steps,
            method="raw",
            require_natural_component=False,
            engine=engine,
        )
        cem_payload = {
            "control": fit.control,
            "converged": fit.converged,
            "history": [asdict(item) for item in fit.history],
            "training_seed": training_seed,
            "training_work_units": cem_training_work,
            "training_wall_seconds": cem_training_seconds,
        }

    runs: list[dict[str, Any]] = []
    for replicate in range(repetitions):
        single_levels: list[dict[str, Any]] = []
        for level in range(finest_level + 1):
            steps = coarsest * 2**level
            sampler = _sampler(
                simulator,
                controls,
                weights,
                cell["task"],
                spot=spot,
                maturity=maturity,
                coarsest_steps=steps,
                method="dcs_mgi",
                require_natural_component=True,
                engine=engine,
            )
            single_levels.append(
                _profile(
                    sampler(
                        0,
                        "pilot",
                        paths,
                        _batch_seeds(
                            ledger,
                            protocol=config["protocol_id"],
                            regime=cell["regime"],
                            task=cell["task_id"],
                            method="dcs_single",
                            level=level,
                            replicate=replicate,
                        ),
                    )
                )
            )
        corrections: list[dict[str, Any]] = []
        for level in range(1, finest_level + 1):
            corrections.append(
                _profile(
                    dcs_correction_sampler(
                        level,
                        "pilot",
                        paths,
                        _batch_seeds(
                            ledger,
                            protocol=config["protocol_id"],
                            regime=cell["regime"],
                            task=cell["task_id"],
                            method="dcs_correction",
                            level=level,
                            replicate=replicate,
                        ),
                    )
                )
            )
        crude = _profile(
            natural_sampler(
                0,
                "pilot",
                paths,
                _batch_seeds(
                    ledger,
                    protocol=config["protocol_id"],
                    regime=cell["regime"],
                    task=cell["task_id"],
                    method="crude_single",
                    level=finest_level,
                    replicate=replicate,
                ),
            )
        )
        cem_profile = (
            _profile(
                cem_sampler(
                    0,
                    "pilot",
                    paths,
                    _batch_seeds(
                        ledger,
                        protocol=config["protocol_id"],
                        regime=cell["regime"],
                        task=cell["task_id"],
                        method="cem_slis",
                        level=finest_level,
                        replicate=replicate,
                    ),
                )
            )
            if cem_sampler is not None
            else None
        )
        dcs_profile_work = math.fsum(
            float(item["work_units"]) for item in (*single_levels, *corrections)
        )
        decisions: dict[str, Any] = {}
        for relative_rmse_value in config["relative_rmse_targets"]:
            relative_rmse = float(relative_rmse_value)
            variance_target = (cell["target_probability"] * relative_rmse) ** 2
            crossover = evaluate_total_work_crossover(
                single_level_variances=[float(item["variance"]) for item in single_levels],
                single_level_costs=[float(item["cost_per_sample"]) for item in single_levels],
                correction_variances=[float(item["variance"]) for item in corrections],
                correction_costs=[float(item["cost_per_sample"]) for item in corrections],
                preprocessing_work_by_start_level=[dcs_profile_work] * (finest_level + 1),
                sampling_variance_target=variance_target,
            )
            crude_total = None
            candidates = {"dcs_mlmc_or_slis": crossover.optimal_total_work}
            if float(crude["variance"]) > 0.0:
                crude_total = float(crude["work_units"]) + (
                    float(crude["variance"]) * float(crude["cost_per_sample"]) / variance_target
                )
                candidates["crude_single"] = crude_total
            cem_total = None
            if cem_profile is not None and float(cem_profile["variance"]) > 0.0:
                cem_total = (
                    cem_training_work
                    + float(cem_profile["work_units"])
                    + float(cem_profile["variance"])
                    * float(cem_profile["cost_per_sample"])
                    / variance_target
                )
                candidates["cem_slis"] = cem_total
            selected_method = min(candidates, key=lambda method: (candidates[method], method))
            decisions[f"{relative_rmse:.2f}"] = {
                "variance_target": variance_target,
                "dcs": asdict(crossover),
                "crude_single_total_work": crude_total,
                "cem_slis_total_work": cem_total,
                "selected_method": selected_method,
                "selected_total_work": candidates[selected_method],
            }
        estimates = [
            float(single_levels[start]["mean"])
            + math.fsum(float(item["mean"]) for item in corrections[start:])
            for start in range(finest_level + 1)
        ]
        standard_errors = [
            math.sqrt(
                float(single_levels[start]["standard_error"]) ** 2
                + math.fsum(float(item["standard_error"]) ** 2 for item in corrections[start:])
            )
            for start in range(finest_level + 1)
        ]
        runs.append(
            {
                "replicate": replicate,
                "dcs_single_levels": single_levels,
                "dcs_corrections": corrections,
                "dcs_telescoping_estimates_by_start": estimates,
                "dcs_telescoping_standard_errors_by_start": standard_errors,
                "crude_single": crude,
                "cem_slis": cem_profile,
                "decisions": decisions,
            }
        )
    return {
        "regime": cell["regime"],
        "changed_parameter": cell["changed_parameter"],
        "model": model,
        "task": cell["task_id"],
        "target_probability": cell["target_probability"],
        "reference_estimate": cell["reference_estimate"],
        "reference_standard_error": cell["reference_standard_error"],
        "cem": cem_payload,
        "runs": runs,
        "seed_ledger_sha256": ledger.sha256,
        "seed_count": len(ledger),
    }


def _summary(config: dict[str, Any], cells: list[dict[str, Any]]) -> dict[str, Any]:
    decision_counts: dict[str, dict[str, int]] = {}
    start_level_counts: dict[str, dict[str, int]] = {}
    reference_z: list[float] = []
    for relative_rmse_value in config["relative_rmse_targets"]:
        key = f"{float(relative_rmse_value):.2f}"
        decision_counts[key] = {}
        start_level_counts[key] = {}
        for cell in cells:
            for run in cell["runs"]:
                decision = run["decisions"][key]
                method = str(decision["selected_method"])
                decision_counts[key][method] = decision_counts[key].get(method, 0) + 1
                start = str(decision["dcs"]["optimal_start_level"])
                start_level_counts[key][start] = start_level_counts[key].get(start, 0) + 1
    for cell in cells:
        for run in cell["runs"]:
            estimate = float(run["dcs_telescoping_estimates_by_start"][0])
            standard_error = float(run["dcs_telescoping_standard_errors_by_start"][0])
            combined = math.sqrt(standard_error**2 + float(cell["reference_standard_error"]) ** 2)
            if combined > 0.0:
                reference_z.append((estimate - float(cell["reference_estimate"])) / combined)
    absolute_z = [abs(value) for value in reference_z]
    return {
        "cell_count": len(cells),
        "run_count": sum(len(cell["runs"]) for cell in cells),
        "decision_counts": decision_counts,
        "dcs_optimal_start_level_counts": start_level_counts,
        "reference_comparison_count": len(reference_z),
        "reference_within_4_se_fraction": statistics.fmean(
            float(value <= 4.0) for value in absolute_z
        ),
        "reference_median_absolute_z": statistics.median(absolute_z),
        "reference_maximum_absolute_z": max(absolute_z),
    }


def run(config_path: Path, *, progress_path: Path | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    config, config_hash = _load(config_path)
    _verify_freeze(config)
    expected_cells, inputs = _load_cells(config)
    completed: list[dict[str, Any]] = []
    if progress_path is not None and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if (
            progress.get("schema") != "npi.g11.v4-crossover-progress.v1"
            or progress.get("config_sha256") != config_hash
        ):
            raise ValueError("V4 crossover progress does not match the config")
        completed = list(progress["cells"])
    completed_keys = {(cell["regime"], cell["task"]) for cell in completed}
    for cell in expected_cells:
        key = (cell["regime"], cell["task_id"])
        if key in completed_keys:
            continue
        completed.append(_cell_result(config, cell))
        completed_keys.add(key)
        if progress_path is not None:
            _atomic_json(
                progress_path,
                {
                    "schema": "npi.g11.v4-crossover-progress.v1",
                    "config_sha256": config_hash,
                    "complete_cell_count": len(completed),
                    "cells": completed,
                },
            )
    summary = _summary(config, completed)
    gates = {
        "complete_cell_matrix": len(completed) == len(expected_cells),
        "minimum_repetitions": int(config["sampling"]["repetitions"]) >= 5,
        "all_reference_comparisons_within_fraction": summary["reference_within_4_se_fraction"]
        >= float(config["gates"]["minimum_reference_within_4_se_fraction"]),
        "multiple_rmse_targets": len(config["relative_rmse_targets"]) >= 3,
        "parameter_separated": {cell["changed_parameter"] for cell in expected_cells}
        >= {"base", "H", "eta", "rho"},
    }
    gates["qualification_passed"] = all(gates.values())
    return {
        "schema": "npi.g11.v4-crossover-qualification.v1",
        "protocol_id": config["protocol_id"],
        "run_class": config["run_class"],
        "config_sha256": config_hash,
        "estimand": "fixed finest finite-grid probability",
        "continuous_time_claim": False,
        "input_artifacts": inputs,
        "relative_rmse_targets": config["relative_rmse_targets"],
        "cells": completed,
        "summary": summary,
        "gates": gates,
        "work_ledger": {"orchestration_seconds": time.perf_counter() - started},
        "environment": runtime_provenance(dtype="torch.float64"),
        **source_provenance(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", type=Path)
    arguments = parser.parse_args()
    progress = arguments.progress or arguments.output.with_suffix(
        arguments.output.suffix + ".progress.json"
    )
    result = run(arguments.config, progress_path=progress)
    _atomic_json(arguments.output, result)
    print(json.dumps(result["gates"], sort_keys=True))


if __name__ == "__main__":
    main()
