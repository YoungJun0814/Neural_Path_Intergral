"""Actual achieved-RMSE crude and CEM baselines for V6 manifest cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import torch
import yaml

from experiments.g11_v6_reference import _load_manifest
from src.path_integral import (
    DiscreteBarrierHitTask,
    HybridTarget,
    LevelBatch,
    SeedKey,
    SeedLedger,
    SingleTermDesign,
    TerminalThresholdTask,
    TimePiecewiseTwoDriverControl,
    V6ProgressJournal,
    V6WorkLedger,
    V6WorkRecord,
    conservative_bernoulli_variance_upper,
    exact_binomial_probability_interval,
    execute_v6_policy,
    heavy_tail_diagnostics,
    load_v6_progress,
    prepare_v6_direct_policy,
    save_v6_progress,
    simulate_rbergomi_mixture,
    update_profile_intervals,
    v6_policy_preparation_to_dict,
)
from src.path_integral.provenance import runtime_provenance, source_provenance
from src.path_integral.rbergomi_fft import simulate_rbergomi_fft
from src.physics_engine import RBergomiSimulator
from src.training import fit_rbergomi_piecewise_cem

_SCHEMA = "npi.g11.v6-baseline-qualification.config.v1"
BaselineMethod = Literal["crude", "pure_cem", "defensive_cem"]


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA:
        raise ValueError("unsupported V6 baseline-qualification config")
    expected = {
        "schema",
        "protocol_id",
        "phase",
        "frozen",
        "estimand",
        "sampling",
        "training",
        "defensive_mixture",
    }
    if set(payload) != expected:
        raise ValueError("malformed V6 baseline-qualification config fields")
    if payload["phase"] not in ("development", "qualification", "confirmation"):
        raise ValueError("unsupported V6 baseline phase")
    if payload["phase"] != "development" and payload["frozen"] is not True:
        raise ValueError("qualification and confirmation baseline configs must be frozen")
    if payload["estimand"] != "fixed_finest_grid":
        raise ValueError("baseline qualification must declare a fixed-grid estimand")
    return payload, hashlib.sha256(raw).hexdigest()


def _load_references(
    path: Path,
) -> tuple[dict[str, tuple[float, float, dict[str, Any]]], str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or payload.get("schema") != "npi.g11.v6-reference.v1":
        raise ValueError("unsupported V6 reference artifact")
    references = {}
    for cell in payload.get("cells", []):
        methods = cell.get("methods", [])
        dcs = next((method for method in methods if method.get("method") == "dcs_reference"), None)
        if dcs is None:
            raise ValueError("reference cell lacks dcs_reference")
        cell_payload = cell.get("cell")
        if not isinstance(cell_payload, dict):
            raise ValueError("reference cell lacks its estimand definition")
        references[str(cell["cell_id"])] = (
            float(dcs["estimate"]),
            float(dcs["standard_error"]),
            dict(cell_payload),
        )
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")
    return references, hashlib.sha256(canonical).hexdigest()


def _task(cell):
    if cell.task == "terminal_left_tail":
        return TerminalThresholdTask(cell.event_threshold)
    return DiscreteBarrierHitTask(cell.event_threshold)


def _smoke_cells(cells):
    first = cells[0]
    different = next((cell for cell in cells if cell.task != first.task), None)
    return (first,) if different is None else (first, different)


class _DirectRBergomiSampler:
    def __init__(
        self,
        *,
        method: BaselineMethod,
        simulator: RBergomiSimulator,
        task,
        natural,
        fitted,
        defensive_weight: float,
        cell,
        engine: str,
    ) -> None:
        self.method = method
        self.simulator = simulator
        self.task = task
        self.natural = natural
        self.fitted = fitted
        self.defensive_weight = defensive_weight
        self.cell = cell
        self.engine = engine
        self.cost = cell.finest_steps * max(1.0, math.log2(cell.finest_steps))

    def __call__(self, profile_id, role, count, seeds):
        del profile_id, role
        started = time.perf_counter()
        torch.manual_seed(seeds["proposal"])
        if self.method == "defensive_cem":
            if self.fitted is None or set(seeds) != {"proposal", "labels"}:
                raise ValueError("defensive CEM requires a fitted control and two streams")
            sample = simulate_rbergomi_mixture(
                self.simulator,
                (self.natural, self.fitted),
                torch.tensor(
                    [self.defensive_weight, 1.0 - self.defensive_weight], dtype=torch.float64
                ),
                spot=self.cell.spot,
                maturity=self.cell.maturity,
                dt=self.cell.maturity / self.cell.finest_steps,
                num_paths=count,
                dtype=torch.float64,
                label_generator=torch.Generator().manual_seed(seeds["labels"]),
                engine=self.engine,
            )
            event = self.task.hard_event(sample.paths.spot, sample.paths.step_dt)
            values = event.to(torch.float64) * torch.exp(sample.mixture_log_likelihood)
        else:
            control = self.natural if self.method == "crude" else self.fitted
            if control is None or set(seeds) != {"proposal"}:
                raise ValueError("direct crude/pure CEM requires one proposal stream")
            paths = simulate_rbergomi_fft(
                self.simulator,
                S0=self.cell.spot,
                T=self.cell.maturity,
                dt=self.cell.maturity / self.cell.finest_steps,
                num_paths=count,
                control_fn=control,
                dtype=torch.float64,
            )
            event = self.task.hard_event(paths.spot, paths.step_dt)
            values = event.to(torch.float64)
            if self.method == "pure_cem":
                values = values * torch.exp(paths.log_likelihood)
        return LevelBatch(
            values.detach(),
            count * self.cost,
            wall_seconds=time.perf_counter() - started,
        )


def _work_record(
    category,
    *,
    method: str,
    cell_id: str,
    samples: int,
    work_units: float,
    wall_seconds: float,
    cpu_seconds: float,
) -> V6WorkRecord:
    return V6WorkRecord(
        category=category,
        method=method,
        cell_id=cell_id,
        attempt=0,
        samples=samples,
        work_units=work_units,
        wall_seconds=wall_seconds,
        cpu_seconds=cpu_seconds,
        peak_memory_bytes=0,
        successful=True,
    )


def _fit_control(config, simulator, task, cell, seed, *, smoke: bool):
    training = config["training"]
    paths = 256 if smoke else int(training["paths_per_iteration"])
    iterations = 2 if smoke else int(training["maximum_iterations"])
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    fit = fit_rbergomi_piecewise_cem(
        simulator,
        task,
        spot=cell.spot,
        maturity=cell.maturity,
        dt=cell.maturity / cell.finest_steps,
        initial_control=tuple(
            tuple(float(value) for value in segment) for segment in training["initial_control"]
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
    wall = time.perf_counter() - wall_start
    cpu = time.process_time() - cpu_start
    control = TimePiecewiseTwoDriverControl(fit.control, maturity=cell.maturity)
    operation_scale = cell.finest_steps * max(1.0, math.log2(cell.finest_steps))
    return fit, control, paths * len(fit.history) * operation_scale, wall, cpu


def _design_from_pilot(
    method: BaselineMethod,
    values: torch.Tensor,
    *,
    cost_per_sample: float,
    nominal_probability: float,
    confidence_level: float,
    pure_safety: float,
    defensive_bound: float,
    bounded_alpha: float,
) -> SingleTermDesign:
    count = int(values.numel())
    mean = float(torch.mean(values))
    variance = float(torch.var(values, unbiased=True))
    if method == "crude":
        hits = int(torch.count_nonzero(values))
        interval = exact_binomial_probability_interval(
            hits, count, confidence_level=confidence_level
        )
        design_variance = max(variance, conservative_bernoulli_variance_upper(interval))
        bound = 1.0
    elif method == "defensive_cem":
        profile = update_profile_intervals(
            {"defensive": values},
            absolute_bounds={"defensive": defensive_bound},
            costs_per_sample={"defensive": cost_per_sample},
            familywise_alpha=bounded_alpha,
            total_predeclared_looks=1,
        )[0]
        design_variance = max(variance, profile.moments.variance_interval[1])
        bound = defensive_bound
    else:
        design_variance = pure_safety * variance
        if design_variance == 0.0:
            # Pure CEM has no deterministic likelihood bound.  A zero-variance
            # pilot therefore falls back to crude Bernoulli scale and must still
            # pass the achieved-RMSE gate; this is a design safeguard, not a bound.
            design_variance = nominal_probability * (1.0 - nominal_probability)
        design_variance = max(variance, design_variance)
        bound = None
    return SingleTermDesign(
        profile_id=method,
        pilot_count=count,
        pilot_mean=mean,
        pilot_variance=variance,
        design_variance=design_variance,
        cost_per_sample=cost_per_sample,
        absolute_bound=bound,
    )


def run(
    config_path: Path,
    manifest_path: Path,
    reference_path: Path,
    *,
    smoke: bool = False,
    checkpoint_directory: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    config, config_hash = _load_config(config_path)
    manifest = _load_manifest(manifest_path)
    references, reference_hash = _load_references(reference_path)
    if resume and checkpoint_directory is None:
        raise ValueError("baseline resume requires a checkpoint directory")
    if not smoke and config["phase"] != "development":
        if (
            manifest.phase != config["phase"]
            or not manifest.frozen
            or manifest.smoke
        ):
            raise ValueError("formal baselines require a same-phase frozen manifest")
    cells = _smoke_cells(manifest.cells) if smoke else manifest.cells
    clusters = 1 if smoke else int(config["sampling"]["clusters"])
    sampling = config["sampling"]
    pilot_count = 256 if smoke else int(sampling["pilot_samples"])
    relative_rmse = max(0.50, float(sampling["relative_sampling_rmse"])) if smoke else float(
        sampling["relative_sampling_rmse"]
    )
    progress_identities: dict[str, object] = {
        "config_sha256": config_hash,
        "manifest_sha256": manifest.sha256,
        "reference_sha256": reference_hash,
        "smoke": smoke,
    }
    progress_path = (
        None if checkpoint_directory is None else checkpoint_directory / "progress.json"
    )
    if progress_path is not None and resume:
        records = list(
            load_v6_progress(
                progress_path,
                experiment="g11_v6_baseline",
                identities=progress_identities,
            ).records
        )
    else:
        if progress_path is not None and progress_path.exists():
            raise FileExistsError("fresh baseline execution refuses existing progress")
        records = []
    completed = {
        (str(record["cell_id"]), int(record["cluster"]), str(record["method"]))
        for record in records
    }
    master_ledger = SeedLedger()
    for record in records:
        restored = SeedLedger.from_dict(record["result"]["core"]["seed_ledger_payload"])
        for seed_record in restored.records:
            master_ledger.allocate(seed_record.key)
    if progress_path is not None and not progress_path.exists():
        save_v6_progress(
            progress_path,
            V6ProgressJournal("g11_v6_baseline", progress_identities, tuple(records)),
        )
    for cell in cells:
        if cell.cell_id not in references:
            raise ValueError(f"reference artifact lacks cell {cell.cell_id}")
        reference_probability, reference_se, reference_cell = references[cell.cell_id]
        if reference_cell != cell.to_dict():
            raise ValueError(f"reference estimand drift for cell {cell.cell_id}")
        task = _task(cell)
        simulator = RBergomiSimulator(
            H=cell.hurst, eta=cell.eta, xi=cell.xi, rho=cell.rho, device="cpu"
        )
        segments = int(config["training"]["segments"])
        natural = TimePiecewiseTwoDriverControl(
            tuple((0.0, 0.0) for _ in range(segments)), maturity=cell.maturity
        )
        for cluster in range(clusters):
            for method in ("crude", "pure_cem", "defensive_cem"):
                record_identity = (cell.cell_id, cluster, method)
                if record_identity in completed:
                    continue
                policy_name = method
                ledger = SeedLedger()
                work = V6WorkLedger()
                fitted = None
                fit_payload = None
                if method in ("pure_cem", "defensive_cem"):
                    training_seed = ledger.allocate(
                        SeedKey(
                            str(config["protocol_id"]),
                            "proposal-training",
                            f"{cell.cell_id}:cluster-{cluster}",
                            method,
                            0,
                            0,
                            "proposal",
                        )
                    )
                    fit, fitted, training_work, training_wall, training_cpu = _fit_control(
                        config, simulator, task, cell, training_seed, smoke=smoke
                    )
                    work = work.append(
                        _work_record(
                            "proposal_training",
                            method=policy_name,
                            cell_id=cell.cell_id,
                            samples=(256 if smoke else int(config["training"]["paths_per_iteration"]))
                            * len(fit.history),
                            work_units=training_work,
                            wall_seconds=training_wall,
                            cpu_seconds=training_cpu,
                        )
                    )
                    fit_payload = {
                        "converged": fit.converged,
                        "iterations": len(fit.history),
                        "control": fit.control,
                        "history": [asdict(item) for item in fit.history],
                    }
                sampler = _DirectRBergomiSampler(
                    method=method,
                    simulator=simulator,
                    task=task,
                    natural=natural,
                    fitted=fitted,
                    defensive_weight=float(config["defensive_mixture"]["natural_weight"]),
                    cell=cell,
                    engine=str(sampling["engine"]),
                )
                streams = ("proposal", "labels") if method == "defensive_cem" else ("proposal",)
                pilot_seeds = {
                    stream: ledger.allocate(
                        SeedKey(
                            str(config["protocol_id"]),
                            "allocation-pilot",
                            f"{cell.cell_id}:cluster-{cluster}",
                            method,
                            0,
                            0,
                            stream,
                        )
                    )
                    for stream in streams
                }
                pilot_cpu_start = time.process_time()
                pilot = sampler(method, "pilot", pilot_count, pilot_seeds)
                pilot_cpu = time.process_time() - pilot_cpu_start
                work = work.append(
                    _work_record(
                        "allocation_pilot",
                        method=policy_name,
                        cell_id=cell.cell_id,
                        samples=pilot_count,
                        work_units=pilot.work_units,
                        wall_seconds=pilot.wall_seconds,
                        cpu_seconds=pilot_cpu,
                    )
                )
                design = _design_from_pilot(
                    method,
                    pilot.values,
                    cost_per_sample=sampler.cost,
                    nominal_probability=cell.nominal_probability,
                    confidence_level=float(sampling["confidence_level"]),
                    pure_safety=float(sampling["pure_cem_variance_safety_factor"]),
                    defensive_bound=(
                        1.0 / float(config["defensive_mixture"]["natural_weight"])
                    ),
                    bounded_alpha=float(sampling["bounded_familywise_alpha"]),
                )
                prepared = prepare_v6_direct_policy(
                    HybridTarget(
                        f"{cell.cell_id}:{method}",
                        cell.nominal_probability,
                        relative_rmse,
                        confidence_level=float(sampling["confidence_level"]),
                    ),
                    design,
                    policy_name=policy_name,
                    cell_id=cell.cell_id,
                    execution_method=method,
                    protocol=f"{config['protocol_id']}:cluster-{cluster}:{method}",
                    regime=f"{cell.cell_id}:cluster-{cluster}",
                    task=cell.task,
                    operation_work_cap=float(sampling["operation_work_cap"]),
                    preprocessing_work=work,
                    chunk_size=(512 if smoke else int(sampling["chunk_size"])),
                    minimum_final_samples=(128 if smoke else int(sampling["minimum_final_samples"])),
                    streams=streams,
                    preparation_seed_ledger=ledger,
                )
                result = execute_v6_policy(
                    prepared,
                    sampler,
                    reference_probability=reference_probability,
                    reference_standard_error=reference_se,
                    final_peak_memory_bytes=0,
                )
                records.append(
                    {
                        "cell_id": cell.cell_id,
                        "cluster": cluster,
                        "method": method,
                        "nominal_probability": cell.nominal_probability,
                        "reference_probability": reference_probability,
                        "reference_standard_error": reference_se,
                        "cem_fit": fit_payload,
                        "pilot_tail_diagnostics": asdict(heavy_tail_diagnostics(pilot.values)),
                        "design": asdict(design),
                        "preparation_hash": prepared.policy_hash,
                        "preparation": v6_policy_preparation_to_dict(prepared),
                        "result": asdict(result),
                    }
                )
                final_ledger = SeedLedger.from_dict(result.core.seed_ledger_payload)
                for seed_record in final_ledger.records:
                    master_ledger.allocate(seed_record.key)
                if progress_path is not None:
                    save_v6_progress(
                        progress_path,
                        V6ProgressJournal(
                            "g11_v6_baseline", progress_identities, tuple(records)
                        ),
                    )
    gates = {
        "complete_matrix": len(records) == len(cells) * clusters * 3,
        "all_runs_complete": all(record["result"]["core"]["complete"] for record in records),
        "no_resource_censoring": all(
            not record["result"]["core"]["resource_censored"] for record in records
        ),
        "all_design_targets_attained": all(
            record["result"]["core"]["design_target_attained"] for record in records
        ),
        "all_cem_training_charged": all(
            record["method"] == "crude"
            or record["result"]["total_work"]["records"][0]["category"]
            == "proposal_training"
            for record in records
        ),
        "all_final_seed_roles_separate": all(
            all(
                seed["key"]["role"] == "final"
                for seed in record["result"]["core"]["seed_ledger_payload"]["records"]
                if seed["key"]["role"] not in {"proposal-training", "allocation-pilot"}
            )
            for record in records
        ),
    }
    provenance = source_provenance()
    formal = {
        "frozen_config": bool(config["frozen"]),
        "frozen_manifest": manifest.frozen,
        "clean_source": not bool(provenance["dirty_worktree"]),
        "non_smoke": not smoke,
    }
    return {
        "schema": "npi.g11.v6-baseline-qualification.v1",
        "protocol_id": config["protocol_id"],
        "phase": config["phase"],
        "config_sha256": config_hash,
        "manifest_sha256": manifest.sha256,
        "reference_artifact_sha256": reference_hash,
        "smoke": smoke,
        "records": records,
        "gates": gates,
        "formal_readiness": formal,
        "baseline_qualified": all(gates.values()) and all(formal.values()),
        "seed_ledger": master_ledger.to_dict(),
        "seed_ledger_sha256": master_ledger.sha256,
        "environment": runtime_provenance(dtype="torch.float64"),
        **provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/g11_v6/baseline_qualification_development.yaml"),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--checkpoint-directory", type=Path)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    result = run(
        arguments.config,
        arguments.manifest,
        arguments.reference,
        smoke=arguments.smoke,
        checkpoint_directory=arguments.checkpoint_directory,
        resume=arguments.resume,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"qualified": result["baseline_qualified"], **result["gates"]}))


if __name__ == "__main__":
    main()
