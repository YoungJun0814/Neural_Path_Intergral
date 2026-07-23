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
    execute_v6_policy_durable,
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

_SCHEMA_V1 = "npi.g11.v6-baseline-qualification.config.v1"
_SCHEMA_V2 = "npi.g11.v6-baseline-qualification.config.v2"
_SCHEMA_V3 = "npi.g11.v6-baseline-qualification.config.v3"
_SCHEMA_V4 = "npi.g11.v6-baseline-qualification.config.v4"
_SCHEMA_V5 = "npi.g11.v6-baseline-qualification.config.v5"
_SCHEMA_V6 = "npi.g11.v6-baseline-qualification.config.v6"
BaselineMethod = Literal["crude", "pure_cem", "defensive_cem"]
_OPERATIONAL_QUALIFICATION_GATE_NAMES = (
    "complete_matrix",
    "all_runs_complete",
    "no_resource_censoring",
    "all_design_targets_attained",
    "all_cem_training_charged",
    "all_cem_fits_converged",
    "all_cem_controls_finite_and_bounded",
    "all_defensive_designs_certified",
    "all_crude_designs_certified",
    "all_final_seed_roles_separate",
)
_AGGREGATE_ACCURACY_RULE = (
    "deferred_to_prespecified_method_cell_attainment_and_bootstrap_rmse_co_gates"
)


def _record_checkpoint_path(
    directory: Path, *, cell_id: str, cluster: int, method: str
) -> Path:
    """Return a path-safe, deterministic checkpoint name for one final run."""

    identity = json.dumps(
        {"cell_id": cell_id, "cluster": cluster, "method": method},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(identity).hexdigest()
    return directory / "records" / f"{digest}.json"


def _clear_record_checkpoint(checkpoint: Path) -> None:
    """Remove a completed record's replay checkpoint after its journal is durable."""

    checkpoint.unlink(missing_ok=True)
    checkpoint.with_suffix(checkpoint.suffix + ".v6.json").unlink(missing_ok=True)


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict) or payload.get("schema") not in (
        _SCHEMA_V1,
        _SCHEMA_V2,
        _SCHEMA_V3,
        _SCHEMA_V4,
        _SCHEMA_V5,
        _SCHEMA_V6,
    ):
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
    if payload["schema"] in (
        _SCHEMA_V2,
        _SCHEMA_V3,
        _SCHEMA_V4,
        _SCHEMA_V5,
        _SCHEMA_V6,
    ):
        expected.add("methods")
    if payload["schema"] == _SCHEMA_V3:
        expected.add("defensive_design")
    if payload["schema"] in (_SCHEMA_V4, _SCHEMA_V5, _SCHEMA_V6):
        expected.add("rarity_band_design")
    if payload["schema"] == _SCHEMA_V6:
        expected.add("qualification_decision")
    if set(payload) != expected:
        raise ValueError("malformed V6 baseline-qualification config fields")
    if payload["phase"] not in ("development", "qualification", "confirmation"):
        raise ValueError("unsupported V6 baseline phase")
    if payload["phase"] != "development" and payload["frozen"] is not True:
        raise ValueError("qualification and confirmation baseline configs must be frozen")
    if payload["estimand"] != "fixed_finest_grid":
        raise ValueError("baseline qualification must declare a fixed-grid estimand")
    if payload["schema"] in (
        _SCHEMA_V2,
        _SCHEMA_V3,
        _SCHEMA_V4,
        _SCHEMA_V5,
        _SCHEMA_V6,
    ):
        methods = payload["methods"]
        allowed = {"crude", "pure_cem", "defensive_cem"}
        if (
            not isinstance(methods, list)
            or not methods
            or len(set(methods)) != len(methods)
            or any(method not in allowed for method in methods)
        ):
            raise ValueError(
                "V2/V3/V4/V5/V6 baseline methods must be a nonempty unique "
                "supported list"
            )
    if payload["schema"] in (_SCHEMA_V3, _SCHEMA_V4, _SCHEMA_V5, _SCHEMA_V6):
        design_key = (
            "defensive_design"
            if payload["schema"] == _SCHEMA_V3
            else "rarity_band_design"
        )
        design_contract = payload[design_key]
        expected_design_fields = {
            "nominal_probability_upper_multiplier",
            "reference_certificate_z",
        }
        if payload["schema"] in (_SCHEMA_V5, _SCHEMA_V6):
            expected_design_fields.add("defensive_variance_safety_factor")
        if (
            not isinstance(design_contract, dict)
            or set(design_contract) != expected_design_fields
        ):
            raise ValueError("malformed V3/V4/V5/V6 rarity-band design contract")
        multiplier = design_contract["nominal_probability_upper_multiplier"]
        certificate_z = design_contract["reference_certificate_z"]
        if (
            isinstance(multiplier, bool)
            or not isinstance(multiplier, (int, float))
            or not math.isfinite(float(multiplier))
            or float(multiplier) < 1.0
        ):
            raise ValueError(
                "V3/V4/V5/V6 nominal-probability upper multiplier must be finite "
                "and at least one"
            )
        if (
            isinstance(certificate_z, bool)
            or not isinstance(certificate_z, (int, float))
            or not math.isfinite(float(certificate_z))
            or float(certificate_z) <= 0.0
        ):
            raise ValueError(
                "V3/V4/V5/V6 reference-certificate z must be finite and positive"
            )
        if payload["schema"] in (_SCHEMA_V5, _SCHEMA_V6):
            safety = design_contract["defensive_variance_safety_factor"]
            if (
                isinstance(safety, bool)
                or not isinstance(safety, (int, float))
                or not math.isfinite(float(safety))
                or float(safety) < 1.0
            ):
                raise ValueError(
                    "V5/V6 defensive variance safety factor must be finite and at "
                    "least one"
                )
    if payload["schema"] == _SCHEMA_V6:
        decision = payload["qualification_decision"]
        if not isinstance(decision, dict) or set(decision) != {
            "per_record_empirical_target_role",
            "aggregate_accuracy_protocol_id",
        }:
            raise ValueError("malformed V6 baseline qualification decision")
        if decision["per_record_empirical_target_role"] != _AGGREGATE_ACCURACY_RULE:
            raise ValueError("V6 must defer accuracy to the prespecified aggregate co-gates")
        if (
            not isinstance(decision["aggregate_accuracy_protocol_id"], str)
            or not decision["aggregate_accuracy_protocol_id"]
        ):
            raise ValueError("V6 aggregate accuracy protocol ID must be nonempty")
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
    defensive_probability_upper: float | None = None,
    crude_probability_upper: float | None = None,
    defensive_variance_safety_factor: float | None = None,
) -> SingleTermDesign:
    count = int(values.numel())
    mean = float(torch.mean(values))
    variance = float(torch.var(values, unbiased=True))
    if method == "crude":
        hits = int(torch.count_nonzero(values))
        interval = exact_binomial_probability_interval(
            hits, count, confidence_level=confidence_level
        )
        if crude_probability_upper is None:
            design_variance = max(
                variance, conservative_bernoulli_variance_upper(interval)
            )
        else:
            if (
                not math.isfinite(crude_probability_upper)
                or not 0.0 < crude_probability_upper <= 1.0
            ):
                raise ValueError("crude probability upper bound must lie in (0, 1]")
            structural_upper = (
                0.25
                if crude_probability_upper >= 0.5
                else crude_probability_upper * (1.0 - crude_probability_upper)
            )
            design_variance = max(variance, structural_upper)
        bound = 1.0
    elif method == "defensive_cem":
        profile = update_profile_intervals(
            {"defensive": values},
            absolute_bounds={"defensive": defensive_bound},
            costs_per_sample={"defensive": cost_per_sample},
            familywise_alpha=bounded_alpha,
            total_predeclared_looks=1,
        )[0]
        rigorous_upper = profile.moments.variance_interval[1]
        if defensive_probability_upper is None:
            design_variance = max(variance, rigorous_upper)
        else:
            if (
                not math.isfinite(defensive_probability_upper)
                or not 0.0 < defensive_probability_upper <= 1.0
            ):
                raise ValueError(
                    "defensive probability upper bound must lie in (0, 1]"
                )
            # For Y = 1_A dP/dQ with 0 <= dP/dQ <= B,
            # Var_Q(Y) <= E_Q[Y^2] <= B E_Q[Y] = B P(A).
            structural_upper = defensive_bound * defensive_probability_upper
            if defensive_variance_safety_factor is None:
                design_variance = max(
                    variance, min(rigorous_upper, structural_upper)
                )
            else:
                if (
                    not math.isfinite(defensive_variance_safety_factor)
                    or defensive_variance_safety_factor < 1.0
                ):
                    raise ValueError(
                        "defensive variance safety factor must be finite and at least one"
                    )
                design_variance = max(
                    defensive_variance_safety_factor * variance,
                    nominal_probability**2,
                )
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
    methods = tuple(
        config["methods"]
        if config["schema"] in (
            _SCHEMA_V2,
            _SCHEMA_V3,
            _SCHEMA_V4,
            _SCHEMA_V5,
            _SCHEMA_V6,
        )
        else ("crude", "pure_cem", "defensive_cem")
    )
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
        defensive_probability_upper = None
        reference_design_certificate = None
        if config["schema"] in (_SCHEMA_V3, _SCHEMA_V4, _SCHEMA_V5, _SCHEMA_V6):
            design_key = (
                "defensive_design"
                if config["schema"] == _SCHEMA_V3
                else "rarity_band_design"
            )
            multiplier = float(
                config[design_key]["nominal_probability_upper_multiplier"]
            )
            certificate_z = float(config[design_key]["reference_certificate_z"])
            defensive_probability_upper = min(
                1.0, multiplier * cell.nominal_probability
            )
            reference_upper = reference_probability + certificate_z * reference_se
            reference_certified = reference_upper <= defensive_probability_upper
            reference_design_certificate = {
                "schema": "npi.g11.v6-defensive-design-certificate.v1",
                "nominal_probability": cell.nominal_probability,
                "nominal_probability_upper_multiplier": multiplier,
                "probability_upper_bound": defensive_probability_upper,
                "reference_certificate_z": certificate_z,
                "reference_upper_bound": reference_upper,
                "certified": reference_certified,
            }
            if not reference_certified:
                raise ValueError(
                    f"reference upper certificate exceeds the frozen rarity band for "
                    f"{cell.cell_id}"
                )
        task = _task(cell)
        simulator = RBergomiSimulator(
            H=cell.hurst, eta=cell.eta, xi=cell.xi, rho=cell.rho, device="cpu"
        )
        segments = int(config["training"]["segments"])
        natural = TimePiecewiseTwoDriverControl(
            tuple((0.0, 0.0) for _ in range(segments)), maturity=cell.maturity
        )
        for cluster in range(clusters):
            for method in methods:
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
                    defensive_probability_upper=(
                        defensive_probability_upper
                        if method == "defensive_cem"
                        else None
                    ),
                    crude_probability_upper=(
                        defensive_probability_upper
                        if config["schema"] in (_SCHEMA_V4, _SCHEMA_V5, _SCHEMA_V6)
                        and method == "crude"
                        else None
                    ),
                    defensive_variance_safety_factor=(
                        float(
                            config["rarity_band_design"][
                                "defensive_variance_safety_factor"
                            ]
                        )
                        if config["schema"] in (_SCHEMA_V5, _SCHEMA_V6)
                        and method == "defensive_cem"
                        else None
                    ),
                )
                crude_design_certificate = None
                if (
                    config["schema"] in (_SCHEMA_V4, _SCHEMA_V5, _SCHEMA_V6)
                    and method == "crude"
                    and reference_design_certificate is not None
                ):
                    probability_upper = float(
                        reference_design_certificate["probability_upper_bound"]
                    )
                    structural_upper = (
                        0.25
                        if probability_upper >= 0.5
                        else probability_upper * (1.0 - probability_upper)
                    )
                    crude_design_certificate = {
                        **reference_design_certificate,
                        "schema": "npi.g11.v6-crude-design-certificate.v1",
                        "pilot_count": int(pilot.values.numel()),
                        "pilot_mean": float(torch.mean(pilot.values)),
                        "pilot_variance": float(
                            torch.var(pilot.values, unbiased=True)
                        ),
                        "structural_variance_upper": structural_upper,
                        "selected_design_variance": design.design_variance,
                    }
                defensive_design_certificate = None
                if method == "defensive_cem" and reference_design_certificate is not None:
                    bounded_profile = update_profile_intervals(
                        {"defensive": pilot.values},
                        absolute_bounds={
                            "defensive": 1.0
                            / float(config["defensive_mixture"]["natural_weight"])
                        },
                        costs_per_sample={"defensive": sampler.cost},
                        familywise_alpha=float(sampling["bounded_familywise_alpha"]),
                        total_predeclared_looks=1,
                    )[0]
                    structural_upper = (
                        float(bounded_profile.moments.absolute_bound)
                        * float(reference_design_certificate["probability_upper_bound"])
                    )
                    defensive_design_certificate = {
                        **reference_design_certificate,
                        "absolute_bound": bounded_profile.moments.absolute_bound,
                        "familywise_alpha": float(
                            sampling["bounded_familywise_alpha"]
                        ),
                        "pilot_count": bounded_profile.moments.sample_count,
                        "pilot_mean": bounded_profile.moments.sample_mean,
                        "pilot_variance": bounded_profile.moments.sample_variance,
                        "rigorous_bounded_variance_upper": (
                            bounded_profile.moments.variance_interval[1]
                        ),
                        "structural_variance_upper": structural_upper,
                        "selected_design_variance": design.design_variance,
                    }
                    if config["schema"] in (_SCHEMA_V5, _SCHEMA_V6):
                        defensive_design_certificate = {
                            **reference_design_certificate,
                            "schema": (
                                "npi.g11.v6-defensive-plugin-design-certificate.v1"
                            ),
                            "absolute_bound": (
                                bounded_profile.moments.absolute_bound
                            ),
                            "pilot_count": bounded_profile.moments.sample_count,
                            "pilot_mean": bounded_profile.moments.sample_mean,
                            # The allocation routine uses the unbiased
                            # torch.var(..., unbiased=True) estimate.  Record
                            # that exact quantity so an independent auditor
                            # can replay the selected design variance without
                            # an n/(n-1) convention mismatch.
                            "pilot_variance": float(
                                torch.var(pilot.values, unbiased=True)
                            ),
                            "variance_safety_factor": float(
                                config["rarity_band_design"][
                                    "defensive_variance_safety_factor"
                                ]
                            ),
                            "zero_variance_fallback": (
                                cell.nominal_probability**2
                            ),
                            "structural_variance_upper_diagnostic": (
                                structural_upper
                            ),
                            "selected_design_variance": (
                                design.design_variance
                            ),
                        }
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
                record_checkpoint = (
                    None
                    if checkpoint_directory is None
                    else _record_checkpoint_path(
                        checkpoint_directory,
                        cell_id=cell.cell_id,
                        cluster=cluster,
                        method=method,
                    )
                )
                if record_checkpoint is None:
                    result = execute_v6_policy(
                        prepared,
                        sampler,
                        reference_probability=reference_probability,
                        reference_standard_error=reference_se,
                        final_peak_memory_bytes=0,
                    )
                else:
                    state_path = record_checkpoint.with_suffix(
                        record_checkpoint.suffix + ".v6.json"
                    )
                    result = execute_v6_policy_durable(
                        prepared,
                        sampler,
                        checkpoint_path=record_checkpoint,
                        resume=record_checkpoint.exists() or state_path.exists(),
                        chunks_per_checkpoint=1,
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
                        "defensive_design_certificate": (
                            defensive_design_certificate
                            if method == "defensive_cem"
                            else None
                        ),
                        "crude_design_certificate": (
                            crude_design_certificate if method == "crude" else None
                        ),
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
                    if record_checkpoint is not None:
                        _clear_record_checkpoint(record_checkpoint)
    gates = {
        "complete_matrix": len(records) == len(cells) * clusters * len(methods),
        "all_runs_complete": all(record["result"]["core"]["complete"] for record in records),
        "no_resource_censoring": all(
            not record["result"]["core"]["resource_censored"] for record in records
        ),
        "all_design_targets_attained": all(
            record["result"]["core"]["design_target_attained"] for record in records
        ),
        "all_empirical_targets_attained": all(
            record["result"]["core"]["empirical_target_attained"] for record in records
        ),
        "all_cem_training_charged": all(
            record["method"] == "crude"
            or record["result"]["total_work"]["records"][0]["category"]
            == "proposal_training"
            for record in records
        ),
        "all_cem_fits_converged": all(
            record["method"] == "crude"
            or (
                isinstance(record["cem_fit"], dict)
                and record["cem_fit"]["converged"] is True
            )
            for record in records
        ),
        "all_cem_controls_finite_and_bounded": all(
            record["method"] == "crude"
            or (
                isinstance(record["cem_fit"], dict)
                and all(
                    math.isfinite(float(value))
                    and abs(float(value))
                    <= float(config["training"]["control_bound"])
                    for segment in record["cem_fit"]["control"]
                    for value in segment
                )
            )
            for record in records
        ),
        "all_defensive_designs_certified": all(
            record["method"] != "defensive_cem"
            or (
                isinstance(record.get("defensive_design_certificate"), dict)
                and record["defensive_design_certificate"]["certified"] is True
            )
            for record in records
        )
        if config["schema"] in (_SCHEMA_V3, _SCHEMA_V4, _SCHEMA_V5, _SCHEMA_V6)
        else True,
        "all_crude_designs_certified": all(
            record["method"] != "crude"
            or (
                isinstance(record.get("crude_design_certificate"), dict)
                and record["crude_design_certificate"]["certified"] is True
            )
            for record in records
        )
        if config["schema"] in (_SCHEMA_V4, _SCHEMA_V5, _SCHEMA_V6)
        else True,
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
    qualification_gates = (
        {
            name: gates[name]
            for name in _OPERATIONAL_QUALIFICATION_GATE_NAMES
        }
        if config["schema"] == _SCHEMA_V6
        else dict(gates)
    )
    qualification_contract = None
    if config["schema"] == _SCHEMA_V6:
        qualification_contract = {
            "schema": "npi.g11.v6-baseline-qualification-contract.v1",
            "expected_cell_ids": [cell.cell_id for cell in cells],
            "expected_clusters": clusters,
            "methods": list(methods),
            "control_bound": float(config["training"]["control_bound"]),
            "operational_gate_names": list(_OPERATIONAL_QUALIFICATION_GATE_NAMES),
            "per_record_empirical_target_role": _AGGREGATE_ACCURACY_RULE,
            "aggregate_accuracy_protocol_id": config["qualification_decision"][
                "aggregate_accuracy_protocol_id"
            ],
        }
    return {
        "schema": "npi.g11.v6-baseline-qualification.v1",
        "protocol_id": config["protocol_id"],
        "config_schema": config["schema"],
        "methods": list(methods),
        "phase": config["phase"],
        "config_sha256": config_hash,
        "manifest_sha256": manifest.sha256,
        "reference_artifact_sha256": reference_hash,
        "smoke": smoke,
        "records": records,
        "gates": gates,
        "qualification_gates": qualification_gates,
        "qualification_contract": qualification_contract,
        "formal_readiness": formal,
        "baseline_qualified": all(qualification_gates.values()) and all(formal.values()),
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
