"""Independent two-method references for a strict V6 cell manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import torch
import yaml

from experiments.g11_v6_proposal_source import (
    task_conditioned_training_source_audit,
)
from src.path_integral import (
    DiscreteBarrierHitTask,
    OnlineMoments,
    SeedKey,
    SeedLedger,
    TerminalThresholdTask,
    TimePiecewiseTwoDriverControl,
    V6CellManifest,
    evaluate_rbergomi_dcs_level,
    reference_agreement,
    simulate_rbergomi_mixture,
)
from src.path_integral.provenance import runtime_provenance, source_provenance
from src.physics_engine import RBergomiSimulator

_SCHEMA_V1 = "npi.g11.v6-reference.config.v1"
_SCHEMA_V2 = "npi.g11.v6-reference.config.v2"
_SCHEMA_V3 = "npi.g11.v6-reference.config.v3"
ReferenceMethod = Literal["dcs_reference", "raw_crosscheck"]


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict) or payload.get("schema") not in (
        _SCHEMA_V1,
        _SCHEMA_V2,
        _SCHEMA_V3,
    ):
        raise ValueError("unsupported V6 reference config")
    expected = {
        "schema",
        "protocol_id",
        "phase",
        "frozen",
        "estimand",
        "proposal",
        "reference_contract",
        "sampling",
    }
    if set(payload) != expected:
        raise ValueError("malformed V6 reference config fields")
    if payload["phase"] not in ("development", "qualification"):
        raise ValueError("V6 reference phase must be development or qualification")
    if payload["estimand"] != "fixed_finest_grid":
        raise ValueError("V6 reference must declare the fixed-finest-grid estimand")
    if payload["phase"] == "qualification" and payload["frozen"] is not True:
        raise ValueError("qualification reference config must be frozen")
    sampling = payload["sampling"]
    if not isinstance(sampling, dict):
        raise ValueError("V6 reference sampling config must be an object")
    if payload["schema"] == _SCHEMA_V1:
        sampling_fields = {
            "pilot_samples",
            "minimum_final_samples",
            "maximum_final_samples",
            "chunk_size",
            "allocation_safety_factor",
            "engine",
        }
    else:
        sampling_fields = {
            "pilot_replicates",
            "pilot_samples_per_replicate",
            "minimum_final_samples",
            "maximum_final_samples",
            "chunk_size",
            "allocation_safety_factor",
            "allocation_variance_statistic",
            "raw_crosscheck_standard_error_multiplier",
            "engine",
        }
    if set(sampling) != sampling_fields:
        raise ValueError("malformed V6 reference sampling fields")
    if payload["schema"] in (_SCHEMA_V2, _SCHEMA_V3):
        if sampling["allocation_variance_statistic"] != "median_replicate_variance":
            raise ValueError("unsupported V6 reference planning statistic")
        if int(sampling["pilot_replicates"]) < 3:
            raise ValueError("V2 reference requires at least three planning replicates")
        if float(sampling["raw_crosscheck_standard_error_multiplier"]) < 1.0:
            raise ValueError("raw cross-check SE multiplier must be at least one")
    proposal = payload["proposal"]
    if not isinstance(proposal, dict):
        raise ValueError("V6 reference proposal must be an object")
    if payload["schema"] == _SCHEMA_V3:
        required_proposal = {
            "weights",
            "task_controls",
            "training_source_artifact_sha256",
            "training_derivation",
            "training_source_record_count",
            "training_total_samples",
            "training_total_work_units",
            "training_total_wall_seconds",
            "training_total_cpu_seconds",
            "training_amortization_record_count",
        }
        if set(proposal) != required_proposal:
            raise ValueError("malformed task-conditioned reference proposal")
        controls = proposal["task_controls"]
        if not isinstance(controls, dict) or set(controls) != {
            "terminal_left_tail",
            "discrete_lower_barrier",
        }:
            raise ValueError("reference proposal must cover both task families")
    return payload, hashlib.sha256(raw).hexdigest()


def _canonical_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _moment_payload(moment: OnlineMoments) -> dict[str, float | int]:
    return {"count": moment.count, "mean": moment.mean, "m2": moment.m2}


def _moment_from_payload(payload: object, *, name: str) -> OnlineMoments:
    if not isinstance(payload, dict) or set(payload) != {"count", "mean", "m2"}:
        raise ValueError(f"invalid {name} checkpoint moment")
    count = payload["count"]
    mean = payload["mean"]
    m2 = payload["m2"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError(f"invalid {name} checkpoint count")
    if isinstance(mean, bool) or not isinstance(mean, (int, float)) or not math.isfinite(mean):
        raise ValueError(f"invalid {name} checkpoint mean")
    if (
        isinstance(m2, bool)
        or not isinstance(m2, (int, float))
        or not math.isfinite(m2)
        or m2 < 0.0
    ):
        raise ValueError(f"invalid {name} checkpoint m2")
    if count < 2 and m2 != 0.0:
        raise ValueError(f"invalid {name} checkpoint low-count m2")
    if count == 0 and mean != 0.0:
        raise ValueError(f"invalid empty {name} checkpoint mean")
    return OnlineMoments(count=count, mean=float(mean), m2=float(m2))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    for attempt in range(10):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.025 * 2**attempt)


def _checkpoint_path(directory: Path, *, cell_id: str, method: ReferenceMethod) -> Path:
    identity = f"{cell_id}:{method}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return directory / "reference-methods" / f"{digest}.json"


def _load_manifest(path: Path) -> V6CellManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("V6 cell manifest must be a JSON object")
    if "candidate_manifest" in payload:
        payload = payload["candidate_manifest"]
    if not isinstance(payload, dict):
        raise ValueError("candidate_manifest must be an object")
    return V6CellManifest.from_dict(payload)


def _task(cell):
    if cell.task == "terminal_left_tail":
        return TerminalThresholdTask(cell.event_threshold)
    return DiscreteBarrierHitTask(cell.event_threshold)


def _draw_batch(
    *,
    simulator: RBergomiSimulator,
    controls,
    weights: torch.Tensor,
    cell,
    task,
    method: ReferenceMethod,
    count: int,
    proposal_seed: int,
    label_seed: int,
    engine: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(proposal_seed)
    sample = simulate_rbergomi_mixture(
        simulator,
        controls,
        weights,
        spot=cell.spot,
        maturity=cell.maturity,
        dt=cell.maturity / cell.finest_steps,
        num_paths=count,
        dtype=torch.float64,
        label_generator=torch.Generator().manual_seed(label_seed),
        engine=engine,
    )
    evaluation = evaluate_rbergomi_dcs_level(sample, task=task, rho=simulator.rho)
    values = (
        evaluation.marginalized_contribution
        if method == "dcs_reference"
        else evaluation.raw_contribution
    )
    likelihood = torch.exp(sample.mixture_log_likelihood)
    return values, likelihood


def _allocate_seeds(
    ledger: SeedLedger,
    *,
    protocol: str,
    role: str,
    cell_id: str,
    method: str,
    replicate: int,
) -> tuple[int, int]:
    proposal = ledger.allocate(
        SeedKey(protocol, role, cell_id, method, 0, replicate, "proposal")
    )
    labels = ledger.allocate(SeedKey(protocol, role, cell_id, method, 0, replicate, "labels"))
    return proposal, labels


def _method_reference(
    *,
    config: dict[str, Any],
    ledger: SeedLedger,
    cell,
    simulator: RBergomiSimulator,
    controls,
    weights: torch.Tensor,
    method: ReferenceMethod,
    target_standard_error: float,
    smoke: bool,
) -> dict[str, Any]:
    sampling = config["sampling"]
    pilot_count = 256 if smoke else int(sampling["pilot_samples"])
    minimum_final = 512 if smoke else int(sampling["minimum_final_samples"])
    maximum_final = 2048 if smoke else int(sampling["maximum_final_samples"])
    chunk_size = min(int(sampling["chunk_size"]), maximum_final)
    method_role = "reference-a" if method == "dcs_reference" else "reference-b"
    task = _task(cell)

    pilot_seeds = _allocate_seeds(
        ledger,
        protocol=str(config["protocol_id"]),
        role=f"{method_role}-pilot",
        cell_id=cell.cell_id,
        method=method,
        replicate=0,
    )
    pilot, _pilot_likelihood = _draw_batch(
        simulator=simulator,
        controls=controls,
        weights=weights,
        cell=cell,
        task=task,
        method=method,
        count=pilot_count,
        proposal_seed=pilot_seeds[0],
        label_seed=pilot_seeds[1],
        engine=str(sampling["engine"]),
    )
    defensive_bound = 1.0 / float(weights[0])
    if float(torch.amax(torch.abs(pilot))) > defensive_bound * (1.0 + 1e-12):
        raise ValueError("reference pilot exceeds the declared defensive bound")
    pilot_variance = float(torch.var(pilot, unbiased=True))
    design_variance = float(sampling["allocation_safety_factor"]) * max(
        pilot_variance, torch.finfo(torch.float64).eps
    )
    requested = max(minimum_final, math.ceil(design_variance / target_standard_error**2))
    final_count = min(requested, maximum_final)
    contribution_moments = OnlineMoments()
    normalization_moments = OnlineMoments()
    offset = 0
    chunks = []
    while offset < final_count:
        count = min(chunk_size, final_count - offset)
        proposal_seed, label_seed = _allocate_seeds(
            ledger,
            protocol=str(config["protocol_id"]),
            role=f"{method_role}-final",
            cell_id=cell.cell_id,
            method=method,
            replicate=offset // chunk_size,
        )
        values, likelihood = _draw_batch(
            simulator=simulator,
            controls=controls,
            weights=weights,
            cell=cell,
            task=task,
            method=method,
            count=count,
            proposal_seed=proposal_seed,
            label_seed=label_seed,
            engine=str(sampling["engine"]),
        )
        if float(torch.amax(torch.abs(values))) > defensive_bound * (1.0 + 1e-12):
            raise ValueError("reference contribution exceeds the defensive bound")
        if float(torch.amax(likelihood)) > defensive_bound * (1.0 + 1e-12):
            raise ValueError("reference likelihood exceeds the defensive bound")
        contribution_moments.update(values)
        normalization_moments.update(likelihood)
        chunks.append({"offset": offset, "count": count})
        offset += count
    standard_error = math.sqrt(contribution_moments.variance / contribution_moments.count)
    normalization_se = math.sqrt(normalization_moments.variance / normalization_moments.count)
    normalization_z = (
        (normalization_moments.mean - 1.0) / normalization_se
        if normalization_se > 0.0
        else (0.0 if normalization_moments.mean == 1.0 else math.inf)
    )
    return {
        "method": method,
        "pilot_samples_discarded": pilot_count,
        "pilot_variance": pilot_variance,
        "allocation_design_variance": design_variance,
        "requested_final_samples": requested,
        "final_samples": final_count,
        "resource_censored": requested > maximum_final,
        "estimate": contribution_moments.mean,
        "variance": contribution_moments.variance,
        "standard_error": standard_error,
        "target_standard_error": target_standard_error,
        "target_attained": standard_error <= target_standard_error,
        "operation_work": {
            "pilot": pilot_count
            * cell.finest_steps
            * max(1.0, math.log2(cell.finest_steps)),
            "final": final_count
            * cell.finest_steps
            * max(1.0, math.log2(cell.finest_steps)),
        },
        "normalization_mean": normalization_moments.mean,
        "normalization_standard_error": normalization_se,
        "normalization_z": normalization_z,
        "chunks": chunks,
    }


def _v2_state_payload(
    *,
    identity: dict[str, object],
    pilot_variances: list[float],
    requested_final_samples: int | None,
    final_samples: int | None,
    contribution_moments: OnlineMoments,
    normalization_moments: OnlineMoments,
    chunks: list[dict[str, int]],
    ledger: SeedLedger,
) -> dict[str, Any]:
    complete = final_samples is not None and contribution_moments.count == final_samples
    return {
        "schema": "npi.g11.v6-reference-method-checkpoint.v1",
        "identity": identity,
        "identity_sha256": _canonical_hash(identity),
        "pilot_variances": pilot_variances,
        "requested_final_samples": requested_final_samples,
        "final_samples": final_samples,
        "contribution_moments": _moment_payload(contribution_moments),
        "normalization_moments": _moment_payload(normalization_moments),
        "chunks": chunks,
        "seed_ledger": ledger.to_dict(),
        "complete": complete,
    }


def _load_v2_state(
    path: Path,
    *,
    identity: dict[str, object],
    pilot_replicates: int,
) -> tuple[
    list[float],
    int | None,
    int | None,
    OnlineMoments,
    OnlineMoments,
    list[dict[str, int]],
    SeedLedger,
]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema",
        "identity",
        "identity_sha256",
        "pilot_variances",
        "requested_final_samples",
        "final_samples",
        "contribution_moments",
        "normalization_moments",
        "chunks",
        "seed_ledger",
        "complete",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("malformed V2 reference-method checkpoint")
    if payload["schema"] != "npi.g11.v6-reference-method-checkpoint.v1":
        raise ValueError("unsupported V2 reference-method checkpoint")
    if payload["identity"] != identity or payload["identity_sha256"] != _canonical_hash(identity):
        raise ValueError("V2 reference checkpoint identity mismatch")
    raw_pilot = payload["pilot_variances"]
    if (
        not isinstance(raw_pilot, list)
        or len(raw_pilot) > pilot_replicates
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0.0
            for value in raw_pilot
        )
    ):
        raise ValueError("invalid V2 reference pilot variances")
    requested = payload["requested_final_samples"]
    final_samples = payload["final_samples"]
    for name, value in (("requested", requested), ("final", final_samples)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise ValueError(f"invalid V2 reference {name} sample count")
    if len(raw_pilot) < pilot_replicates and (requested is not None or final_samples is not None):
        raise ValueError("V2 reference allocated final samples before planning completed")
    if (requested is None) != (final_samples is None):
        raise ValueError("V2 reference checkpoint has a partial allocation")
    if requested is not None and final_samples is not None and final_samples > requested:
        raise ValueError("V2 reference final allocation exceeds its request")
    contribution = _moment_from_payload(payload["contribution_moments"], name="contribution")
    normalization = _moment_from_payload(payload["normalization_moments"], name="normalization")
    if contribution.count != normalization.count:
        raise ValueError("V2 reference checkpoint moment counts disagree")
    raw_chunks = payload["chunks"]
    if not isinstance(raw_chunks, list):
        raise ValueError("invalid V2 reference checkpoint chunks")
    chunks: list[dict[str, int]] = []
    expected_offset = 0
    for raw_chunk in raw_chunks:
        if not isinstance(raw_chunk, dict) or set(raw_chunk) != {"offset", "count"}:
            raise ValueError("invalid V2 reference checkpoint chunk")
        offset, count = raw_chunk["offset"], raw_chunk["count"]
        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or offset != expected_offset
            or count < 1
        ):
            raise ValueError("non-contiguous V2 reference checkpoint chunks")
        chunks.append({"offset": offset, "count": count})
        expected_offset += count
    if expected_offset != contribution.count:
        raise ValueError("V2 reference checkpoint chunks disagree with its moments")
    if final_samples is None and contribution.count != 0:
        raise ValueError("V2 reference checkpoint has final moments before allocation")
    if final_samples is not None and contribution.count > final_samples:
        raise ValueError("V2 reference checkpoint exceeds its final allocation")
    complete = bool(payload["complete"])
    expected_complete = final_samples is not None and contribution.count == final_samples
    if complete != expected_complete:
        raise ValueError("V2 reference checkpoint completion flag is inconsistent")
    if not isinstance(payload["seed_ledger"], dict):
        raise ValueError("invalid V2 reference checkpoint seed ledger")
    ledger = SeedLedger.from_dict(payload["seed_ledger"])
    return (
        [float(value) for value in raw_pilot],
        requested,
        final_samples,
        contribution,
        normalization,
        chunks,
        ledger,
    )


def _method_reference_v2(
    *,
    config: dict[str, Any],
    config_hash: str,
    manifest_hash: str,
    cell,
    simulator: RBergomiSimulator,
    controls,
    weights: torch.Tensor,
    method: ReferenceMethod,
    target_standard_error: float,
    smoke: bool,
    checkpoint_directory: Path | None,
    resume: bool,
) -> tuple[dict[str, Any], SeedLedger]:
    sampling = config["sampling"]
    pilot_replicates = 3 if smoke else int(sampling["pilot_replicates"])
    pilot_count = 128 if smoke else int(sampling["pilot_samples_per_replicate"])
    minimum_final = 512 if smoke else int(sampling["minimum_final_samples"])
    maximum_final = 2048 if smoke else int(sampling["maximum_final_samples"])
    chunk_size = min(int(sampling["chunk_size"]), maximum_final)
    method_target_se = target_standard_error * (
        float(sampling["raw_crosscheck_standard_error_multiplier"])
        if method == "raw_crosscheck"
        else 1.0
    )
    identity: dict[str, object] = {
        "config_sha256": config_hash,
        "manifest_sha256": manifest_hash,
        "protocol_id": str(config["protocol_id"]),
        "cell": cell.to_dict(),
        "method": method,
        "target_standard_error": method_target_se,
        "smoke": smoke,
    }
    checkpoint = (
        None
        if checkpoint_directory is None
        else _checkpoint_path(checkpoint_directory, cell_id=cell.cell_id, method=method)
    )
    if checkpoint is not None and checkpoint.exists():
        if not resume:
            raise FileExistsError("fresh V2 reference execution refuses an existing checkpoint")
        (
            pilot_variances,
            requested,
            final_count,
            contribution_moments,
            normalization_moments,
            chunks,
            ledger,
        ) = _load_v2_state(
            checkpoint,
            identity=identity,
            pilot_replicates=pilot_replicates,
        )
    else:
        pilot_variances = []
        requested = None
        final_count = None
        contribution_moments = OnlineMoments()
        normalization_moments = OnlineMoments()
        chunks = []
        ledger = SeedLedger()
    method_role = "reference-a" if method == "dcs_reference" else "reference-b"
    task = _task(cell)
    defensive_bound = 1.0 / float(weights[0])

    def save() -> None:
        if checkpoint is not None:
            _atomic_json(
                checkpoint,
                _v2_state_payload(
                    identity=identity,
                    pilot_variances=pilot_variances,
                    requested_final_samples=requested,
                    final_samples=final_count,
                    contribution_moments=contribution_moments,
                    normalization_moments=normalization_moments,
                    chunks=chunks,
                    ledger=ledger,
                ),
            )

    for replicate in range(len(pilot_variances), pilot_replicates):
        proposal_seed, label_seed = _allocate_seeds(
            ledger,
            protocol=str(config["protocol_id"]),
            role=f"{method_role}-planning",
            cell_id=cell.cell_id,
            method=method,
            replicate=replicate,
        )
        pilot, _pilot_likelihood = _draw_batch(
            simulator=simulator,
            controls=controls,
            weights=weights,
            cell=cell,
            task=task,
            method=method,
            count=pilot_count,
            proposal_seed=proposal_seed,
            label_seed=label_seed,
            engine=str(sampling["engine"]),
        )
        if float(torch.amax(torch.abs(pilot))) > defensive_bound * (1.0 + 1e-12):
            raise ValueError("reference planning sample exceeds the declared defensive bound")
        pilot_variances.append(float(torch.var(pilot, unbiased=True)))
        save()
    design_variance = float(sampling["allocation_safety_factor"]) * statistics.median(
        pilot_variances
    )
    expected_requested = max(
        minimum_final, math.ceil(max(design_variance, torch.finfo(torch.float64).eps) / method_target_se**2)
    )
    expected_final = min(expected_requested, maximum_final)
    if requested is None:
        requested = expected_requested
        final_count = expected_final
        save()
    elif requested != expected_requested or final_count != expected_final:
        raise ValueError("V2 reference checkpoint allocation does not match its planning samples")
    assert final_count is not None and requested is not None
    offset = contribution_moments.count
    while offset < final_count:
        count = min(chunk_size, final_count - offset)
        proposal_seed, label_seed = _allocate_seeds(
            ledger,
            protocol=str(config["protocol_id"]),
            role=f"{method_role}-final",
            cell_id=cell.cell_id,
            method=method,
            replicate=offset // chunk_size,
        )
        values, likelihood = _draw_batch(
            simulator=simulator,
            controls=controls,
            weights=weights,
            cell=cell,
            task=task,
            method=method,
            count=count,
            proposal_seed=proposal_seed,
            label_seed=label_seed,
            engine=str(sampling["engine"]),
        )
        if float(torch.amax(torch.abs(values))) > defensive_bound * (1.0 + 1e-12):
            raise ValueError("reference contribution exceeds the declared defensive bound")
        if float(torch.amax(likelihood)) > defensive_bound * (1.0 + 1e-12):
            raise ValueError("reference likelihood exceeds the declared defensive bound")
        contribution_moments.update(values)
        normalization_moments.update(likelihood)
        chunks.append({"offset": offset, "count": count})
        offset += count
        save()
    standard_error = math.sqrt(contribution_moments.variance / contribution_moments.count)
    normalization_se = math.sqrt(normalization_moments.variance / normalization_moments.count)
    normalization_z = (
        (normalization_moments.mean - 1.0) / normalization_se
        if normalization_se > 0.0
        else (0.0 if normalization_moments.mean == 1.0 else math.inf)
    )
    return (
        {
            "method": method,
            "planning_replicates": pilot_replicates,
            "planning_samples_per_replicate": pilot_count,
            "pilot_samples_discarded": pilot_replicates * pilot_count,
            "pilot_variances": pilot_variances,
            "allocation_variance_statistic": "median_replicate_variance",
            "allocation_design_variance": design_variance,
            "requested_final_samples": requested,
            "final_samples": final_count,
            "resource_censored": requested > maximum_final,
            "estimate": contribution_moments.mean,
            "variance": contribution_moments.variance,
            "standard_error": standard_error,
            "target_standard_error": method_target_se,
            "target_attained": standard_error <= method_target_se,
            "independent_fixed_final_allocation": True,
            "operation_work": {
                "planning": pilot_replicates
                * pilot_count
                * cell.finest_steps
                * max(1.0, math.log2(cell.finest_steps)),
                "final": final_count
                * cell.finest_steps
                * max(1.0, math.log2(cell.finest_steps)),
            },
            "normalization_mean": normalization_moments.mean,
            "normalization_standard_error": normalization_se,
            "normalization_z": normalization_z,
            "chunks": chunks,
        },
        ledger,
    )


def run(
    config_path: Path,
    manifest_path: Path,
    *,
    smoke: bool = False,
    checkpoint_directory: Path | None = None,
    resume: bool = False,
    proposal_training_source_path: Path | None = None,
) -> dict[str, Any]:
    config, config_hash = _load_config(config_path)
    if resume and checkpoint_directory is None:
        raise ValueError("V6 reference resume requires a checkpoint directory")
    if config["schema"] == _SCHEMA_V1 and (checkpoint_directory is not None or resume):
        raise ValueError("durable reference execution requires a V2 reference config")
    manifest = _load_manifest(manifest_path)
    if not smoke and config["phase"] == "qualification":
        if manifest.phase != "qualification" or not manifest.frozen or manifest.smoke:
            raise ValueError("qualification reference requires a frozen qualification manifest")
    proposal = config["proposal"]
    weights = torch.tensor(proposal["weights"], dtype=torch.float64)
    proposal_training_audit = None
    if config["schema"] == _SCHEMA_V3:
        if proposal_training_source_path is None:
            raise ValueError(
                "task-conditioned reference requires its proposal-training source"
            )
        proposal_training_audit = task_conditioned_training_source_audit(
            proposal, proposal_training_source_path
        )
        controls_by_task = {
            task: tuple(
                TimePiecewiseTwoDriverControl(
                    tuple(
                        tuple(float(value) for value in segment)
                        for segment in schedule
                    ),
                    maturity=manifest.cells[0].maturity,
                )
                for schedule in schedules
            )
            for task, schedules in proposal["task_controls"].items()
        }
    else:
        controls = tuple(
            TimePiecewiseTwoDriverControl(
                tuple(
                    tuple(float(value) for value in segment)
                    for segment in schedule
                ),
                maturity=manifest.cells[0].maturity,
            )
            for schedule in proposal["controls"]
        )
        controls_by_task = {
            "terminal_left_tail": controls,
            "discrete_lower_barrier": controls,
        }
    if (
        any(len(controls) != weights.numel() for controls in controls_by_task.values())
        or bool((weights <= 0.0).any())
        or not math.isclose(
        float(weights.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12
        )
    ):
        raise ValueError("reference proposal weights and controls are invalid")
    if smoke:
        first = manifest.cells[0]
        different_task = next((cell for cell in manifest.cells if cell.task != first.task), None)
        cells = (first,) if different_task is None else (first, different_task)
    else:
        cells = manifest.cells
    ledger = SeedLedger()
    output_cells = []
    contract = config["reference_contract"]
    for cell in cells:
        if cell.maturity != manifest.cells[0].maturity:
            raise ValueError("one reference proposal schedule requires a common maturity")
        simulator = RBergomiSimulator(
            H=cell.hurst,
            eta=cell.eta,
            xi=cell.xi,
            rho=cell.rho,
            device="cpu",
        )
        controls = controls_by_task[cell.task]
        target_se = (
            float(contract["se_fraction_of_requested"])
            * float(contract["minimum_relative_sampling_rmse"])
            * cell.nominal_probability
        )
        if config["schema"] in (_SCHEMA_V2, _SCHEMA_V3):
            method_outputs = tuple(
                _method_reference_v2(
                    config=config,
                    config_hash=config_hash,
                    manifest_hash=manifest.sha256,
                    cell=cell,
                    simulator=simulator,
                    controls=controls,
                    weights=weights,
                    method=method,
                    target_standard_error=target_se,
                    smoke=smoke,
                    checkpoint_directory=checkpoint_directory,
                    resume=resume,
                )
                for method in ("dcs_reference", "raw_crosscheck")
            )
            methods = tuple(output[0] for output in method_outputs)
            ledger = SeedLedger(
                [
                    *ledger.records,
                    *(record for output in method_outputs for record in output[1].records),
                ]
            )
        else:
            methods = tuple(
                _method_reference(
                    config=config,
                    ledger=ledger,
                    cell=cell,
                    simulator=simulator,
                    controls=controls,
                    weights=weights,
                    method=method,
                    target_standard_error=target_se,
                    smoke=smoke,
                )
                for method in ("dcs_reference", "raw_crosscheck")
            )
        agreement = reference_agreement(
            methods[0]["estimate"],
            methods[0]["standard_error"],
            methods[1]["estimate"],
            methods[1]["standard_error"],
            maximum_z_score=float(contract["maximum_combined_z_score"]),
        )
        output_cells.append(
            {
                "cell_id": cell.cell_id,
                "cell": cell.to_dict(),
                "target_standard_error": target_se,
                "methods": list(methods),
                "independent_method_agreement": asdict(agreement),
                "gates": {
                    "dcs_reference_se_contract": methods[0]["target_attained"],
                    **(
                        {"raw_crosscheck_se_contract": methods[1]["target_attained"]}
                        if config["schema"] in (_SCHEMA_V2, _SCHEMA_V3)
                        else {}
                    ),
                    "no_reference_resource_censoring": not any(
                        method["resource_censored"] for method in methods
                    ),
                    "independent_methods_agree": agreement.agrees,
                    "likelihood_normalization": all(
                        abs(float(method["normalization_z"]))
                        <= float(contract["maximum_normalization_z"])
                        for method in methods
                    ),
                },
            }
        )
    gates = {
        "complete_reference_matrix": len(output_cells) == len(cells) and bool(cells),
        "all_dcs_reference_se_contracts": all(
            cell["gates"]["dcs_reference_se_contract"] for cell in output_cells
        ),
        **(
            {
                "all_raw_crosscheck_se_contracts": all(
                    cell["gates"]["raw_crosscheck_se_contract"] for cell in output_cells
                )
            }
            if config["schema"] in (_SCHEMA_V2, _SCHEMA_V3)
            else {}
        ),
        "no_reference_resource_censoring": all(
            cell["gates"]["no_reference_resource_censoring"] for cell in output_cells
        ),
        "all_independent_methods_agree": all(
            cell["gates"]["independent_methods_agree"] for cell in output_cells
        ),
        "all_likelihood_normalizations": all(
            cell["gates"]["likelihood_normalization"] for cell in output_cells
        ),
    }
    provenance = source_provenance()
    formal = {
        "frozen_config": bool(config["frozen"]),
        "frozen_manifest": manifest.frozen,
        "clean_source": not bool(provenance["dirty_worktree"]),
        "non_smoke": not smoke,
        "proposal_training_source_verified": (
            config["schema"] != _SCHEMA_V3
            or bool(proposal_training_audit and proposal_training_audit["verified"])
        ),
        "proposal_training_source_formal": (
            config["schema"] != _SCHEMA_V3
            or bool(
                proposal_training_audit
                and proposal_training_audit[
                    "formal_training_source_readiness"
                ]
            )
        ),
    }
    return {
        "schema": "npi.g11.v6-reference.v1",
        "protocol_id": config["protocol_id"],
        "config_schema": config["schema"],
        "config_sha256": config_hash,
        "manifest_sha256": manifest.sha256,
        "proposal_training_audit": proposal_training_audit,
        "smoke": smoke,
        "estimand": "fixed finest-grid event probability",
        "continuous_time_claim": False,
        "cells": output_cells,
        "gates": gates,
        "formal_readiness": formal,
        "reference_qualified": all(gates.values()) and all(formal.values()),
        "seed_ledger": ledger.to_dict(),
        "seed_ledger_sha256": ledger.sha256,
        "environment": runtime_provenance(dtype="torch.float64"),
        **provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/g11_v6/reference_development.yaml"),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--checkpoint-directory", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--proposal-training-source", type=Path)
    arguments = parser.parse_args()
    result = run(
        arguments.config,
        arguments.manifest,
        smoke=arguments.smoke,
        checkpoint_directory=arguments.checkpoint_directory,
        resume=arguments.resume,
        proposal_training_source_path=arguments.proposal_training_source,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"qualified": result["reference_qualified"], **result["gates"]}))


if __name__ == "__main__":
    main()
