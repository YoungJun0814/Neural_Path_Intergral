"""Independent DCS/raw references for the hash-bound V8 P5 threshold manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Literal, cast

import torch
import yaml

from experiments.g11_v8_p5_threshold_binding_audit import audit_binding, load_binding
from experiments.g11_v8_p7_calibration import _controls, _draw, load_config
from src.path_integral import (
    DiscreteBarrierHitTask,
    OnlineMoments,
    SeedKey,
    SeedLedger,
    TerminalThresholdTask,
    evaluate_rbergomi_dcs_level,
    reference_agreement,
)
from src.path_integral.provenance import runtime_provenance, source_provenance
from src.physics_engine import RBergomiSimulator

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "npi.g11.v8-p5-independent-reference-execution.v1"
RESULT_SCHEMA = "npi.g11.v8-p5-independent-reference.v1"
ROOT_KEYS = {
    "schema",
    "protocol_id",
    "date",
    "phase",
    "outcome_data_used",
    "threshold_binding",
    "threshold_binding_sha256",
    "reference_seed_namespace",
    "final_method_seed_namespace",
    "reference_contract",
    "sampling",
    "decision",
}
ReferenceMethod = Literal["dcs_reference", "raw_crosscheck"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_reference_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(config, dict) or config.get("schema") != SCHEMA:
        raise ValueError("unexpected P5 reference-execution schema")
    if set(config) != ROOT_KEYS:
        raise ValueError("malformed P5 reference-execution root fields")
    if config.get("phase") != "p5_reference_execution" or config.get("outcome_data_used") is not False:
        raise ValueError("P5 reference execution must be outcome-blind")
    if config.get("reference_seed_namespace") != "p5-reference":
        raise ValueError("P5 reference namespace must be fixed")
    if config.get("final_method_seed_namespace") != "p5-final-method":
        raise ValueError("P5 final namespace must remain distinct")
    contract = config.get("reference_contract")
    sampling = config.get("sampling")
    if not isinstance(contract, dict) or not isinstance(sampling, dict):
        raise ValueError("P5 reference contract and sampling must be mappings")
    if contract.get("methods") != ["dcs_reference", "raw_crosscheck"]:
        raise ValueError("P5 reference methods must be fixed")
    if (
        float(contract.get("final_relative_rmse_design_target", 0.0)) != 0.20
        or float(contract.get("maximum_reference_se_fraction_of_final_target", 0.0))
        != 0.10
        or float(contract.get("maximum_combined_z_score", 0.0)) != 4.0
    ):
        raise ValueError("P5 reference precision or agreement contract is invalid")
    if int(sampling.get("pilot_replicates", 0)) < 3 or int(
        sampling.get("pilot_samples_per_replicate", 0)
    ) < 2:
        raise ValueError("P5 reference needs at least three pilot replicates")
    if int(sampling.get("maximum_final_samples", 0)) < int(
        sampling.get("minimum_final_samples", 0)):
        raise ValueError("P5 reference final-sample bounds are invalid")
    return config, hashlib.sha256(raw).hexdigest()


def _binding(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    binding_path = ROOT / str(config["threshold_binding"])
    if config.get("threshold_binding_sha256") != _sha(binding_path):
        raise ValueError("P5 reference is not bound to the threshold-binding artifact")
    binding, digest = load_binding(binding_path)
    report = audit_binding(binding, digest)
    if not report["passed"]:
        raise ValueError(f"P5 threshold binding failed: {report['failures']}")
    result_path = ROOT / str(binding["threshold_calibration_result"])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("P5 threshold calibration result must be a mapping")
    return binding, result


def _cell_task(cell: dict[str, Any]):
    threshold = float(cell["event_threshold"])
    if cell["task"] == "terminal_left_tail":
        return TerminalThresholdTask(threshold)
    if cell["task"] == "discrete_lower_barrier":
        return DiscreteBarrierHitTask(threshold)
    raise ValueError("unsupported P5 threshold-manifest task")


def _method_values(sample: Any, *, task: Any, rho: float, method: ReferenceMethod) -> torch.Tensor:
    if method == "dcs_reference":
        return evaluate_rbergomi_dcs_level(
            sample, task=task, rho=rho
        ).marginalized_contribution
    hard_event = task.hard_event(sample.paths.spot, sample.paths.step_dt)
    likelihood = torch.exp(sample.mixture_log_likelihood)
    values = hard_event.to(dtype=likelihood.dtype) * likelihood
    if not torch.isfinite(values).all():
        raise FloatingPointError("raw P5 reference contribution became nonfinite")
    return values


def _seeds(
    ledger: SeedLedger,
    *,
    protocol_id: str,
    method: ReferenceMethod,
    stage: str,
    cell_id: str,
    replicate: int,
) -> tuple[int, int]:
    proposal_seed = ledger.allocate(
        SeedKey(protocol_id, f"p5-reference-{method}-{stage}", cell_id, "fixed-grid", 0, replicate, "proposal")
    )
    label_seed = ledger.allocate(
        SeedKey(protocol_id, f"p5-reference-{method}-{stage}", cell_id, "fixed-grid", 0, replicate, "labels")
    )
    return proposal_seed, label_seed


def _draw_values(
    *,
    threshold_config: dict[str, Any],
    simulator: RBergomiSimulator,
    cell: dict[str, Any],
    method: ReferenceMethod,
    stage: str,
    replicate: int,
    count: int,
    ledger: SeedLedger,
) -> tuple[torch.Tensor, torch.Tensor]:
    task_name = str(cell["task"])
    protocol = "g11-v8-p5-independent-reference-execution-v1"
    proposal_seed, label_seed = _seeds(
        ledger,
        protocol_id=protocol,
        method=method,
        stage=stage,
        cell_id=str(cell["cell_id"]),
        replicate=replicate,
    )
    model = {
        "spot": float(cell["spot"]),
        "maturity": float(cell["maturity"]),
        "xi": float(cell["xi"]),
        "eta": float(cell["eta"]),
        "rho": float(cell["rho"]),
        "H": float(cell["hurst"]),
    }
    weights = torch.tensor(threshold_config["proposal"]["weights"], dtype=torch.float64)
    engine = cast(Literal["fft", "reference"], threshold_config["sampling"]["engine"])
    sample = _draw(
        simulator=simulator,
        controls=_controls(threshold_config, task_name),
        weights=weights,
        model=model,
        steps=int(cell["finest_steps"]),
        count=count,
        proposal_seed=proposal_seed,
        label_seed=label_seed,
        engine=engine,
    )
    if (
        not torch.isfinite(sample.paths.spot).all()
        or not torch.isfinite(sample.paths.variance).all()
        or bool((sample.paths.spot <= 0.0).any())
        or bool((sample.paths.variance <= 0.0).any())
    ):
        raise FloatingPointError("P5 reference simulation produced an invalid path")
    values = _method_values(sample, task=_cell_task(cell), rho=float(cell["rho"]), method=method)
    return values, torch.exp(sample.mixture_log_likelihood)


def _reference_method(
    *,
    threshold_config: dict[str, Any],
    cell: dict[str, Any],
    method: ReferenceMethod,
    sampling: dict[str, Any],
    target_standard_error: float,
    smoke: bool,
) -> tuple[dict[str, Any], SeedLedger]:
    pilot_replicates = int(sampling["pilot_replicates"])
    pilot_count = int(
        sampling["smoke_pilot_samples_per_replicate"]
        if smoke
        else sampling["pilot_samples_per_replicate"]
    )
    maximum_final = int(
        sampling["smoke_maximum_final_samples"] if smoke else sampling["maximum_final_samples"]
    )
    minimum_final = min(int(sampling["minimum_final_samples"]), maximum_final)
    chunk_size = int(sampling["chunk_size"])
    if chunk_size < 1 or pilot_count < 2 or target_standard_error <= 0.0:
        raise ValueError("invalid P5 reference allocation inputs")
    simulator = RBergomiSimulator(
        H=float(cell["hurst"]),
        eta=float(cell["eta"]),
        xi=float(cell["xi"]),
        rho=float(cell["rho"]),
        device="cpu",
    )
    ledger = SeedLedger()
    pilot_variances: list[float] = []
    for replicate in range(pilot_replicates):
        values, _ = _draw_values(
            threshold_config=threshold_config,
            simulator=simulator,
            cell=cell,
            method=method,
            stage="pilot",
            replicate=replicate,
            count=pilot_count,
            ledger=ledger,
        )
        pilot_variances.append(float(torch.var(values, unbiased=True)))
    design_variance = statistics.median(pilot_variances)
    requested = max(
        minimum_final,
        math.ceil(
            float(sampling["allocation_safety_factor"])
            * design_variance
            / target_standard_error**2
        ),
    )
    final_count = min(requested, maximum_final)
    moments = OnlineMoments()
    normalization = OnlineMoments()
    chunks: list[dict[str, int]] = []
    for offset in range(0, final_count, chunk_size):
        count = min(chunk_size, final_count - offset)
        values, weights = _draw_values(
            threshold_config=threshold_config,
            simulator=simulator,
            cell=cell,
            method=method,
            stage="final",
            replicate=offset // chunk_size,
            count=count,
            ledger=ledger,
        )
        moments.update(values)
        normalization.update(weights)
        chunks.append({"offset": offset, "count": count})
    standard_error = math.sqrt(moments.variance / moments.count)
    normalization_se = math.sqrt(normalization.variance / normalization.count)
    normalization_z = (
        (normalization.mean - 1.0) / normalization_se
        if normalization_se > 0.0
        else (0.0 if normalization.mean == 1.0 else math.inf)
    )
    return (
        {
            "method": method,
            "pilot_replicates": pilot_replicates,
            "pilot_samples_per_replicate": pilot_count,
            "pilot_variances": pilot_variances,
            "allocation_variance_statistic": "median_replicate_variance",
            "allocation_design_variance": design_variance,
            "requested_final_samples": requested,
            "final_samples": final_count,
            "resource_censored": requested > maximum_final,
            "estimate": moments.mean,
            "variance": moments.variance,
            "standard_error": standard_error,
            "target_standard_error": target_standard_error,
            "target_attained": standard_error <= target_standard_error,
            "independent_pilot_and_final": True,
            "normalization_mean": normalization.mean,
            "normalization_standard_error": normalization_se,
            "normalization_z": normalization_z,
            "chunks": chunks,
        },
        ledger,
    )


def run(config_path: Path, *, smoke: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    config, config_hash = load_reference_config(config_path)
    binding, threshold_result = _binding(config)
    threshold_config_path = ROOT / str(binding["threshold_calibration_config"])
    threshold_config, threshold_config_hash = load_config(threshold_config_path)
    manifest = threshold_result["candidate_manifest"]
    all_cells = manifest["cells"]
    if not isinstance(all_cells, list):
        raise ValueError("P5 bound threshold manifest must contain cells")
    if smoke:
        first = all_cells[0]
        second = next(cell for cell in all_cells if cell["task"] != first["task"])
        cells = [first, second]
    else:
        cells = all_cells
    contract = config["reference_contract"]
    final_relative_rmse = float(contract["final_relative_rmse_design_target"])
    maximum_fraction = float(contract["maximum_reference_se_fraction_of_final_target"])
    output_cells: list[dict[str, Any]] = []
    ledgers: list[SeedLedger] = []
    for cell in cells:
        nominal_probability = float(cell["nominal_probability"])
        target_se = maximum_fraction * final_relative_rmse * nominal_probability
        dcs, dcs_ledger = _reference_method(
            threshold_config=threshold_config,
            cell=cell,
            method="dcs_reference",
            sampling=config["sampling"],
            target_standard_error=target_se,
            smoke=smoke,
        )
        raw, raw_ledger = _reference_method(
            threshold_config=threshold_config,
            cell=cell,
            method="raw_crosscheck",
            sampling=config["sampling"],
            target_standard_error=target_se,
            smoke=smoke,
        )
        ledgers.extend((dcs_ledger, raw_ledger))
        agreement = reference_agreement(
            dcs["estimate"],
            dcs["standard_error"],
            raw["estimate"],
            raw["standard_error"],
            maximum_z_score=float(contract["maximum_combined_z_score"]),
        )
        output_cells.append(
            {
                "cell_id": cell["cell_id"],
                "cell": cell,
                "target_standard_error": target_se,
                "methods": [dcs, raw],
                "independent_method_agreement": {
                    "combined_z_score": agreement.combined_z_score,
                    "maximum_z_score": agreement.maximum_z_score,
                    "agrees": agreement.agrees,
                },
                "gates": {
                    "dcs_target_standard_error": dcs["target_attained"],
                    "raw_target_standard_error": raw["target_attained"],
                    "no_resource_censoring": not dcs["resource_censored"]
                    and not raw["resource_censored"],
                    "independent_methods_agree": agreement.agrees,
                    "likelihood_normalization": abs(float(dcs["normalization_z"])) <= 4.0
                    and abs(float(raw["normalization_z"])) <= 4.0,
                },
            }
        )
    merged_ledger = SeedLedger(record for ledger in ledgers for record in ledger.records)
    gates = {
        "complete_reference_matrix": len(output_cells) == len(cells) and bool(cells),
        "all_dcs_target_standard_errors": all(
            cell["gates"]["dcs_target_standard_error"] for cell in output_cells
        ),
        "all_raw_target_standard_errors": all(
            cell["gates"]["raw_target_standard_error"] for cell in output_cells
        ),
        "no_reference_resource_censoring": all(
            cell["gates"]["no_resource_censoring"] for cell in output_cells
        ),
        "all_independent_methods_agree": all(
            cell["gates"]["independent_methods_agree"] for cell in output_cells
        ),
        "all_likelihood_normalizations": all(
            cell["gates"]["likelihood_normalization"] for cell in output_cells
        ),
        "reference_final_seed_namespaces_disjoint": config["reference_seed_namespace"]
        != config["final_method_seed_namespace"],
    }
    provenance = source_provenance()
    return {
        "schema": RESULT_SCHEMA,
        "protocol_id": config["protocol_id"],
        "config_sha256": config_hash,
        "threshold_binding_sha256": config["threshold_binding_sha256"],
        "threshold_manifest_sha256": binding["threshold_manifest_sha256"],
        "threshold_calibration_config_sha256": threshold_config_hash,
        "smoke": smoke,
        "estimand": "hash-bound fixed 128-step threshold-manifest probability",
        "continuous_time_claim": False,
        "cells": output_cells,
        "gates": gates,
        "reference_complete": all(gates.values()) and not smoke,
        "performance_claim_authorized": False,
        "seed_ledger": merged_ledger.to_dict(),
        "seed_ledger_sha256": merged_ledger.sha256,
        "elapsed_seconds": time.perf_counter() - started,
        "environment": runtime_provenance(dtype="torch.float64"),
        **provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing reference: {args.output}")
    result = run(args.config, smoke=args.smoke)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"complete": result["reference_complete"], **result["gates"]}, sort_keys=True))


if __name__ == "__main__":
    main()
