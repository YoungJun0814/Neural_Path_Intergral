"""Independent two-method references for a strict V6 cell manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import torch
import yaml

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

_SCHEMA = "npi.g11.v6-reference.config.v1"
ReferenceMethod = Literal["dcs_reference", "raw_crosscheck"]


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA:
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
    return payload, hashlib.sha256(raw).hexdigest()


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


def run(config_path: Path, manifest_path: Path, *, smoke: bool = False) -> dict[str, Any]:
    config, config_hash = _load_config(config_path)
    manifest = _load_manifest(manifest_path)
    if not smoke and config["phase"] == "qualification":
        if manifest.phase != "qualification" or not manifest.frozen or manifest.smoke:
            raise ValueError("qualification reference requires a frozen qualification manifest")
    proposal = config["proposal"]
    weights = torch.tensor(proposal["weights"], dtype=torch.float64)
    controls = tuple(
        TimePiecewiseTwoDriverControl(
            tuple(tuple(float(value) for value in segment) for segment in schedule),
            maturity=manifest.cells[0].maturity,
        )
        for schedule in proposal["controls"]
    )
    if len(controls) != weights.numel() or bool((weights <= 0.0).any()) or not math.isclose(
        float(weights.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12
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
        target_se = (
            float(contract["se_fraction_of_requested"])
            * float(contract["minimum_relative_sampling_rmse"])
            * cell.nominal_probability
        )
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
    }
    return {
        "schema": "npi.g11.v6-reference.v1",
        "protocol_id": config["protocol_id"],
        "config_sha256": config_hash,
        "manifest_sha256": manifest.sha256,
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
    arguments = parser.parse_args()
    result = run(arguments.config, arguments.manifest, smoke=arguments.smoke)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"qualified": result["reference_qualified"], **result["gates"]}))


if __name__ == "__main__":
    main()
