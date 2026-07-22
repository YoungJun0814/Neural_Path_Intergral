"""Independent finite-grid reference probabilities for G11 V5 cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from src.path_integral import (
    DiscreteBarrierHitTask,
    OnlineMoments,
    SeedKey,
    SeedLedger,
    TerminalThresholdTask,
    TimePiecewiseTwoDriverControl,
    black_scholes_discrete_lower_barrier_probability,
    black_scholes_left_digital_probability,
    evaluate_rbergomi_dcs_level,
    reference_agreement,
    simulate_rbergomi_mixture,
)
from src.path_integral.provenance import runtime_provenance, source_provenance
from src.physics_engine import RBergomiSimulator


def _load(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict) or payload.get("schema") != ("npi.g11.v5-reference.config.v1"):
        raise ValueError("unsupported V5 reference config")
    if payload.get("estimand") != "finite_grid":
        raise ValueError("reference config must declare the finite-grid estimand")
    return payload, hashlib.sha256(raw).hexdigest()


def _task(specification: dict[str, Any]):
    if specification["kind"] == "terminal":
        return TerminalThresholdTask(float(specification["level"]))
    if specification["kind"] == "barrier":
        return DiscreteBarrierHitTask(float(specification["barrier"]))
    raise ValueError("reference runner supports terminal and barrier tasks")


def _draw_batch(
    *,
    simulator: RBergomiSimulator,
    controls,
    weights: torch.Tensor,
    model: dict[str, Any],
    steps: int,
    task,
    method: str,
    count: int,
    proposal_seed: int,
    label_seed: int,
    engine: str,
) -> torch.Tensor:
    torch.manual_seed(proposal_seed)
    sample = simulate_rbergomi_mixture(
        simulator,
        controls,
        weights,
        spot=float(model["spot"]),
        maturity=float(model["maturity"]),
        dt=float(model["maturity"]) / steps,
        num_paths=count,
        label_generator=torch.Generator().manual_seed(label_seed),
        engine=engine,
    )
    evaluation = evaluate_rbergomi_dcs_level(sample, task=task, rho=simulator.rho)
    if method == "dcs_reference":
        return evaluation.marginalized_contribution
    if method == "raw_crosscheck":
        return evaluation.raw_contribution
    raise ValueError("unknown reference method")


def _method_reference(
    *,
    config: dict[str, Any],
    ledger: SeedLedger,
    cell_id: str,
    task_name: str,
    task,
    simulator: RBergomiSimulator,
    controls,
    weights: torch.Tensor,
    model: dict[str, Any],
    method: str,
    target_standard_error: float,
    smoke: bool,
) -> dict[str, Any]:
    sampling = config["sampling"]
    pilot_count = 256 if smoke else int(sampling["pilot_samples"])
    maximum_final = 2048 if smoke else int(sampling["maximum_final_samples"])
    minimum_final = 512 if smoke else int(sampling["minimum_final_samples"])
    steps = int(config["grid"]["steps"])

    def seeds(role: str, replicate: int) -> tuple[int, int]:
        return tuple(
            ledger.allocate(
                SeedKey(
                    config["protocol_id"],
                    role,
                    cell_id,
                    f"{task_name}:{method}",
                    0,
                    replicate,
                    stream,
                )
            )
            for stream in ("proposal", "labels")
        )  # type: ignore[return-value]

    pilot_seeds = seeds("reference-pilot", 0)
    pilot = _draw_batch(
        simulator=simulator,
        controls=controls,
        weights=weights,
        model=model,
        steps=steps,
        task=task,
        method=method,
        count=pilot_count,
        proposal_seed=pilot_seeds[0],
        label_seed=pilot_seeds[1],
        engine=str(sampling["engine"]),
    )
    pilot_variance = float(torch.var(pilot, unbiased=True))
    defensive_bound = 1.0 / float(min(config["proposal"]["weights"]))
    if not math.isfinite(pilot_variance) or pilot_variance <= 0.0:
        pilot_variance = defensive_bound**2
    design_variance = float(sampling["allocation_safety_factor"]) * pilot_variance
    requested_final = max(minimum_final, math.ceil(design_variance / target_standard_error**2))
    final_count = min(requested_final, maximum_final)
    moments = OnlineMoments()
    chunk_size = int(sampling["chunk_size"])
    chunks: list[dict[str, Any]] = []
    offset = 0
    while offset < final_count:
        count = min(chunk_size, final_count - offset)
        chunk_seed = seeds("reference-final", offset // chunk_size)
        values = _draw_batch(
            simulator=simulator,
            controls=controls,
            weights=weights,
            model=model,
            steps=steps,
            task=task,
            method=method,
            count=count,
            proposal_seed=chunk_seed[0],
            label_seed=chunk_seed[1],
            engine=str(sampling["engine"]),
        )
        moments.update(values)
        chunks.append(
            {
                "offset": offset,
                "count": count,
                "maximum_absolute_contribution": float(torch.amax(torch.abs(values))),
            }
        )
        offset += count
    standard_error = math.sqrt(moments.variance / moments.count)
    return {
        "method": method,
        "pilot_samples_discarded": pilot_count,
        "pilot_variance": float(torch.var(pilot, unbiased=True)),
        "allocation_design_variance": design_variance,
        "requested_final_samples": requested_final,
        "final_samples": final_count,
        "resource_censored": requested_final > maximum_final,
        "estimate": moments.mean,
        "variance": moments.variance,
        "standard_error": standard_error,
        "target_standard_error": target_standard_error,
        "target_attained": standard_error <= target_standard_error,
        "chunks": chunks,
    }


def run(config_path: Path, *, smoke: bool = False) -> dict[str, Any]:
    config, config_hash = _load(config_path)
    common = config["model_common"]
    maturity = float(common["maturity"])
    controls = tuple(
        TimePiecewiseTwoDriverControl(
            tuple(tuple(float(value) for value in segment) for segment in schedule),
            maturity=maturity,
        )
        for schedule in config["proposal"]["controls"]
    )
    weights = torch.tensor(config["proposal"]["weights"], dtype=torch.float64)
    ledger = SeedLedger()
    cells: list[dict[str, Any]] = []
    for model_spec in config["models"]:
        model = {**common, **model_spec}
        model_id = str(model_spec["id"])
        simulator = RBergomiSimulator(
            H=float(model["H"]),
            eta=float(model["eta"]),
            xi=float(model["xi"]),
            rho=float(model["rho"]),
            device="cpu",
        )
        for task_name, specification in config["tasks"].items():
            task = _task(specification)
            nominal = float(specification["nominal_probability"])
            target_se = (
                float(config["reference_contract"]["se_fraction_of_requested"])
                * float(config["reference_contract"]["minimum_relative_sampling_rmse"])
                * nominal
            )
            methods = [
                _method_reference(
                    config=config,
                    ledger=ledger,
                    cell_id=model_id,
                    task_name=task_name,
                    task=task,
                    simulator=simulator,
                    controls=controls,
                    weights=weights,
                    model=model,
                    method=method,
                    target_standard_error=target_se,
                    smoke=smoke,
                )
                for method in ("dcs_reference", "raw_crosscheck")
            ]
            agreement = reference_agreement(
                methods[0]["estimate"],
                methods[0]["standard_error"],
                methods[1]["estimate"],
                methods[1]["standard_error"],
                maximum_z_score=float(config["reference_contract"]["maximum_combined_z_score"]),
            )
            analytic: dict[str, Any] | None = None
            if float(model["eta"]) == 0.0:
                volatility = math.sqrt(float(model["xi"]))
                if isinstance(task, TerminalThresholdTask):
                    probability = black_scholes_left_digital_probability(
                        spot=float(model["spot"]),
                        level=task.level,
                        volatility=volatility,
                        maturity=maturity,
                    )
                    source = "closed_form_black_scholes_terminal"
                else:
                    oracle = black_scholes_discrete_lower_barrier_probability(
                        spot=float(model["spot"]),
                        barrier=task.barrier,
                        volatility=volatility,
                        maturity=maturity,
                        steps=int(config["grid"]["steps"]),
                        state_points=int(config["eta_zero_oracle"]["state_points"]),
                        upper_standard_deviations=float(
                            config["eta_zero_oracle"]["upper_standard_deviations"]
                        ),
                    )
                    probability = oracle.probability
                    source = "deterministic_killed_density_quadrature"
                analytic_agreement = reference_agreement(
                    methods[0]["estimate"],
                    methods[0]["standard_error"],
                    probability,
                    0.0,
                    maximum_z_score=float(config["reference_contract"]["maximum_combined_z_score"]),
                )
                analytic = {
                    "source": source,
                    "probability": probability,
                    "dcs_agreement": asdict(analytic_agreement),
                }
            cells.append(
                {
                    "cell_id": f"{model_id}:{task_name}",
                    "model": model,
                    "task": specification,
                    "target_standard_error": target_se,
                    "methods": methods,
                    "independent_method_agreement": asdict(agreement),
                    "eta_zero_oracle": analytic,
                    "gate": {
                        "reference_se_contract": all(
                            method["target_attained"]
                            for method in methods
                            if method["method"] == "dcs_reference"
                        ),
                        "independent_agreement": agreement.agrees,
                        "eta_zero_agreement": (
                            analytic is None or analytic["dcs_agreement"]["agrees"]
                        ),
                    },
                }
            )
    formal_gates = {
        "all_reference_standard_errors_attained": all(
            cell["gate"]["reference_se_contract"] for cell in cells
        ),
        "all_independent_crosschecks_agree": all(
            cell["gate"]["independent_agreement"] for cell in cells
        ),
        "all_eta_zero_oracles_agree": all(cell["gate"]["eta_zero_agreement"] for cell in cells),
    }
    provenance = source_provenance()
    formal_readiness = {
        "frozen_config": bool(config.get("frozen")),
        "clean_source": not bool(provenance["dirty_worktree"]),
        "non_smoke": not smoke,
    }
    return {
        "schema": "npi.g11.v5-reference.v1",
        "protocol_id": config["protocol_id"],
        "config_sha256": config_hash,
        "smoke": smoke,
        "estimand": "fixed finite-grid event probabilities",
        "cells": cells,
        "gates": formal_gates,
        "formal_readiness": formal_readiness,
        "reference_qualified": all(formal_gates.values()) and all(formal_readiness.values()),
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
        default=Path("configs/g11_v5_reference_development.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    result = run(arguments.config, smoke=arguments.smoke)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
