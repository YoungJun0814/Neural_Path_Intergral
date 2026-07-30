"""Falsify barrier-aware defensive proposals before reopening full references."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, cast

import torch
import yaml

from experiments.g11_v8_p5_reference import _cell_task, _method_values
from experiments.g11_v8_p5_sharded_reference_common import ROOT, load_context
from experiments.g11_v8_p7_calibration import _draw
from src.path_integral import (
    REFERENCE_METHODS,
    SeedKey,
    SufficientStatistics,
    TimePiecewiseTwoDriverControl,
    derive_seed,
)
from src.path_integral.provenance import runtime_provenance, source_provenance
from src.physics_engine import RBergomiSimulator

SCHEMA = "npi.g11.v8-p5-barrier-reference-proposal-falsification.v1"
RESULT_SCHEMA = "npi.g11.v8-p5-barrier-reference-proposal-falsification-result.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bound_path(record: Any) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError("barrier proposal artifact binding is malformed")
    path = (ROOT / str(record["path"])).resolve()
    if ROOT not in path.parents or not path.is_file():
        raise ValueError("barrier proposal artifact path is invalid")
    if record["sha256"] != _sha256(path):
        raise ValueError("barrier proposal artifact hash mismatch")
    return path


def load_falsification_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(config, dict) or config.get("schema") != SCHEMA:
        raise ValueError("unexpected barrier proposal falsification schema")
    for field in (
        "allocation_failure_receipt",
        "pilot_package",
        "threshold_binding",
    ):
        _bound_path(config.get(field))
    if (
        config.get("design_informed_by_prior_development_outcomes") is not True
        or config.get("current_namespace_outcomes_inspected_before_freeze") is not False
        or config.get("namespace") != "v8-r2-barrier-proposal-falsification-v1"
    ):
        raise ValueError("barrier proposal provenance contract is invalid")
    cells = config.get("cells")
    candidates = config.get("candidates")
    if (
        not isinstance(cells, list)
        or len(cells) != 6
        or len(set(cells)) != 6
        or not isinstance(candidates, list)
        or [candidate.get("id") for candidate in candidates]
        != [
            "current_constant_mixture",
            "front_loaded_rank_one",
            "mid_loaded_rank_one",
            "decaying_rank_one",
        ]
    ):
        raise ValueError("barrier proposal cell or candidate roster changed")
    for candidate in candidates:
        weights = candidate.get("weights")
        schedules = candidate.get("schedules")
        if (
            not isinstance(weights, list)
            or not isinstance(schedules, list)
            or len(weights) != len(schedules)
            or any(float(weight) <= 0.0 for weight in weights)
            or not math.isclose(sum(float(weight) for weight in weights), 1.0)
        ):
            raise ValueError("barrier proposal mixture is invalid")
    sampling = config.get("sampling")
    gates = config.get("gates")
    decision = config.get("decision")
    if (
        not isinstance(sampling, dict)
        or sampling.get("replicates") != 4
        or sampling.get("paths_per_replicate") != 8192
        or sampling.get("engine") != "fft"
        or not isinstance(gates, dict)
        or gates.get("allocation_safety_factor") != 6.0
        or gates.get("maximum_final_samples") != 8388608
        or gates.get("minimum_raw_nonzero_contributions_per_replicate") != 20
        or not isinstance(decision, dict)
        or decision.get("new_full_pilot_authorized") is not False
        or decision.get("final_execution_authorized") is not False
        or decision.get("performance_claim_authorized") is not False
    ):
        raise ValueError("barrier proposal sampling or gate contract is invalid")
    return config, hashlib.sha256(raw).hexdigest()


def _controls(
    candidate: dict[str, Any], maturity: float
) -> tuple[TimePiecewiseTwoDriverControl, ...]:
    return tuple(
        TimePiecewiseTwoDriverControl(
            tuple((float(pair[0]), float(pair[1])) for pair in schedule),
            maturity=maturity,
        )
        for schedule in candidate["schedules"]
    )


def _seeds(
    protocol_id: str,
    namespace: str,
    candidate_id: str,
    cell_id: str,
    replicate: int,
) -> tuple[SeedKey, SeedKey]:
    seed_protocol = f"{protocol_id}/{namespace}"
    role = f"proposal-falsification/{candidate_id}"
    return (
        SeedKey(
            seed_protocol, role, cell_id, "paired-dcs-raw", 0, replicate, "proposal"
        ),
        SeedKey(
            seed_protocol, role, cell_id, "paired-dcs-raw", 0, replicate, "labels"
        ),
    )


def run_falsification(config_path: Path, *, smoke: bool = False) -> dict[str, Any]:
    config, config_sha256 = load_falsification_config(config_path)
    execution_config = ROOT / "configs/g11_v8/p5_sharded_reference_execution_v3.yaml"
    context = load_context(execution_config)
    if context.binding_sha256 != config["threshold_binding"]["sha256"]:
        raise ValueError("falsification config binds a different threshold manifest")
    sampling = config["sampling"]
    cells = config["cells"][: sampling["smoke_cells"]] if smoke else config["cells"]
    candidates = (
        config["candidates"][: sampling["smoke_candidates"]]
        if smoke
        else config["candidates"]
    )
    replicates = (
        int(sampling["smoke_replicates"])
        if smoke
        else int(sampling["replicates"])
    )
    paths = (
        int(sampling["smoke_paths_per_replicate"])
        if smoke
        else int(sampling["paths_per_replicate"])
    )
    engine = cast(Literal["fft", "reference"], sampling["engine"])
    gates = config["gates"]
    outputs: list[dict[str, Any]] = []
    seed_records: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_entries: list[dict[str, Any]] = []
        all_normalization_values: list[torch.Tensor] = []
        for cell_id in cells:
            cell = context.cells_by_id[cell_id]
            simulator = RBergomiSimulator(
                H=float(cell["hurst"]),
                eta=float(cell["eta"]),
                xi=float(cell["xi"]),
                rho=float(cell["rho"]),
                device="cpu",
            )
            method_variances: dict[str, list[float]] = {
                method: [] for method in REFERENCE_METHODS
            }
            method_means: dict[str, list[float]] = {
                method: [] for method in REFERENCE_METHODS
            }
            method_maxima: dict[str, list[float]] = {
                method: [] for method in REFERENCE_METHODS
            }
            raw_nonzero_counts: list[int] = []
            for replicate in range(replicates):
                proposal_key, label_key = _seeds(
                    config["protocol_id"],
                    config["namespace"],
                    candidate["id"],
                    cell_id,
                    replicate,
                )
                proposal_seed = derive_seed(proposal_key)
                label_seed = derive_seed(label_key)
                seed_records.extend(
                    (
                        {"key": asdict(proposal_key), "seed": proposal_seed},
                        {"key": asdict(label_key), "seed": label_seed},
                    )
                )
                model = {
                    "spot": float(cell["spot"]),
                    "maturity": float(cell["maturity"]),
                    "xi": float(cell["xi"]),
                    "eta": float(cell["eta"]),
                    "rho": float(cell["rho"]),
                    "H": float(cell["hurst"]),
                }
                sample = _draw(
                    simulator=simulator,
                    controls=_controls(candidate, float(cell["maturity"])),
                    weights=torch.tensor(candidate["weights"], dtype=torch.float64),
                    model=model,
                    steps=int(cell["finest_steps"]),
                    count=paths,
                    proposal_seed=proposal_seed,
                    label_seed=label_seed,
                    engine=engine,
                )
                if (
                    not bool(torch.isfinite(sample.paths.spot).all())
                    or not bool(torch.isfinite(sample.paths.variance).all())
                    or bool((sample.paths.spot <= 0.0).any())
                    or bool((sample.paths.variance <= 0.0).any())
                ):
                    raise FloatingPointError("barrier proposal produced invalid paths")
                normalization = torch.exp(sample.mixture_log_likelihood)
                if not bool(torch.isfinite(normalization).all()):
                    raise FloatingPointError("barrier proposal likelihood is nonfinite")
                all_normalization_values.append(normalization.detach().cpu())
                task = _cell_task(cell)
                for method in REFERENCE_METHODS:
                    values = _method_values(
                        sample,
                        task=task,
                        rho=float(cell["rho"]),
                        method=method,
                    ).detach().to(device="cpu", dtype=torch.float64)
                    if not bool(torch.isfinite(values).all()):
                        raise FloatingPointError("barrier contribution is nonfinite")
                    method_variances[method].append(
                        float(torch.var(values, unbiased=True))
                    )
                    method_means[method].append(float(torch.mean(values)))
                    method_maxima[method].append(float(torch.max(torch.abs(values))))
                    if method == "raw_crosscheck":
                        raw_nonzero_counts.append(int(torch.count_nonzero(values)))
            target = (
                0.10 * 0.20 * float(cell["nominal_probability"])
            )
            for method in REFERENCE_METHODS:
                design_variance = max(method_variances[method])
                requested = max(
                    8192,
                    math.ceil(
                        float(gates["allocation_safety_factor"])
                        * design_variance
                        / (target * target)
                    ),
                )
                candidate_entries.append(
                    {
                        "cell_id": cell_id,
                        "method": method,
                        "target_standard_error": target,
                        "replicate_variances": method_variances[method],
                        "replicate_means": method_means[method],
                        "replicate_max_absolute_contributions": method_maxima[method],
                        "allocation_design_variance": design_variance,
                        "projected_final_samples": requested,
                        "requested_to_cap_ratio": requested
                        / int(gates["maximum_final_samples"]),
                        "projected_cap_pass": requested
                        <= int(gates["maximum_final_samples"]),
                        "raw_nonzero_counts": (
                            raw_nonzero_counts if method == "raw_crosscheck" else None
                        ),
                        "raw_coverage_pass": (
                            min(raw_nonzero_counts)
                            >= int(
                                gates[
                                    "minimum_raw_nonzero_contributions_per_replicate"
                                ]
                            )
                            if method == "raw_crosscheck"
                            else True
                        ),
                    }
                )
        normalization_statistics = SufficientStatistics.from_tensor(
            torch.cat(all_normalization_values)
        )
        normalization_z = (
            (normalization_statistics.mean - 1.0)
            / normalization_statistics.standard_error
            if normalization_statistics.standard_error > 0.0
            else (
                0.0 if normalization_statistics.mean == 1.0 else math.inf
            )
        )
        worst_ratio = max(
            float(entry["requested_to_cap_ratio"]) for entry in candidate_entries
        )
        candidate_gates = {
            "all_projected_caps_pass": all(
                entry["projected_cap_pass"] for entry in candidate_entries
            ),
            "all_raw_coverage_pass": all(
                entry["raw_coverage_pass"] for entry in candidate_entries
            ),
            "likelihood_normalization_pass": abs(normalization_z)
            <= float(gates["maximum_likelihood_normalization_absolute_z"]),
            "worst_ratio_pass": worst_ratio
            <= float(gates["selected_worst_requested_to_cap_ratio"]),
        }
        outputs.append(
            {
                "candidate_id": candidate["id"],
                "weights": candidate["weights"],
                "schedules": candidate["schedules"],
                "entries": candidate_entries,
                "normalization_mean": normalization_statistics.mean,
                "normalization_standard_error": (
                    normalization_statistics.standard_error
                ),
                "normalization_z": normalization_z,
                "worst_requested_to_cap_ratio": worst_ratio,
                "gates": candidate_gates,
                "passes": all(candidate_gates.values()),
            }
        )
    passing = [candidate for candidate in outputs if candidate["passes"]]
    selected = (
        min(passing, key=lambda item: item["worst_requested_to_cap_ratio"])
        if passing
        else None
    )
    provenance = source_provenance()
    return {
        "schema": RESULT_SCHEMA,
        "protocol_id": config["protocol_id"],
        "config_sha256": config_sha256,
        "namespace": config["namespace"],
        "smoke": smoke,
        "design_informed_by_prior_development_outcomes": True,
        "current_namespace_outcomes_inspected_before_freeze": False,
        "cells": cells,
        "replicates": replicates,
        "paths_per_replicate": paths,
        "candidates": outputs,
        "seed_records": seed_records,
        "seed_count": len(seed_records),
        "selected_candidate_id": (
            selected["candidate_id"] if selected is not None else None
        ),
        "passed": selected is not None,
        "decision": {
            "status": (
                "barrier_proposal_falsification_pass"
                if selected is not None
                else "barrier_proposal_falsification_fail"
            ),
            "selected_proposal_frozen": False,
            "new_full_pilot_authorized": False,
            "final_execution_authorized": False,
            "performance_claim_authorized": False,
        },
        "environment": runtime_provenance(dtype="torch.float64"),
        **provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(
            f"refusing to overwrite barrier proposal result: {arguments.output}"
        )
    result = run_falsification(arguments.config, smoke=arguments.smoke)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
