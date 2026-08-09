"""Execute V10 Full-Dimensional Defensive CEM + DCS terminal benchmark matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch
import yaml

from experiments.g11_v8_d1_p7_falsification_stage_a import _external_record
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.provenance import runtime_provenance, source_provenance
from src.path_integral.v9_terminal_protocol import (
    aggregate_terminal_benchmark,
    attach_work_to_target,
)
from src.path_integral.v10_full_cem_dcs_protocol import evaluate_v10_paired_dcs_benchmark
from src.path_integral.v10_proposal_bank import V10ProposalBankEntry

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "npi.g11.v10-terminal-benchmark.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(config, dict) or config.get("schema") != SCHEMA:
        raise ValueError("unexpected V10 terminal benchmark config schema")
    return config, hashlib.sha256(raw).hexdigest()


def _validate_bindings_and_source(config: dict[str, Any]) -> None:
    for name, binding in config.get("bindings", {}).items():
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise ValueError(f"invalid V10 benchmark binding: {name}")
        path = ROOT / str(binding["path"])
        if not path.is_file() or _sha256(path) != str(binding["sha256"]):
            raise ValueError(f"V10 benchmark binding mismatch: {name}")


def _bound_cells(config: dict[str, Any]) -> list[dict[str, Any]]:
    reference = json.loads(
        (ROOT / str(config["bindings"]["reference"]["path"])).read_text(encoding="utf-8")
    )
    references = {str(cell["cell_id"]): cell for cell in reference["cells"]}
    claim = yaml.safe_load(
        (ROOT / str(config["bindings"]["claim_contract"]["path"])).read_text(
            encoding="utf-8"
        )
    )
    claim_cells = {str(cell["cell_id"]): cell for cell in claim["cells"]}
    cells: list[dict[str, Any]] = []
    for cell_id, design in claim_cells.items():
        reference_cell = references[cell_id]
        cells.append(
            {
                **design,
                "reference_estimate": float(reference_cell["estimate"]),
                "reference_standard_error": float(reference_cell["standard_error"]),
            }
        )
    return cells


def _moments(values: torch.Tensor) -> tuple[float, float, float]:
    mean = float(torch.mean(values))
    variance = float(torch.var(values, unbiased=True))
    return mean, variance, float(torch.std(values, unbiased=True) / math.sqrt(values.numel()))


def _paired_v10_record(
    config: dict[str, Any],
    cell: dict[str, Any],
    budget: dict[str, Any],
    cluster: int,
    seeds: tuple[int, int],
    bank_entries: dict[str, V10ProposalBankEntry],
) -> dict[str, Any]:
    entry = bank_entries[str(cell["cell_id"])]
    task = TerminalThresholdTask(float(cell["threshold"]))
    batch = evaluate_v10_paired_dcs_benchmark(
        entry=entry,
        task=task,
        spot=float(config["model"]["spot"]),
        maturity=float(config["model"]["maturity"]),
        steps=int(config["model"]["steps"]),
        eta=float(config["model"]["eta"]),
        xi=float(config["model"]["xi"]),
        rho=float(config["model"]["rho"]),
        sample_count=int(budget["paired_dcs_paths"]),
        path_seed=seeds[0],
        label_seed=seeds[1],
    )
    raw_mean, raw_variance, raw_se = _moments(batch.raw_contribution)
    dcs_mean, dcs_variance, dcs_se = _moments(batch.dcs_contribution)
    difference = batch.raw_contribution - batch.dcs_contribution
    difference_mean, difference_variance, difference_se = _moments(difference)
    normalization_mean, normalization_variance, normalization_se = _moments(
        batch.likelihood_normalization
    )
    normalization_z = (
        (normalization_mean - 1.0) / normalization_se
        if normalization_se > 0.0
        else (0.0 if normalization_mean == 1.0 else math.inf)
    )
    reference = float(cell["reference_estimate"])
    reference_se = float(cell["reference_standard_error"])
    return {
        "cell_id": cell["cell_id"],
        "hurst": float(cell["hurst"]),
        "nominal_probability": float(cell["nominal_probability"]),
        "reference_estimate": reference,
        "reference_standard_error": reference_se,
        "budget_id": budget["id"],
        "cluster": cluster,
        "path_seed": seeds[0],
        "label_seed": seeds[1],
        "sample_count": batch.raw_contribution.numel(),
        "proposal_source": entry.cell_id,
        "proposal_training_cost": entry.training_cost,
        "proposal_training_budget_work_units": entry.training_budget_work_units,
        "component_counts": batch.component_counts,
        "raw": {
            "estimate": raw_mean,
            "variance": raw_variance,
            "standard_error": raw_se,
            "combined_reference_z": abs(raw_mean - reference)
            / math.sqrt(raw_se**2 + reference_se**2),
            "cost": asdict(batch.raw_cost),
        },
        "dcs": {
            "estimate": dcs_mean,
            "variance": dcs_variance,
            "standard_error": dcs_se,
            "combined_reference_z": abs(dcs_mean - reference)
            / math.sqrt(dcs_se**2 + reference_se**2),
            "cost": asdict(batch.dcs_cost),
        },
        "mechanism": {
            "variance_ratio_raw_over_dcs": raw_variance / dcs_variance
            if dcs_variance > 0.0
            else math.inf,
            "difference_mean": difference_mean,
            "difference_variance": difference_variance,
            "difference_standard_error": difference_se,
            "difference_z": abs(difference_mean) / difference_se
            if difference_se > 0.0
            else (0.0 if difference_mean == 0.0 else math.inf),
        },
        "likelihood": {
            "normalization_mean": normalization_mean,
            "normalization_variance": normalization_variance,
            "normalization_standard_error": normalization_se,
            "normalization_z": normalization_z,
            "log_weight_minimum": float(torch.amin(batch.log_likelihood)),
            "log_weight_median": float(torch.quantile(batch.log_likelihood, 0.5)),
            "log_weight_q99": float(torch.quantile(batch.log_likelihood, 0.99)),
            "log_weight_maximum": float(torch.amax(batch.log_likelihood)),
        },
        "exactness": {
            "maximum_path_reconstruction_error": batch.maximum_path_reconstruction_error,
            "maximum_component_density_error": batch.maximum_component_density_error,
            "maximum_mixture_density_error": batch.maximum_mixture_density_error,
            "maximum_full_likelihood_error": batch.maximum_full_likelihood_error,
        },
    }


def run(config: dict[str, Any], config_sha256: str) -> dict[str, Any]:
    _validate_bindings_and_source(config)
    bank_data = json.loads(
        (ROOT / str(config["bindings"]["dcs_proposal_bank"]["path"])).read_text(
            encoding="utf-8"
        )
    )
    bank_entries = {
        entry["cell_id"]: V10ProposalBankEntry(
            cell_id=entry["cell_id"],
            hurst=entry["hurst"],
            dimension=entry["dimension"],
            learned_mean=tuple(entry["learned_mean"]),
            defensive_weight=entry["defensive_weight"],
            training_cost=entry["training_cost"],
            training_budget_work_units=entry["training_budget_work_units"],
        )
        for entry in bank_data["entries"]
    }

    cells = _bound_cells(config)
    paired: list[dict[str, Any]] = []
    external: list[dict[str, Any]] = []
    used_seeds: set[int] = set()
    cursor = int(config["base_seed"])

    def allocate(count: int) -> tuple[int, ...]:
        nonlocal cursor
        values = tuple(range(cursor, cursor + count))
        cursor += count
        used_seeds.update(values)
        return values

    budget = config["budgets"][0]
    methods = list(config["external_methods"]["primary"])
    for cell in cells:
        for cluster in range(int(config["clusters"])):
            paired.append(
                _paired_v10_record(
                    config,
                    cell,
                    budget,
                    cluster,
                    cast(tuple[int, int], allocate(2)),
                    bank_entries,
                )
            )
            for method in methods:
                method_budget = dict(budget)
                if "pilot_units_by_method" in budget:
                    method_budget["pilot_units"] = int(
                        budget["pilot_units_by_method"][method]
                    )
                ext_rec = _external_record(
                    config,
                    cell,
                    method_budget,
                    method,
                    cluster,
                    cast(tuple[int, int, int, int], allocate(4)),
                )
                ext_rec["hurst"] = float(cell["hurst"])
                ext_rec["nominal_probability"] = float(cell["nominal_probability"])
                ext_rec["reference_estimate"] = float(cell["reference_estimate"])
                ext_rec["reference_standard_error"] = float(cell["reference_standard_error"])
                external.append(ext_rec)

    paired, external = attach_work_to_target(
        config=config,
        paired_records=paired,
        external_records=external,
        bank=bank_data,
    )
    aggregate = aggregate_terminal_benchmark(
        config=config,
        paired_records=paired,
        external_records=external,
    )
    seed_payload = json.dumps(sorted(used_seeds), separators=(",", ":")).encode()
    development = config["stage"] == "development"
    decision = {
        "development_complete": development,
        "qualification_authorized": development and aggregate["stage_pass"],
        "qualification_complete": not development,
        "regime_conditional_empirical_claim_authorized": (
            not development and aggregate["stage_pass"]
        ),
        "broad_performance_claim_authorized": aggregate["stage_pass"],
        "top_journal_claim_authorized": aggregate["stage_pass"],
        "submission_authorized": aggregate["stage_pass"],
    }
    return {
        "schema": "npi.g11.v10-terminal-benchmark-result.v1",
        "protocol_id": config["protocol_id"],
        "namespace": config["namespace"],
        "stage": config["stage"],
        "config_sha256": config_sha256,
        **source_provenance(),
        "environment": runtime_provenance(dtype="torch.float64"),
        "seed_count": len(used_seeds),
        "seed_set_sha256": hashlib.sha256(seed_payload).hexdigest(),
        "paired_records": paired,
        "external_records": external,
        "aggregate": aggregate,
        "decision": decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        args.output.unlink()
    config, digest = load_config(args.config)
    result = run(config, digest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
