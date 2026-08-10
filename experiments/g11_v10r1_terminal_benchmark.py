"""Execute the corrected full-3N defensive CEM plus DCS development matrix."""

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
from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.provenance import runtime_provenance, source_provenance
from src.path_integral.v10r1_full_latent_dcs import evaluate_full_latent_dcs
from src.path_integral.v10r1_proposal_bank import proposal_from_dict
from src.path_integral.v10r1_protocol import aggregate_v10r1, attach_v10r1_work

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "npi.g11.v10r1-terminal-benchmark.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(config, dict) or config.get("schema") != SCHEMA:
        raise ValueError("unexpected V10R1 benchmark schema")
    return config, hashlib.sha256(raw).hexdigest()


def validate_config(config: dict[str, Any], *, root: Path = ROOT) -> None:
    bindings = config.get("bindings")
    if not isinstance(bindings, dict):
        raise ValueError("V10R1 benchmark requires exact artifact bindings")
    for name, binding in bindings.items():
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise ValueError(f"invalid V10R1 binding: {name}")
        path = root / str(binding["path"])
        if not path.is_file() or _sha256(path) != str(binding["sha256"]):
            raise ValueError(f"V10R1 binding mismatch: {name}")
    bank_audit = json.loads(
        (root / str(bindings["proposal_bank_audit"]["path"])).read_text(encoding="utf-8")
    )
    if (
        bank_audit.get("passed") is not True
        or bank_audit.get("replay_performed") is not True
        or bank_audit.get("decision", {}).get("development_authorized") is not True
    ):
        raise ValueError("V10R1 development requires a replay-passed bank audit")
    claim = yaml.safe_load(
        (root / str(bindings["claim_contract"]["path"])).read_text(encoding="utf-8")
    )
    if (
        claim.get("schema") != "npi.g11.v10r1-terminal-claim-contract.v1"
        or config.get("model") != claim.get("model")
        or config.get("relative_rmse_target") != claim.get("relative_rmse_target")
        or config.get("amortization_query_counts")
        != claim.get("amortization_query_counts")
        or config.get("primary_query_count") != claim.get("primary_query_count")
    ):
        raise ValueError("V10R1 benchmark differs from its claim contract")
    reference_audit = json.loads(
        (root / str(bindings["reference_audit"]["path"])).read_text(
            encoding="utf-8"
        )
    )
    if (
        reference_audit.get("passed") is not True
        or reference_audit.get("dirty_worktree") is not False
        or reference_audit.get("benchmark_authorized") is not True
    ):
        raise ValueError("V10R1 requires a clean-source passing reference re-audit")
    if config.get("current_namespace_outcomes_inspected_before_freeze") is not False:
        raise ValueError("V10R1 development namespace was not outcome-blind at freeze")
    if int(config["clusters"]) != int(config["proposal_replicates"]):
        raise ValueError("each inferential cluster requires an independent proposal replicate")
    if int(config["proposal_replicates"]) != int(claim["proposal_replicates"]):
        raise ValueError("proposal-replicate count differs from the claim contract")
    candidate_training = config.get("candidate_training")
    comparator_training = config.get("external_budget", {}).get("cem")
    if not isinstance(candidate_training, dict) or candidate_training != comparator_training:
        raise ValueError("candidate and defensive-CEM comparator training budgets differ")
    if candidate_training.get("time_bins", None) is not None:
        raise ValueError("V10R1 CEM training must keep the complete 3N mean")
    bank = json.loads(
        (root / str(bindings["proposal_bank"]["path"])).read_text(encoding="utf-8")
    )
    if int(config["proposal_replicates"]) * len(claim["cells"]) != int(
        bank["entry_count"]
    ):
        raise ValueError("proposal bank does not cover the development roster")


def _bound_cells(config: dict[str, Any], *, root: Path = ROOT) -> list[dict[str, Any]]:
    bindings = config["bindings"]
    reference = json.loads(
        (root / str(bindings["reference"]["path"])).read_text(encoding="utf-8")
    )
    references = {str(cell["cell_id"]): cell for cell in reference["cells"]}
    claim = yaml.safe_load(
        (root / str(bindings["claim_contract"]["path"])).read_text(encoding="utf-8")
    )
    cells: list[dict[str, Any]] = []
    for design in claim["cells"]:
        cell_id = str(design["cell_id"])
        ref = references[cell_id]
        cells.append(
            {
                **design,
                "reference_estimate": float(ref["estimate"]),
                "reference_standard_error": float(ref["standard_error"]),
            }
        )
    return cells


def _problem(config: dict[str, Any], cell: dict[str, Any]) -> RBergomiBaselineProblem:
    model = config["model"]
    return RBergomiBaselineProblem(
        task_id=str(cell["cell_id"]),
        task=TerminalThresholdTask(float(cell["threshold"])),
        spot=float(model["spot"]),
        maturity=float(model["maturity"]),
        steps=int(model["steps"]),
        hurst=float(cell["hurst"]),
        eta=float(model["eta"]),
        xi=float(model["xi"]),
        rho=float(model["rho"]),
    )


def _moments(values: torch.Tensor) -> tuple[float, float, float]:
    mean = float(torch.mean(values))
    variance = float(torch.var(values, unbiased=True))
    return mean, variance, math.sqrt(variance / values.numel())


def _paired_record(
    *,
    config: dict[str, Any],
    cell: dict[str, Any],
    cluster: int,
    seeds: tuple[int, int],
    bank_entry: dict[str, Any],
) -> dict[str, Any]:
    problem = _problem(config, cell)
    proposal = proposal_from_dict(bank_entry["proposal"])
    batch = evaluate_full_latent_dcs(
        problem,
        proposal,
        sample_count=int(config["paired_paths"]),
        gaussian_seed=seeds[0],
        label_seed=seeds[1],
    )
    raw_mean, raw_variance, raw_se = _moments(batch.raw_contribution)
    dcs_mean, dcs_variance, dcs_se = _moments(batch.dcs_contribution)
    difference = batch.raw_contribution - batch.dcs_contribution
    difference_mean, difference_variance, difference_se = _moments(difference)
    norm_mean, norm_variance, norm_se = _moments(batch.likelihood_normalization)
    norm_z = (
        (norm_mean - 1.0) / norm_se
        if norm_se > 0.0
        else (0.0 if norm_mean == 1.0 else math.inf)
    )
    reference = float(cell["reference_estimate"])
    reference_se = float(cell["reference_standard_error"])
    return {
        "cell_id": cell["cell_id"],
        "hurst": float(cell["hurst"]),
        "nominal_probability": float(cell["nominal_probability"]),
        "reference_estimate": reference,
        "reference_standard_error": reference_se,
        "budget_id": "high",
        "cluster": cluster,
        "proposal_replicate": int(bank_entry["replicate"]),
        "proposal_training_seed": int(bank_entry["training_seed"]),
        "proposal_sha256": proposal.sha256,
        "proposal_training_cost": asdict(proposal.training_cost),
        "gaussian_seed": seeds[0],
        "label_seed": seeds[1],
        "sample_count": batch.raw_contribution.numel(),
        "component_counts": batch.component_counts,
        "integration_direction_sha256": hashlib.sha256(
            batch.integration_direction.numpy().tobytes()
        ).hexdigest(),
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
            "variance_ratio_raw_over_dcs": raw_variance / dcs_variance,
            "difference_mean": difference_mean,
            "difference_variance": difference_variance,
            "difference_standard_error": difference_se,
            "difference_z": abs(difference_mean) / difference_se
            if difference_se > 0.0
            else (0.0 if difference_mean == 0.0 else math.inf),
        },
        "likelihood": {
            "normalization_mean": norm_mean,
            "normalization_variance": norm_variance,
            "normalization_standard_error": norm_se,
            "normalization_z": norm_z,
            "log_weight_minimum": float(torch.amin(batch.log_likelihood)),
            "log_weight_median": float(torch.quantile(batch.log_likelihood, 0.5)),
            "log_weight_q99": float(torch.quantile(batch.log_likelihood, 0.99)),
            "log_weight_maximum": float(torch.amax(batch.log_likelihood)),
        },
        "exactness": {
            "maximum_local_latent_reconstruction_error": batch.maximum_local_latent_reconstruction_error,
            "maximum_price_latent_reconstruction_error": batch.maximum_price_latent_reconstruction_error,
            "maximum_path_reconstruction_error": batch.maximum_path_reconstruction_error,
            "maximum_coordinate_error": batch.maximum_coordinate_error,
            "maximum_component_density_error": batch.maximum_component_density_error,
            "maximum_mixture_density_error": batch.maximum_mixture_density_error,
            "maximum_full_likelihood_error": batch.maximum_full_likelihood_error,
            "maximum_full_bound_violation": batch.maximum_full_bound_violation,
            "maximum_residual_bound_violation": batch.maximum_residual_bound_violation,
        },
    }


def run(config: dict[str, Any], config_sha256: str, *, root: Path = ROOT) -> dict[str, Any]:
    validate_config(config, root=root)
    provenance = source_provenance()
    if provenance["dirty_worktree"] is not False:
        raise RuntimeError("V10R1 benchmark requires a clean committed source tree")
    bank = json.loads(
        (root / str(config["bindings"]["proposal_bank"]["path"])).read_text(
            encoding="utf-8"
        )
    )
    bank_index = {
        (str(entry["cell_id"]), int(entry["replicate"])): entry
        for entry in bank["entries"]
    }
    cells = _bound_cells(config, root=root)
    paired: list[dict[str, Any]] = []
    external: list[dict[str, Any]] = []
    used_seeds: list[int] = []
    cursor = int(config["base_seed"])

    def allocate(count: int) -> tuple[int, ...]:
        nonlocal cursor
        values = tuple(range(cursor, cursor + count))
        cursor += count
        used_seeds.extend(values)
        return values

    methods = list(config["external_methods"]["primary"])
    budget = config["external_budget"]
    for cell in cells:
        for cluster in range(int(config["clusters"])):
            paired.append(
                _paired_record(
                    config=config,
                    cell=cell,
                    cluster=cluster,
                    seeds=cast(tuple[int, int], allocate(2)),
                    bank_entry=bank_index[(str(cell["cell_id"]), cluster)],
                )
            )
            for method in methods:
                method_budget = dict(budget)
                method_budget["pilot_units"] = int(
                    budget["pilot_units_by_method"][method]
                )
                record = _external_record(
                    config,
                    cell,
                    method_budget,
                    method,
                    cluster,
                    cast(tuple[int, int, int, int], allocate(4)),
                )
                record["hurst"] = float(cell["hurst"])
                record["nominal_probability"] = float(cell["nominal_probability"])
                record["reference_estimate"] = float(cell["reference_estimate"])
                record["reference_standard_error"] = float(cell["reference_standard_error"])
                external.append(record)
    paired, external = attach_v10r1_work(
        config=config,
        paired_records=paired,
        external_records=external,
        bank=bank,
    )
    aggregate = aggregate_v10r1(
        config=config,
        paired_records=paired,
        external_records=external,
    )
    seed_payload = json.dumps(used_seeds, separators=(",", ":")).encode()
    development = config["stage"] == "development"
    decision = {
        "development_complete": development,
        "qualification_authorized": development and bool(aggregate["stage_pass"]),
        "qualification_complete": not development,
        "regime_conditional_empirical_claim_authorized": (
            not development and bool(aggregate["stage_pass"])
        ),
        "broad_performance_claim_authorized": False,
        "top_journal_claim_authorized": False,
        "submission_authorized": False,
    }
    return {
        "schema": "npi.g11.v10r1-terminal-benchmark-result.v1",
        "protocol_id": config["protocol_id"],
        "namespace": config["namespace"],
        "stage": config["stage"],
        "config_path": str(config["config_path"]),
        "config_sha256": config_sha256,
        **provenance,
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
        raise FileExistsError("V10R1 benchmark outputs are immutable")
    config, digest = load_config(args.config)
    config["config_path"] = args.config.resolve().relative_to(ROOT).as_posix()
    result = run(config, digest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
