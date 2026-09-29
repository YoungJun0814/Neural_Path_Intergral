"""Run the frozen V14 exact conditional local-Volterra development matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from experiments.g11_v13_structured_ecrpt_development import (
    _allocate,
    _bound_cells,
    _combined_reference_z,
    _comparator_record,
    _moments_v13,
    _plugin_work_to_target,
    _problem,
    _sha256,
    _tail_forecast,
    _z_difference,
)
from src.path_integral.baseline_framework import BaselineCostLedger
from src.path_integral.baselines.conditional_rbergomi import (
    evaluate_conditional_terminal_units,
    freeze_conditional_rbergomi_proposal,
)
from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.baselines.smoothing_rqmc import (
    evaluate_smoothing_rqmc_units,
    freeze_smoothing_rqmc_proposal,
)
from src.path_integral.ecrpt_protocol import aggregate_local_volterra_development
from src.path_integral.provenance import runtime_provenance, source_provenance
from src.path_integral.rbergomi_local_volterra_transport import (
    LocalVolterraTransportTrainingConfig,
    evaluate_local_volterra_transport,
    train_local_volterra_transport,
)
from src.path_integral.residual_smc import AdaptiveResidualSMCConfig
from src.path_integral.residual_stability import rao_blackwell_variance_gap_integrand
from src.path_integral.seed_ledger import SeedLedger
from src.path_integral.tail_safe_allocation import BoundedRange
from src.path_integral.v10r1_full_latent_dcs import evaluate_full_latent_dcs
from src.path_integral.v10r1_proposal_bank import proposal_from_dict

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "npi.g11.v14-local-volterra-development.v1"


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(config, dict) or config.get("schema") != SCHEMA:
        raise ValueError("unexpected V14 development schema")
    config["config_path"] = str(path)
    return config, hashlib.sha256(raw).hexdigest()


def validate_config(config: dict[str, Any], *, root: Path = ROOT) -> None:
    if config.get("stage") != "development":
        raise ValueError("V14 runner accepts development evidence only")
    for name, binding in config.get("bindings", {}).items():
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise ValueError(f"invalid binding: {name}")
        path = root / str(binding["path"])
        if not path.is_file() or _sha256(path) != str(binding["sha256"]):
            raise ValueError(f"binding mismatch: {name}")
    if int(config["clusters"]) < 2:
        raise ValueError("V14 requires at least two independent clusters")
    points = int(config["final_budget"]["rqmc_points_per_randomization"])
    if points < 1 or points & (points - 1):
        raise ValueError("RQMC points per randomization must be a power of two")
    if bool(config["gate"]["distribution_free_tail_certificate_is_claim_gate"]):
        raise ValueError("V14 v1 freezes the distribution-free certificate as diagnostic")


def _training_config(
    config: dict[str, Any], cell: dict[str, Any]
) -> LocalVolterraTransportTrainingConfig:
    frozen = config["candidate_training"]
    return LocalVolterraTransportTrainingConfig(
        target_powers=tuple(float(x) for x in frozen["target_powers"]),
        shifted_weights=tuple(float(x) for x in frozen["shifted_weights"]),
        defensive_weight=float(frozen["defensive_weight"]),
        replicates_per_power=int(cell.get("training_replicates", 1)),
        smc=AdaptiveResidualSMCConfig(
            particles=int(frozen["smc_particles"]),
            target_ess_fraction=float(frozen["target_ess_fraction"]),
            pcn_scale=float(frozen["pcn_scale"]),
            pcn_sweeps_per_stage=int(frozen["pcn_sweeps_per_stage"]),
            maximum_stages=int(frozen["maximum_stages"]),
        ),
    )


def _candidate_record(
    problem: RBergomiBaselineProblem,
    cell: dict[str, Any],
    cluster: int,
    config: dict[str, Any],
    ledger: SeedLedger,
) -> tuple[dict[str, Any], float]:
    training = train_local_volterra_transport(
        problem,
        training_seed=_allocate(
            ledger,
            protocol_id=str(config["protocol_id"]),
            role="candidate-training",
            cell_id=problem.task_id,
            cluster=cluster,
            stream="local-volterra",
        ),
        config=_training_config(config, cell),
    )
    budget = config["final_budget"]
    final = evaluate_local_volterra_transport(
        problem,
        training.proposal,
        sample_count=int(budget["candidate_iid_paths"]),
        proposal_seed=_allocate(
            ledger,
            protocol_id=str(config["protocol_id"]),
            role="candidate-final",
            cell_id=problem.task_id,
            cluster=cluster,
            stream="proposal",
        ),
        coordinate_seed=_allocate(
            ledger,
            protocol_id=str(config["protocol_id"]),
            role="candidate-final",
            cell_id=problem.task_id,
            cluster=cluster,
            stream="coordinate",
        ),
    )
    estimate = _moments_v13(final.contribution)
    raw = _moments_v13(final.raw_contribution)
    difference = _moments_v13(final.raw_contribution - final.contribution)
    likelihood = _moments_v13(final.likelihood)
    gap = rao_blackwell_variance_gap_integrand(final.likelihood, final.conditional_probability)
    target_variance = (
        float(config["relative_rmse_target"]) * float(cell["reference_estimate"])
    ) ** 2
    one_time = training.proposal.training_cost.algorithmic_work_units
    unit_work = final.evaluation_cost.algorithmic_work_units / int(estimate["count"])
    return {
        "proposal": asdict(training.proposal),
        "target_powers": list(training.target_powers),
        "mean_norms": list(training.mean_norms),
        "replicate_mean_norms": [list(values) for values in training.replicate_mean_norms],
        "replicate_mean_minimum_cosines": list(training.replicate_mean_minimum_cosines),
        "all_training_seeds": list(training.all_training_seeds),
        "smc": [
            {
                "root_seed": result.root_seed,
                "used_seeds": list(result.used_seeds),
                "final_beta": result.final_beta,
                "final_particles_equally_weighted": result.final_particles_equally_weighted,
                "particles_are_final_inferential_units": result.particles_are_final_inferential_units,
                "stages": [asdict(stage) for stage in result.stages],
            }
            for result in training.smc_results
        ],
        "charged_one_time_algorithmic_work_units": one_time,
        "estimate": estimate,
        "raw_estimate": raw,
        "difference": difference,
        "likelihood": likelihood,
        "paired_difference_z": _z_difference(
            float(difference["estimate"]), float(difference["standard_error"])
        ),
        "likelihood_normalization_z": _z_difference(
            float(likelihood["estimate"]) - 1.0,
            float(likelihood["standard_error"]),
        ),
        "combined_reference_z": _combined_reference_z(estimate, cell),
        "raw_over_ecrpt_variance_ratio": float(raw["variance"]) / float(estimate["variance"])
        if float(estimate["variance"]) > 0.0
        else 0.0,
        "rao_blackwell_gap_estimate": float(torch.mean(gap)),
        "rao_blackwell_gap_minimum": float(torch.amin(gap)),
        "exactness": {
            "maximum_likelihood_bound_violation": final.maximum_likelihood_bound_violation
        },
        "evaluation_cost": asdict(final.evaluation_cost),
        "plugin_work_to_target": _plugin_work_to_target(
            unit_variance=float(estimate["variance"]),
            unit_work=unit_work,
            one_time_work=one_time,
            target_variance=target_variance,
            query_count=int(config["primary_query_count"]),
            minimum_units=int(config["tail_safe_policy"]["minimum_iid_units"]),
        ),
        "tail_safe_forecast": _tail_forecast(
            final.contribution,
            bounds=BoundedRange(0.0, 1.0 / float(training.proposal.component_weights[0])),
            target_variance=target_variance,
            unit_kind="iid_path",
            config=config,
        ),
    }, target_variance


def _record(
    problem: RBergomiBaselineProblem,
    cell: dict[str, Any],
    cluster: int,
    config: dict[str, Any],
    ledger: SeedLedger,
    bank_entry: dict[str, Any],
) -> dict[str, Any]:
    candidate, target_variance = _candidate_record(problem, cell, cluster, config, ledger)
    budget = config["final_budget"]
    conditional_proposal = freeze_conditional_rbergomi_proposal(
        problem,
        training_seed=_allocate(
            ledger,
            protocol_id=str(config["protocol_id"]),
            role="comparator-freeze",
            cell_id=problem.task_id,
            cluster=cluster,
            stream="conditional",
        ),
    )
    conditional = evaluate_conditional_terminal_units(
        problem,
        conditional_proposal,
        sample_count=int(budget["conditional_iid_paths"]),
        seed=_allocate(
            ledger,
            protocol_id=str(config["protocol_id"]),
            role="comparator-final",
            cell_id=problem.task_id,
            cluster=cluster,
            stream="conditional",
        ),
    )
    conditional_cost = BaselineCostLedger(
        final_samples=conditional.raw_sample_count,
        cdf_calls=conditional.cdf_calls,
        algorithmic_work_units=float(
            conditional.raw_sample_count * (problem.local_dimension + problem.steps + 1)
        ),
    )
    rqmc_proposal = freeze_smoothing_rqmc_proposal(
        problem,
        training_seed=_allocate(
            ledger,
            protocol_id=str(config["protocol_id"]),
            role="comparator-freeze",
            cell_id=problem.task_id,
            cluster=cluster,
            stream="rqmc",
        ),
    )
    rqmc = evaluate_smoothing_rqmc_units(
        problem,
        rqmc_proposal,
        randomizations=int(budget["rqmc_randomizations"]),
        points_per_randomization=int(budget["rqmc_points_per_randomization"]),
        seed=_allocate(
            ledger,
            protocol_id=str(config["protocol_id"]),
            role="comparator-final",
            cell_id=problem.task_id,
            cluster=cluster,
            stream="rqmc",
        ),
    )
    rqmc_cost = BaselineCostLedger(
        final_samples=rqmc.raw_sample_count,
        cdf_calls=rqmc.cdf_calls,
        algorithmic_work_units=float(
            rqmc.raw_sample_count * (problem.latent_dimension + problem.steps + 1)
        ),
    )
    v10_proposal = proposal_from_dict(bank_entry["proposal"])
    v10 = evaluate_full_latent_dcs(
        problem,
        v10_proposal,
        sample_count=int(budget["v10r1_iid_paths"]),
        gaussian_seed=_allocate(
            ledger,
            protocol_id=str(config["protocol_id"]),
            role="comparator-final",
            cell_id=problem.task_id,
            cluster=cluster,
            stream="v10-gaussian",
        ),
        label_seed=_allocate(
            ledger,
            protocol_id=str(config["protocol_id"]),
            role="comparator-final",
            cell_id=problem.task_id,
            cluster=cluster,
            stream="v10-label",
        ),
    )
    comparators = {
        "conditional_rbergomi": _comparator_record(
            "conditional_rbergomi",
            conditional.unit_contributions,
            conditional_cost,
            0.0,
            BoundedRange(0.0, 1.0),
            "iid_path",
            cell,
            config,
            target_variance,
        ),
        "smoothing_rqmc": _comparator_record(
            "smoothing_rqmc",
            rqmc.unit_contributions,
            rqmc_cost,
            0.0,
            BoundedRange(0.0, 1.0),
            "rqmc_randomization",
            cell,
            config,
            target_variance,
        ),
        "v10r1_full_cem_dcs": _comparator_record(
            "v10r1_full_cem_dcs",
            v10.dcs_contribution,
            v10.dcs_cost,
            v10_proposal.training_cost.algorithmic_work_units,
            BoundedRange(0.0, 1.0 / float(v10_proposal.component_weights[0])),
            "iid_path",
            cell,
            config,
            target_variance,
        ),
    }
    return {
        "cell_id": problem.task_id,
        "cluster": cluster,
        "hurst": float(cell["hurst"]),
        "nominal_probability": float(cell["nominal_probability"]),
        "threshold": float(cell["threshold"]),
        "reference_estimate": float(cell["reference_estimate"]),
        "reference_standard_error": float(cell["reference_standard_error"]),
        "target_estimator_variance": target_variance,
        "candidate": candidate,
        "comparators": comparators,
    }


def run(config: dict[str, Any], config_sha256: str, *, root: Path = ROOT) -> dict[str, Any]:
    validate_config(config, root=root)
    cells = _bound_cells(config, root=root)
    bank = json.loads(
        (root / config["bindings"]["v10r1_proposal_bank"]["path"]).read_text(encoding="utf-8")
    )
    bank_index = {
        (str(entry["cell_id"]), int(entry["replicate"])): entry for entry in bank["entries"]
    }
    ledger = SeedLedger()
    records = []
    for cell in cells:
        problem = _problem(config, cell)
        for cluster in range(int(config["clusters"])):
            key = (problem.task_id, cluster)
            if key not in bank_index:
                raise ValueError(f"V10R1 bank lacks {key}")
            records.append(
                _record(
                    problem,
                    cell,
                    cluster,
                    config,
                    ledger,
                    bank_index[key],
                )
            )
    aggregate = aggregate_local_volterra_development(config=config, records=records)
    return {
        "schema": "npi.g11.v14-local-volterra-development-result.v1",
        "protocol_id": config["protocol_id"],
        "namespace": config["namespace"],
        "stage": "development",
        "config_path": str(config["config_path"]),
        "config_sha256": config_sha256,
        **source_provenance(),
        "environment": runtime_provenance(dtype="torch.float64"),
        "bindings": config["bindings"],
        "seed_ledger": ledger.to_dict(),
        "seed_ledger_sha256": ledger.sha256,
        "records": records,
        "aggregate": aggregate,
        "decision": {
            "software_correctness_evidence_pass": bool(aggregate["correctness_pass"]),
            "development_performance_gate_pass": bool(aggregate["performance_pass"]),
            "qualification_authorized": bool(aggregate["stage_pass"]),
            "qualification_executed": False,
            "distribution_free_tail_claim_authorized": bool(
                aggregate["distribution_free_tail_certificate_pass"]
            ),
            "broad_performance_claim_authorized": False,
            "top_journal_claim_authorized": False,
            "submission_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config, digest = load_config(args.config)
    result = run(config, digest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
