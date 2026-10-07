"""Small recorded end-to-end measurement; no performance claim is made."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import yaml

from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.baselines.weighted_conditional_ce import (
    WeightedConditionalCEConfig,
    evaluate_weighted_conditional_ce,
    proposal_parameters,
    train_weighted_conditional_ce,
)
from src.path_integral.comparator_qualification import (
    QualificationPolicy,
    qualify_estimate,
    work_normalized_variance,
)
from src.path_integral.finite_rank_gaussian_transport import (
    DefensiveFiniteRankGaussianMixture,
    FiniteRankGaussianComponent,
)
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.research_cost_accounting import measure_stage
from src.path_integral.research_result_audit import audit_measurement
from src.path_integral.research_result_contract import (
    SCHEMA,
    Moments,
    StageCost,
    canonical_digest,
    source_manifest,
)
from src.path_integral.seed_ledger import SeedKey, SeedLedger
from src.path_integral.volterra_conditional_payoffs import (
    evaluate_rbergomi_conditional_terminal,
)

ROOT = Path(__file__).resolve().parents[1]


def _combined(items: list[Moments]) -> Moments:
    result = items[0]
    for item in items[1:]:
        result = result.merge(item)
    return result


def _normalization_z(moments: Moments) -> float | None:
    se = moments.standard_error
    if se == 0:
        return 0.0 if moments.mean == 1.0 else None
    return abs(moments.mean - 1.0) / se


def _seed_record(
    ledger: SeedLedger, *, task: str, role: str, stream: str,
    level: int, replicate: int,
) -> dict[str, Any]:
    key = SeedKey("post-audit-r0-smoke", role, "r0-smoke", task, level, replicate, stream)
    return {"seed_key": key.__dict__, "seed": ledger.allocate(key)}


def build_smoke(config: dict[str, Any]) -> dict[str, Any]:
    task = config["task"]
    problem = RBergomiBaselineProblem(
        task_id=str(task["task_id"]),
        task=TerminalThresholdTask(level=float(task["strike"])),
        spot=float(task["spot"]), maturity=float(task["maturity"]),
        steps=int(task["steps"]), hurst=float(task["hurst"]),
        eta=float(task["eta"]), xi=float(task["xi"]), rho=float(task["rho"]),
    )
    training_rep = int(config["training_rep"])
    ledger = SeedLedger()
    reference_seed = _seed_record(
        ledger, task=problem.task_id, role="reference", stream="reference",
        level=0, replicate=training_rep,
    )

    def reference_action() -> torch.Tensor:
        generator = torch.Generator(device="cpu").manual_seed(reference_seed["seed"])
        local = torch.randn(
            (int(config["reference_units"]), problem.local_dimension),
            dtype=torch.float64, generator=generator,
        )
        return evaluate_rbergomi_conditional_terminal(problem, local).payoffs.left_probability

    reference_values, measured_reference_cost = measure_stage("reference", reference_action)
    reference_cost = StageCost(
        "reference", measured_reference_cost.wall_seconds,
        measured_reference_cost.cpu_seconds, measured_reference_cost.peak_memory_bytes,
        int(config["reference_units"]) * (problem.local_dimension + problem.steps),
    )
    reference = Moments.from_values(reference_values)
    ce_seed = _seed_record(
        ledger, task=problem.task_id, role="training", stream="weighted_conditional_ce",
        level=0, replicate=training_rep,
    )
    fit = train_weighted_conditional_ce(
        problem, training_seed=ce_seed["seed"],
        config=WeightedConditionalCEConfig(**config["ce"]),
    )
    natural = DefensiveFiniteRankGaussianMixture(
        (FiniteRankGaussianComponent.natural(problem.local_dimension),),
        torch.ones(1, dtype=torch.float64),
    )
    policy = QualificationPolicy(**config["qualification"])
    methods: list[dict[str, Any]] = []
    for method_id, proposal, fit_cost, fit_seed, substreams in (
        (
            "conditional_natural", natural,
            StageCost("fit", 0.0, 0.0, 0, 0.0), None, None,
        ),
        (
            "weighted_conditional_ce", fit.proposal, fit.cost,
            ce_seed, fit.seed_ledger,
        ),
    ):
        clusters: list[dict[str, Any]] = []
        value_moments: list[Moments] = []
        weight_moments: list[Moments] = []
        for index in range(int(config["evaluation"]["clusters"])):
            path_seed = _seed_record(
                ledger, task=problem.task_id, role="final", stream=f"{method_id}:path",
                level=0, replicate=index,
            )
            label_seed = _seed_record(
                ledger, task=problem.task_id, role="final", stream=f"{method_id}:label",
                level=0, replicate=index,
            )
            contributions, weights, cost = evaluate_weighted_conditional_ce(
                problem, proposal,
                sample_count=int(config["evaluation"]["units_per_cluster"]),
                path_seed=path_seed["seed"], label_seed=label_seed["seed"],
            )
            values = Moments.from_values(contributions)
            normalization = Moments.from_values(weights)
            clusters.append({
                "evaluation_rep": index, **path_seed,
                "label_seed_key": label_seed["seed_key"], "label_seed": label_seed["seed"],
                "moments": values.to_dict(),
                "normalization_moments": normalization.to_dict(),
                "raw_sample_count": values.count,
                "maximum_contribution": float(torch.max(contributions)),
                "cost": cost.to_dict(),
            })
            value_moments.append(values)
            weight_moments.append(normalization)
        combined = _combined(value_moments)
        normalized = _combined(weight_moments)
        qualify = qualify_estimate(
            estimate=combined.mean, standard_error=combined.standard_error,
            reference_estimate=reference.mean,
            reference_standard_error=reference.standard_error,
            reference_independent=True, policy=policy,
        )
        total_wall = fit_cost.wall_seconds + sum(
            cluster["cost"]["wall_seconds"] for cluster in clusters
        )
        total_cpu = fit_cost.cpu_seconds + sum(
            cluster["cost"]["cpu_seconds"] for cluster in clusters
        )
        peak_memory = max(
            [fit_cost.peak_memory_bytes] + [
                cluster["cost"]["peak_memory_bytes"] for cluster in clusters
            ]
        )
        total_work = fit_cost.proxy_work_units + sum(
            cluster["cost"]["proxy_work_units"] for cluster in clusters
        )
        methods.append({
            "method_id": method_id,
            "proposal_digest": fit.proposal_digest if fit_seed else canonical_digest(
                proposal_parameters(proposal)
            ),
            "proposal_parameters": proposal_parameters(proposal),
            "payoff_id": "terminal_left_conditional_cdf",
            "dimension": proposal.dimension,
            "inferential_unit": "iid_path",
            "points_per_unit": 1,
            "defensive_mass": proposal.defensive_mass,
            "fitted": fit_seed is not None,
            "fit_seed": fit_seed,
            "training_substream_ledger": substreams,
            "offline_cost": StageCost("offline", 0.0, 0.0, 0, 0.0).to_dict(),
            "fit_cost": fit_cost.to_dict(),
            "selection_cost": StageCost("selection", 0.0, 0.0, 0, 0.0).to_dict(),
            "clusters": clusters,
            "reported": {
                "estimate": combined.mean,
                "sample_variance": combined.sample_variance,
                "standard_error": combined.standard_error,
                "relative_standard_error": (
                    combined.standard_error / combined.mean if combined.mean > 0 else None
                ),
                "normalization_z": _normalization_z(normalized),
                "accuracy_z": qualify.accuracy_z_diagnostic,
                "total_wall_seconds": total_wall,
                "total_cpu_seconds": total_cpu,
                "peak_memory_bytes": peak_memory,
                "total_proxy_work": total_work,
                "qualification": qualify.status,
            },
        })
    natural_record, ce_record = methods
    def wnv(record: dict[str, Any]) -> float:
        combined = _combined([Moments(**cluster["moments"]) for cluster in record["clusters"]])
        return work_normalized_variance(
            combined.sample_variance,
            record["reported"]["total_proxy_work"], combined.count,
        )
    denominator = wnv(ce_record)
    ratio = wnv(natural_record) / denominator if denominator > 0 else None
    source = source_manifest(ROOT, config=config)
    return {
        "schema": SCHEMA,
        "manifest": {
            "source": source,
            "config": config,
            "task_id": problem.task_id,
            "payoff_id": "terminal_left_conditional_cdf",
            "convention": "BLP-local-2N-left-endpoint-epsilon1",
            "steps": problem.steps,
            "level": 0,
            "training_rep": training_rep,
            "seed_ledger": ledger.to_dict(),
            "qualification_policy": config["qualification"],
            "timing_context": {
                "warmup": "not_performed_smoke",
                "power_mode": "not_recorded_smoke",
                "concurrent_timed_jobs": False,
            },
        },
        "reference": {
            **reference_seed, "independent": True,
            "inferential_unit": "iid_path", "points_per_unit": 1,
            "raw_sample_count": reference.count,
            "moments": reference.to_dict(),
            "cost": reference_cost.to_dict(),
            "reported": {"estimate": reference.mean, "standard_error": reference.standard_error},
        },
        "methods": methods,
        "comparisons": [{
            "candidate": "weighted_conditional_ce",
            "comparator": "conditional_natural",
            "reported_proxy_ratio": ratio,
        }],
        "claim_context": {
            "purpose": "R0 two-method schema and timing smoke only",
            "ce_target_reached": fit.target_reached,
            "ce_target_ess_history": fit.target_ess_history,
            "ce_fitting_ess_history": fit.fitting_ess_history,
            "ce_elite_threshold_history": fit.elite_threshold_history,
            "source_dirty_at_start": source["source_dirty"],
            "fresh_confirmation": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/post_audit/r0_smoke_v1.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    payload = build_smoke(config)
    audit = audit_measurement(payload)
    if audit["integrity"] != "pass" or audit["semantic_validity"] != "pass":
        raise RuntimeError(f"R0 smoke audit failed: {audit['reasons']}")
    output = ROOT / config["output_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"R0 smoke output already exists: {output}")
    temporary = output.with_suffix(output.suffix + ".pending")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(json.dumps({"output": str(output), "audit": audit}, allow_nan=False))


if __name__ == "__main__":
    main()
