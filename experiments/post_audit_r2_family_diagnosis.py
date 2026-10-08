"""S0/S1: fresh parents, fixed-family contrast, independent allocation, capped final.

One parent per cell is a throughput/allocation pilot, NOT full-fit reliability
or confirmation. Historical reference is disclosed, never silently upgraded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
import zipfile
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any, Literal, cast

import psutil
import torch
import yaml

from experiments.post_audit_r1_diagnostics import _problem
from experiments.post_audit_r15_reference_crosscheck import _log_potential
from experiments.post_audit_r15_reference_design import temperatures
from src.path_integral.baselines.weighted_conditional_ce import proposal_parameters
from src.path_integral.conditional_second_moment import allocate_precision
from src.path_integral.finite_rank_gaussian_transport import DefensiveFiniteRankGaussianMixture
from src.path_integral.proposal_family_diagnostics import refit_identity_mixture
from src.path_integral.r1_bottleneck_diagnostics import (
    fit_projected_mean_shift,
    summarize_log_contributions,
)
from src.path_integral.research_result_contract import (
    canonical_digest,
    source_manifest,
    source_tree_digest,
)
from src.path_integral.seed_ledger import SeedKey, SeedLedger
from src.path_integral.weighted_bank_mixture import fit_weighted_bank_mixture
from src.path_integral.weighted_tempered_smc import (
    WeightedSMCConfig,
    estimate_weighted_tempered_normalizer,
)

ROOT = Path(__file__).resolve().parents[1]


def freeze_source(output: Path, config: dict[str, Any], *, extra_snapshot_paths: tuple[str, ...] = ()) -> dict[str, Any]:
    """Archive runtime source/config plus dependency declarations, before any run."""
    snapshot = output.with_suffix(".source.zip")
    for name in extra_snapshot_paths:
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
            raise ValueError("extra snapshot paths must be existing workspace files")
    paths = subprocess.check_output((
        "git", "ls-files", "--cached", "--others", "--exclude-standard", "-z",
        "--", "src", "experiments", "configs", "tests", "pyproject.toml", "requirements.txt",
        "requirements-dev.txt", "docs/reviews/MODEL_STRUCTURAL_REVIEW_AND_IMPROVEMENT_PLAN_2026-10-07_KO.md",
        "main.py", "train_driftnet.py", *extra_snapshot_paths,
    ), cwd=ROOT).split(b"\0")
    manifest = source_manifest(ROOT, config=config)
    hashes = {}
    with zipfile.ZipFile(snapshot, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for raw in sorted(set(x for x in paths if x)):
            name = raw.decode("utf-8")
            path = ROOT / name
            if path.is_file():
                content = path.read_bytes()
                archive.writestr(name, content)
                hashes[name] = hashlib.sha256(content).hexdigest()
        archive.writestr("RUN_CONFIG.json", json.dumps(config, sort_keys=True, allow_nan=False))
    manifest.update({"snapshot_path": snapshot.relative_to(ROOT).as_posix(),
                     "snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                     "snapshot_file_hashes": hashes,
                     "snapshot_scope": "runtime source/config, tests, improvement plan and dependency declarations; results excluded"})
    return manifest


def run(config: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(int(config["torch_threads"]))
    ledger = SeedLedger()
    protocol = "post-audit-r2-family-" + canonical_digest(config)[:16]
    started = time.perf_counter()
    evaluations = 0
    records: list[dict[str, Any]] = []
    reference_bytes = (ROOT / config["reference_path"]).read_bytes()
    reference = json.loads(reference_bytes)
    smc_spec, fit_spec, allocation = config["smc"], config["fit"], config["allocation"]
    budget = config["budget"]
    process = psutil.Process()
    peak_rss = process.memory_info().rss

    def check_budget(needed: int) -> None:
        if evaluations + needed > budget["max_potential_evaluations"]:
            raise TimeoutError("unresolved_potential_budget")
        if time.perf_counter() - started >= budget["max_wall_seconds"]:
            raise TimeoutError("unresolved_wall_budget")

    for cell in config["cells"]:
        problem = _problem(config["model"], cell, int(smc_spec["steps"]))
        ref = next(x["new_reference"] for x in reference["cells"]
                   if x["cell"]["id"] == cell["id"])
        for parent_rep in range(int(config["parent_training_replicates"])):
            row_start = time.perf_counter()
            row_evaluations_start = evaluations
            row: dict[str, Any] = {"cell": cell, "task_id": problem.task_id,
                                   "task_digest": canonical_digest({"model": config["model"],
                                                                    "cell": cell, "steps": smc_spec["steps"]}),
                                   "parent_training_rep": parent_rep, "refit_rep": 0,
                                   "selection_rep": None, "reference_rep": "historical-r15",
                                   "candidates": [], "failure_reasons": []}
            records.append(row)

            def seed(role: str, stream: str, level: int = 0,
                     task: str = problem.task_id, replicate: int = parent_rep) -> int:
                return ledger.allocate(SeedKey(protocol, role, "development",
                                               task, level, replicate, stream))

            try:
                samples, weights, diagnostics = [], [], []
                offline_start = time.perf_counter()
                for island in range(int(smc_spec["islands"])):
                    # Conservative upper bound; terminal stage has no mutation.
                    upper = int(smc_spec["particles"]) * (
                        1 + (int(smc_spec["levels"]) - 1) * int(smc_spec["mutation_steps"]))
                    check_budget(upper)
                    bank = estimate_weighted_tempered_normalizer(
                        partial(_log_potential, problem), dimension=problem.local_dimension,
                        config=WeightedSMCConfig(
                            particles=int(smc_spec["particles"]),
                            temperatures=temperatures(int(smc_spec["levels"]), int(smc_spec["bridge_power"])),
                            mutation_steps=int(smc_spec["mutation_steps"]),
                            pcn_scale=float(smc_spec["pcn_scale"]), replicates=1,
                            seed=seed("parent-training", f"island-{island}"),
                            resample_every=int(smc_spec["resample_every"]),
                            resampling_scheme=cast(Literal["multinomial", "stratified"], smc_spec["resampling_scheme"]),
                            retain_final_particles=True,
                        ))
                    evaluations += bank.potential_evaluations
                    if bank.final_particles is None or bank.final_weights is None:
                        raise RuntimeError("missing SMC bank")
                    samples.append(bank.final_particles)
                    weights.append(bank.final_weights / smc_spec["islands"])
                    diagnostics.append(bank.replicate_diagnostics[0])
                parent = fit_weighted_bank_mixture(torch.cat(samples), torch.cat(weights),
                                                  **config["mixture"]).proposal
                offline_wall = time.perf_counter() - offline_start
                row.update({"parent_proposal_digest": canonical_digest(proposal_parameters(parent)),
                            "bank_diagnostics": diagnostics, "offline_wall_seconds": offline_wall})
                check_budget(int(fit_spec["bank_count"]))
                fit_start = time.perf_counter()
                draw = parent.sample(int(fit_spec["bank_count"]),
                                     path_seed=seed("refit-bank", "path"),
                                     label_seed=seed("refit-bank", "label"))
                log_g = _log_potential(problem, draw.samples)
                evaluations += draw.samples.shape[0]
                shared_bank_wall = time.perf_counter() - fit_start
                fit_start = time.perf_counter()
                preserved, fit_diagnostic = refit_identity_mixture(
                    parent, draw.samples, log_g, draw.log_p_over_q,
                    steps=int(fit_spec["steps"]), learning_rate=float(fit_spec["learning_rate"]),
                    maximum_norm=float(fit_spec["maximum_norm"]))
                preserved_wall = time.perf_counter() - fit_start
                fit_start = time.perf_counter()
                compressed, compressed_loss = fit_projected_mean_shift(
                    draw.samples, log_g, draw.log_p_over_q,
                    torch.eye(problem.local_dimension, dtype=torch.float64), objective="kl",
                    defensive_mass=parent.defensive_mass, steps=int(fit_spec["steps"]),
                    learning_rate=float(fit_spec["learning_rate"]),
                    maximum_norm=float(fit_spec["maximum_norm"]))
                compressed_wall = time.perf_counter() - fit_start
                row["refit_bank"] = {"count": draw.samples.shape[0], "proposal_digest": row["parent_proposal_digest"],
                                      "role": "fit_only", "same_family_diagnostic": fit_diagnostic,
                                      "single_shift_empirical_kl_loss": compressed_loss,
                                      "target_weight_ess": float(1 / torch.softmax(
                                          log_g + draw.log_p_over_q, dim=0).square().sum())}
                candidates = [("parent-as-is", parent, 0.0),
                              ("same-family-refit", preserved, shared_bank_wall + preserved_wall),
                              ("single-shift", compressed, shared_bank_wall + compressed_wall)]
                # Freeze all candidates before *any* pilot or final observations.
                for method, proposal, fit_wall in candidates:
                    row["candidates"].append({
                        "method": method, "proposal_parameters": proposal_parameters(proposal),
                        "proposal_digest": canonical_digest(proposal_parameters(proposal)),
                        "offline_wall_seconds": offline_wall, "fit_wall_seconds": fit_wall,
                        "failure_reasons": [], "qualified": False,
                    })
                frozen: list[tuple[DefensiveFiniteRankGaussianMixture, dict[str, Any]]] = []
                for (_, proposal, _), item in zip(candidates, row["candidates"], strict=True):
                    pilot_start = time.perf_counter()
                    logs = []
                    for batch in range(math.ceil(allocation["pilot_count"] / allocation["batch_size"])):
                        n = min(allocation["batch_size"], allocation["pilot_count"] - batch * allocation["batch_size"])
                        check_budget(n)
                        pilot = proposal.sample(n, path_seed=seed("allocation-pilot", item["method"] + "-path", batch),
                                                label_seed=seed("allocation-pilot", item["method"] + "-label", batch))
                        logs.append(_log_potential(problem, pilot.samples) + pilot.log_p_over_q)
                        evaluations += n
                        peak_rss = max(peak_rss, process.memory_info().rss)
                    values = torch.cat(logs)
                    item["pilot"] = asdict(summarize_log_contributions(values))
                    ordered = torch.sort(values, descending=True).values
                    top_count = max(1, math.ceil(.01 * ordered.numel()))
                    item["pilot_tail_diagnostic"] = {
                        "top_one_percent_contribution_share": float(torch.exp(
                            torch.logsumexp(ordered[:top_count], 0) - torch.logsumexp(ordered, 0))),
                        "top_one_percent_second_moment_share": float(torch.exp(
                            torch.logsumexp(2 * ordered[:top_count], 0) - torch.logsumexp(2 * ordered, 0))),
                        "role": "development_only_not_mode_completeness_certificate"}
                    item["allocation"] = asdict(allocate_precision(
                        values, target_relative_se=allocation["target_relative_se"],
                        safety_factor=allocation["safety_factor"], batch_size=allocation["batch_size"],
                        maximum_count=allocation["maximum_count_per_method"]))
                    item["selection_wall_seconds"] = time.perf_counter() - pilot_start
                    frozen.append((proposal, item))
                row["allocation_frozen_before_final"] = True
                for proposal, item in frozen:
                    if item["allocation"]["status"] != "allocated":
                        item["failure_reasons"].append(item["allocation"]["status"])
                        continue
                    final_start = time.perf_counter()
                    logs, batch_rows = [], []
                    count = item["allocation"]["planned_count"]
                    try:
                        for batch in range(count // allocation["batch_size"]):
                            n = int(allocation["batch_size"])
                            check_budget(n)
                            final = proposal.sample(n, path_seed=seed("final", item["method"] + "-path", batch),
                                                    label_seed=seed("final", item["method"] + "-label", batch))
                            log_values = _log_potential(problem, final.samples) + final.log_p_over_q
                            logs.append(log_values)
                            batch_rows.append({"final_batch": batch, **asdict(summarize_log_contributions(log_values))})
                            evaluations += n
                            peak_rss = max(peak_rss, process.memory_info().rss)
                    except TimeoutError as error:
                        item["failure_reasons"].append(str(error))
                    item["inference_wall_seconds"] = time.perf_counter() - final_start
                    item["final_batches"] = batch_rows
                    if logs:
                        summary = summarize_log_contributions(torch.cat(logs))
                        item["final"] = asdict(summary)
                        item["final_completed"] = summary.count == count
                        if item["final_completed"] and summary.log_mean is not None and summary.relative_se is not None:
                            mean = math.exp(summary.log_mean)
                            se = mean * summary.relative_se
                            q = config["qualification"]
                            uncertainty_ok = ref["standard_error"] <= q["reference_se_fraction_of_method_se"] * se
                            upper = abs(mean - ref["mean"]) + q["confidence_z"] * math.hypot(se, ref["standard_error"])
                            equivalence_ok = upper <= q["relative_equivalence_margin"] * ref["mean"]
                            precision_ok = summary.relative_se <= allocation["target_relative_se"]
                            item["accuracy"] = {"mean": mean, "standard_error": se,
                                                "reference_mean": ref["mean"], "reference_standard_error": ref["standard_error"],
                                                "reference_precision_ok": uncertainty_ok,
                                                "method_precision_ok": precision_ok, "equivalence_ok": equivalence_ok,
                                                "equivalence_upper_difference": upper}
                            for ok, reason in [(uncertainty_ok, "unresolved_reference_precision"),
                                               (precision_ok, "insufficient_final_precision"),
                                               (equivalence_ok, "equivalence_not_established")]:
                                if not ok:
                                    item["failure_reasons"].append(reason)
                            item["qualified"] = not item["failure_reasons"]
                    else:
                        item["final_completed"] = False
                    if not item["final_completed"]:
                        item["failure_reasons"].append("incomplete_final")
                row["status"] = "completed_development_pilot"
            except (TimeoutError, ValueError, FloatingPointError, RuntimeError) as error:
                row["status"] = "unresolved"
                row["failure_reasons"].append(f"{type(error).__name__}: {error}")
            for item in row["candidates"]:
                item["total_deployment_wall_seconds"] = sum(
                    item.get(k, 0) for k in ("offline_wall_seconds", "fit_wall_seconds",
                                             "selection_wall_seconds", "inference_wall_seconds"))
            row["precision_cost_comparison_authorized"] = sum(
                bool(x["qualified"]) for x in row["candidates"]) >= 2
            row["physical_wall_seconds_including_failure"] = time.perf_counter() - row_start
            row["potential_evaluations"] = evaluations - row_evaluations_start
    if source_tree_digest(ROOT) != source["source_tree_digest"]:
        raise RuntimeError("runtime source changed during experiment; result not authorized")
    return {"schema": "npi.post-audit.r2-family-diagnosis.v1", "source": source, "config": config,
            "role": "whole_fit_engineering_development_not_confirmation",
            "reference_sha256": hashlib.sha256(reference_bytes).hexdigest(),
            "seed_ledger": ledger.to_dict(), "records": records,
            "potential_evaluations": evaluations, "physical_wall_seconds": time.perf_counter() - started,
            "sampled_peak_process_rss_bytes": peak_rss,
            "limitations": ["One parent per cell cannot establish whole-training stability.",
                            "Pilot safety factor is not a tail-risk confidence guarantee.",
                            "Historical reference may fail the SE-ratio requirement.",
                            "Only identity-covariance mixture refitting is implemented.",
                            "Descriptive notebook CPU timing; no publication cost claim."]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/post_audit/r2_family_diagnosis_pilot_v1.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = ROOT / config["output_path"]
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    source = freeze_source(output, config)
    payload = run(config, source)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    print(json.dumps({"output": str(output), "potential_evaluations": payload["potential_evaluations"],
                      "wall_seconds": payload["physical_wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
