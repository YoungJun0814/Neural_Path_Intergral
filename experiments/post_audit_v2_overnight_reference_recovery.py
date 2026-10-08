"""Sealed, resumable overnight development diagnostics; production stays locked."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import subprocess
import time
from pathlib import Path
from statistics import NormalDist
from typing import Any

import psutil
import torch
import yaml

from experiments.post_audit_r1_diagnostics import _problem
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_v2_independent_reference import inputs
from src.path_integral.baselines.weighted_conditional_ce import proposal_parameters
from src.path_integral.cached_volterra_terminal import CachedShiftDensity, CachedVolterraTerminal
from src.path_integral.conditional_second_moment import log_risk_potential
from src.path_integral.elliptical_slice_kernel import elliptical_slice_transition
from src.path_integral.gaussian_bump_oracle import GaussianBumpOracle
from src.path_integral.gaussian_mixture_marginal import ShiftMixtureMarginal
from src.path_integral.recovery_checkpoint import RecoveryJournal
from src.path_integral.reference_protocol import canonical_sha256
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.research_result_contract import source_tree_digest
from src.path_integral.structural_v2_inventory import audit_snapshot
from src.path_integral.structural_v2_reference import log_moments, relative_equivalence, sensitivity
from src.path_integral.structural_v2_terminal_geometry import terminal_geometry
from src.path_integral.volterra_conditional_payoffs import evaluate_rbergomi_conditional_terminal
from src.path_integral.volterra_excursion_guide import build_volterra_excursion_guide
from src.path_integral.weighted_bank_mixture import proposal_from_parameters
from src.path_integral.weighted_tempered_smc import (
    WeightedSMCConfig,
    estimate_weighted_tempered_normalizer,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = "docs/plans/OVERNIGHT_REFERENCE_RECOVERY_AND_RESEARCH_DECISION_PLAN_2026-10-08_KO.md"
TOYS = {
    "centered": GaussianBumpOracle((1.,), ((0., 0.),), (1.,)),
    "shifted": GaussianBumpOracle((1.,), ((6., 0.),), (.5,)),
    "bimodal": GaussianBumpOracle((.000001, .999999), ((0., 0.), (6., 0.)), (.25, .5)),
}


def seed(identity: str, stream: str) -> int:
    return int(hashlib.sha256(("overnight-reference-recovery-v1/" + identity + "/" + stream).encode()).hexdigest()[:15], 16)


def validate_config(config: dict[str, Any]) -> None:
    for name in ("torch_threads", "unit_count", "batch_size", "replications", "whole_replicates"):
        if isinstance(config[name], bool) or not isinstance(config[name], int):
            raise ValueError("integer experiment counts required")
    if (config["schema"] != "npi.overnight-reference-recovery.config.v1"
            or config["torch_threads"] != 1 or config["unit_count"] != 262144
            or config["batch_size"] != 8192 or config["replications"] != 2
            or config["whole_replicates"] != 8 or config["guide"]["defensive_mass"] != .1):
        raise ValueError("sealed development grid changed")
    if config["smc"] != {"levels": 64, "bridge_power": 2, "mutation_steps": 1, "pcn_scale": .35, "resample_every": 4}:
        raise ValueError("diagnostic schedule changed")


def resource_check(config: dict[str, Any], directory: Path) -> None:
    spec = config["resources"]
    if psutil.Process().memory_info().rss > min(spec["max_rss_bytes"], psutil.virtual_memory().total//4):
        raise MemoryError("recovery RSS cap")
    if shutil.disk_usage(ROOT).free < spec["minimum_free_disk_bytes"]:
        raise OSError("recovery free disk cap")
    if sum(p.stat().st_size for p in directory.rglob("*") if p.is_file()) > spec["max_artifact_bytes"]:
        raise OSError("recovery artifact cap")


def smc_unit(potential: Any, *, dimension: int, identity: str, islands: int,
             config: dict[str, Any], toy: bool, evaluator: Any = None) -> dict[str, Any]:
    spec = config["smc"]
    logs, records, actual = [], [], 0
    for island in range(islands):
        snapshots: list[dict[str, Any]] = []
        def observer(event: str, replicate: int, beta: float, z: torch.Tensor, w: torch.Tensor,
                     saved: list[Any] = snapshots) -> None:
            # Fixed Gaussian-coordinate proxy, not an estimated Volterra spike region.
            region = (z[:, 0] >= 2.5) & (z[:, 0] <= 7.5) & (z[:, 1].abs() <= 2.)
            saved.append({"event": event, "beta": beta, "region_hits": int(region.sum()),
                              "region_weight": float(w[region].sum()),
                              "mean_first_coordinate": float(w @ z[:, 0]),
                              "mean_squared_norm": float(w @ z.square().sum(1))})
        calls = 0
        def counted(z: torch.Tensor) -> torch.Tensor:
            nonlocal calls
            calls += len(z)
            return potential(z)
        cfg = WeightedSMCConfig(512//islands,
            tuple((i/(spec["levels"]-1))**spec["bridge_power"] for i in range(spec["levels"])),
            spec["mutation_steps"], spec["pcn_scale"], 1, seed(identity, f"island-{island}"),
            resample_every=spec["resample_every"], resampling_scheme="stratified", retain_final_particles=True)
        output = estimate_weighted_tempered_normalizer(counted, dimension=dimension, config=cfg, observer=observer)
        if calls != output.potential_evaluations or calls != (512//islands)*63:
            raise ValueError("fixed SMC call ledger mismatch")
        logs.append(float(output.log_replicate_estimates[0]))
        actual += calls
        record = {"log_normalizer": logs[-1], "stages": output.replicate_diagnostics[0],
                  "snapshots": snapshots, "initial_region_prior_mass":
                  (NormalDist().cdf(7.5)-NormalDist().cdf(2.5))*(2*NormalDist().cdf(2.)-1)}
        if evaluator is not None:
            assert output.final_particles is not None and output.final_weights is not None
            record["terminal_geometry"] = terminal_geometry(evaluator.problem, output.final_particles, output.final_weights)
        records.append(record)
    return {"log_estimate": float(torch.logsumexp(torch.tensor(logs, dtype=torch.float64), 0))-math.log(islands),
            "islands": records, "potential_calls": actual,
            "inference_unit": "aggregate_whole_run", "toy": toy,
            "seeds": [seed(identity, f"island-{i}") for i in range(islands)]}


def compare(logs_a: torch.Tensor, logs_b: torch.Tensor, wall_a: float, wall_b: float) -> dict[str, Any]:
    common = float(torch.max(torch.cat((logs_a, logs_b))))
    a, b = torch.exp(logs_a-common), torch.exp(logs_b-common)
    ratio = float(b.var(unbiased=True))*wall_b/len(b) / (float(a.var(unbiased=True))*wall_a/len(a))
    # Fixed independent equal-size block sensitivity; no unseen-tail certificate.
    block_a, block_b = a.reshape(32, -1), b.reshape(32, -1)
    generator = torch.Generator().manual_seed(314159)
    ia = torch.randint(32, (2000, 32), generator=generator)
    ib = torch.randint(32, (2000, 32), generator=generator)
    def variances(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        sums = x.sum(1)[index].sum(1)
        squares = x.square().sum(1)[index].sum(1)
        count = x.numel()
        return (squares-sums.square()/count)/(count-1)
    ratios = variances(block_b, ib)/variances(block_a, ia)*(wall_b/len(b))/(wall_a/len(a))
    interval = torch.quantile(ratios, torch.tensor((.00625, .99375), dtype=torch.float64)).tolist()
    return {"marginal_over_full_variance_cost": ratio,
            "simultaneous_block_bootstrap_interval": interval,
            "family_comparisons": 4, "alpha": .05, "bootstrap_replicates": 2000,
            "development_point_gain": ratio <= .8,
            "development_uncertainty_gain": interval[1] <= .8,
            "mean_equivalence": relative_equivalence(log_moments(logs_a), log_moments(logs_b), comparisons=4),
            "scope": "development_proxy_not_fixed_precision_or_tail_certificate"}


def run(config: dict[str, Any], *, stop_after_units: int | None = None) -> dict[str, Any]:
    validate_config(config)
    torch.set_num_threads(1)
    directory = ROOT/config["run_directory"]
    if not directory.resolve().is_relative_to((ROOT/"results/post_audit").resolve()):
        raise ValueError("run directory outside research result root")
    directory.mkdir(parents=True, exist_ok=True)
    gate = json.loads((ROOT/"results/post_audit/reference_redesign_allocation_v1.json").read_text())
    if gate["production_config"] is not None or gate["p2_authorized"]:
        raise ValueError("unexpected historical gate; review required")
    metadata, rows = inputs(config)
    source_path = directory/"source.json"
    if source_path.exists():
        source = json.loads(source_path.read_text())
        audit_snapshot(ROOT, source, config)
        if source["source_tree_digest"] != source_tree_digest(ROOT):
            raise ValueError("runtime source changed since sealed run")
    else:
        source = freeze_source(directory/"run.json", config, extra_snapshot_paths=(PLAN,))
        write_json_atomic_nonoverwriting(source_path, source)
    resource_check(config, directory)
    jobs = ["N0-environment", "N4-allocation"]
    for name in TOYS:
        for power in (1, 2):
            for islands in (1, 4):
                for rep in range(8):
                    jobs.append(f"N2-{name}-{power}-{islands}-{rep}")
    for row in rows:
        if row["parent_training_rep"] == 0:
            jobs.append("N3-guide-"+row["cell"]["id"])
            jobs.extend(f"N3-{row['cell']['id']}-{n}" for n in (1, 2, 4, 16, 128, 1024, 8192))
            jobs.extend("N3-throughput-"+row["cell"]["id"]+"-"+m for m in ("full", "marginal"))
            jobs.append("N3-ellipse-profile-"+row["cell"]["id"])
            for kind in ("mu", "risk"):
                for replication in range(2):
                    for method in ("full", "marginal"):
                        for batch in range(32):
                            jobs.append(f"N4-{row['cell']['id']}-{kind}-{replication}-{method}-{batch}")
            for kind in ("mu", "risk"):
                for islands in (1, 4):
                    for rep in range(8):
                        jobs.append(f"N5-{row['cell']['id']}-{kind}-{islands}-{rep}")
    manifest = {"schema": "npi.recovery.manifest.v1", "config": config,
                "source": source["snapshot_sha256"], "source_tree": source_tree_digest(ROOT),
                "limits": config["limits"], "expected_base_units": jobs,
                "q_bindings": [{"cell": r["cell"], "parent": r["parent_training_rep"], "q": r["q_digest"]} for r in rows],
                "input_artifact_sha256": hashlib.sha256((ROOT/config["proposal_artifact_path"]).read_bytes()).hexdigest(),
                "model": metadata["config"]["model"], "steps": metadata["config"]["smc"]["steps"],
                "seed_rule": "sha256(namespace/unit-id/stream), independent method streams",
                "conditional_extension": "all-rep1-4-risk-only-if-both-cells-two-replications-gain",
                "precision_qualification": False, "p2_authorized": False}
    journal = RecoveryJournal(directory, manifest)
    counter = 0
    def unit(identity: str, phase: str, work: int, wall: float, operation: Any) -> dict[str, Any]:
        nonlocal counter
        resource_check(config, directory)
        result = journal.unit(identity, phase=phase, work=work, maximum_wall=wall, operation=operation)
        counter += 1
        if stop_after_units is not None and counter >= stop_after_units:
            raise InterruptedError("intentional complete-boundary interruption")
        return result
    environment = unit("N0-environment", "N0", 0, 10., lambda: {
        "started_unix": time.time(), "python": platform.python_version(), "torch": torch.__version__, "threads": torch.get_num_threads(),
        "cpu_count": psutil.cpu_count(), "ram": psutil.virtual_memory().total,
        "free_disk": shutil.disk_usage(ROOT).free,
        "head": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
        "branch": subprocess.check_output(("git", "branch", "--show-current"), cwd=ROOT, text=True).strip(),
        "dirty_status": subprocess.check_output(("git", "status", "--short"), cwd=ROOT, text=True),
        "historical_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                              (ROOT/"results/post_audit").glob("reference_redesign_*v1.*")},
        "p2_authorized": False})
    def deadline_check() -> None:
        if time.time()-environment["started_unix"] > 27000-2700:
            raise TimeoutError("session report reserve reached; do not start science job")
    toy_results = []
    for name, oracle in TOYS.items():
        for power in (1, 2):
            for islands in (1, 4):
                records = []
                for rep in range(8):
                    deadline_check()
                    identity = f"N2-{name}-{power}-{islands}-{rep}"
                    records.append(unit(identity, "N2", 512*63, 5.,
                        lambda o=oracle, p=power, ident=identity, k=islands:
                        smc_unit(lambda z: p*o.log_value(z), dimension=2, identity=ident,
                                 islands=k, config=config, toy=True)))
                values = torch.tensor([r["log_estimate"] for r in records], dtype=torch.float64)
                summary = log_moments(values)
                oracle_ratio = math.exp(summary["log_mean"]-oracle.log_endpoint(power))
                z_family = NormalDist().inv_cdf(1-.05/(2*12))
                toy_results.append({"toy": name, "power": power, "islands": islands,
                    "analytic_log_mean": oracle.log_endpoint(power), "summary": summary,
                    "relative_error": math.expm1(summary["log_mean"]-oracle.log_endpoint(power)),
                    "approximate_family_interval_relative_to_oracle":
                    [oracle_ratio*(1-z_family*summary["relative_se"]), oracle_ratio*(1+z_family*summary["relative_se"])],
                    "scope": "normal_approximation_eight_whole_runs_not_bias_or_coverage_certificate",
                    "sensitivity": sensitivity(values.tolist(), bootstrap_seed=seed(name+str(power)+str(islands), "bootstrap")),
                    "units": records})
                print(json.dumps({"phase": "N2", "toy": name, "power": power, "islands": islands,
                                  "rse": summary["relative_se"]}), flush=True)
    contexts = []
    profile = []
    for row in rows:
        if row["parent_training_rep"] != 0:
            continue
        problem = _problem(metadata["config"]["model"], row["cell"], metadata["config"]["smc"]["steps"])
        q = proposal_from_parameters(row["q_parameters"])
        evaluator = CachedVolterraTerminal.create(problem)
        guide_parameters = unit("N3-guide-"+row["cell"]["id"], "N3", problem.local_dimension, 10.,
                                lambda p=problem: proposal_parameters(build_volterra_excursion_guide(p, **config["guide"])))
        guide = proposal_from_parameters(guide_parameters)
        fixed_density = CachedShiftDensity.create(q)
        marginal = ShiftMixtureMarginal.from_full(guide, tuple(range(problem.local_dimension-2)))
        contexts.append((row, evaluator, q, guide, marginal))
        ellipse_identity = "N3-ellipse-profile-"+row["cell"]["id"]
        def ellipse_profile(ev: Any = evaluator, density: Any = fixed_density, fixed_q: Any = q,
                            identity: str = ellipse_identity) -> dict[str, Any]:
            calls, active = 0, []
            def potential(z: torch.Tensor) -> torch.Tensor:
                nonlocal calls
                calls += len(z)
                active.append(len(z))
                return log_risk_potential(ev.log_probability(z), density.log_q_over_p(z), defensive_mass=fixed_q.defensive_mass)
            z = torch.randn((128, ev.problem.local_dimension), dtype=torch.float64,
                            generator=torch.Generator().manual_seed(seed(identity, "initial")))
            moved = elliptical_slice_transition(z, potential(z), beta=.5, log_potential=potential,
                generator=torch.Generator().manual_seed(seed(identity, "ellipse")), maximum_attempts=1024)
            return {"profile": "one-transition-128-particles-beta-half-not-stationarity-or-performance",
                    "active_batch_sizes": active, "maximum_attempts": moved.maximum_attempts,
                    "actual_work": calls*2, "fft_paths": calls, "cdf_calls": calls,
                    "density_component_evaluations": calls*len(fixed_q.components)}
        profile.append(unit(ellipse_identity, "N3", 2*128*1025, 10., ellipse_profile))
        for count in (1, 2, 4, 16, 128, 1024, 8192):
            identity = f"N3-{row['cell']['id']}-{count}"
            def measure(n: int = count, ev: Any = evaluator, ident: str = identity,
                        original_q: Any = q, density: Any = fixed_density) -> dict[str, Any]:
                z = torch.randn((n, ev.problem.local_dimension), dtype=torch.float64,
                                generator=torch.Generator().manual_seed(seed(ident, "profile")))
                times: dict[str, list[float]] = {"original": [], "cached": [], "density_original": [], "density_cached": []}
                error = 0.
                for repeat in range(4):
                    for method in (("original", "cached") if repeat % 2 == 0 else ("cached", "original")):
                        start = time.perf_counter()
                        value = (evaluate_rbergomi_conditional_terminal(ev.problem, z).payoffs.log_left_probability
                                 if method == "original" else ev.log_probability(z))
                        times[method].append(time.perf_counter()-start)
                        if method == "original":
                            reference = value
                        else:
                            cached = value
                    error = max(error, float((reference-cached).abs().max()))
                    torch.testing.assert_close(reference, cached, rtol=2e-13, atol=2e-12)
                    begin = time.perf_counter()
                    raw_density = original_q.log_q_over_p(z)
                    times["density_original"].append(time.perf_counter()-begin)
                    begin = time.perf_counter()
                    prepared_density = density.log_q_over_p(z)
                    times["density_cached"].append(time.perf_counter()-begin)
                    torch.testing.assert_close(raw_density, prepared_density, rtol=2e-13, atol=2e-12)
                return {"count": n, "times_including_warmup": times, "maximum_absolute_log_error": error,
                        "fft_paths": n*8, "cdf_calls": n*8,
                        "density_component_evaluations": n*8*len(original_q.components),
                        "cached_over_original_median": float(torch.tensor(times["cached"][1:]).median()
                                                             /torch.tensor(times["original"][1:]).median())}
            profile.append(unit(identity, "N3", count*16, 10., measure))
        # Throughput-only guide draws, disjoint seeds. No pilot means used for selection.
        for method in ("full", "marginal"):
            ident = "N3-throughput-"+row["cell"]["id"]+"-"+method
            def throughput(ev: Any = evaluator, reference: Any = guide, outer_ref: Any = marginal,
                           chosen: str = method, identity: str = ident, density: Any = fixed_density,
                           fixed_q: Any = q) -> dict[str, Any]:
                start = time.perf_counter()
                n = config["batch_size"]
                if chosen == "full":
                    draw = reference.sample(n, path_seed=seed(identity, "path"), label_seed=seed(identity, "label"))
                    logs = 2*ev.log_probability(draw.samples)-density.log_q_over_p(draw.samples)-draw.log_q_over_p
                else:
                    a = outer_ref.sample(n, path_seed=seed(identity, "path"), label_seed=seed(identity, "label"))
                    pair = torch.randn((n, 1, 2), dtype=torch.float64, generator=torch.Generator().manual_seed(seed(identity, "pair")))
                    logs = ev.last_pair(a, fixed_q).log_inner_risk(pair)[:, 0]-outer_ref.log_q_over_p(a)
                if not torch.isfinite(logs).all():
                    raise FloatingPointError("throughput pilot nonfinite")
                return {"count": n, "method": chosen, "wall_seconds": time.perf_counter()-start,
                        "fft_paths": n, "cdf_calls": n,
                        "density_component_evaluations": n*(len(reference.components)+
                            len(fixed_q.components)*(2 if chosen == "marginal" else 1)),
                        "mean_not_used": True, "role": "throughput-only-pilot"}
            profile.append(unit(ident, "N3", 2*config["batch_size"], 10., throughput))
    comparisons = []
    is_records = []
    pilot_seconds = sum(r["wall_seconds"] for r in profile if r.get("role") == "throughput-only-pilot")
    forecast = pilot_seconds*32*2*2  # both estimands and replications; conservative repeated q work
    allocation = unit("N4-allocation", "N4", 0, 1., lambda: {
        "unit_count_per_job": config["unit_count"], "base_grid_work": 8388608,
        "throughput_wall_forecast_seconds": forecast, "safety_multiplier": 3.,
        "resource_pass": forecast*3 < config["limits"]["N4"]["wall"],
        "pilot_stats_used": False, "fixed_counts": True})
    if not allocation["resource_pass"]:
        raise TimeoutError("throughput allocation unresolved; no post-result count adjustment")
    def is_job(row: Any, ev: Any, q: Any, guide: Any, marginal: Any,
               kind: str, replication: int, prefix: str) -> dict[str, Any]:
        outputs: dict[str, list[dict[str, Any]]] = {"full": [], "marginal": []}
        q_density = CachedShiftDensity.create(q)
        for batch in range(32):
            deadline_check()
            for method in (("full", "marginal") if batch % 2 == 0 else ("marginal", "full")):
                identity = f"{prefix}-{row['cell']['id']}-{kind}-{replication}-{method}-{batch}"
                def operation(m: str = method, ident: str = identity) -> dict[str, Any]:
                    n = config["batch_size"]
                    start = time.perf_counter()
                    if m == "full":
                        draw = guide.sample(n, path_seed=seed(ident, "path"), label_seed=seed(ident, "label"))
                        z = draw.samples
                        logg = ev.log_probability(z)
                        values = (2*logg - q_density.log_q_over_p(z) if kind == "risk" else logg) - draw.log_q_over_p
                    else:
                        a = marginal.sample(n, path_seed=seed(ident, "path"), label_seed=seed(ident, "label"))
                        cache = ev.last_pair(a, q)
                        outer = marginal.log_q_over_p(a)
                        pair = torch.randn((n, 1, 2), dtype=torch.float64, generator=torch.Generator().manual_seed(seed(ident, "pair")))
                        values = cache.log_inner_risk(pair)[:, 0]-outer if kind == "risk" else cache.log_mu_mean()-outer
                        z = torch.cat((a, pair[:, 0]), 1)
                    summary = log_moments(values)
                    scaled = torch.exp(values-values.max())
                    # Predeclared coordinate norm bins; no path-dependent clustering.
                    bins = torch.bucketize(z[:, :ev.problem.local_dimension//2].square().sum(1),
                                          torch.tensor((16., 32., 64., 128.), dtype=torch.float64))
                    mass = torch.zeros(5, dtype=torch.float64).scatter_add_(0, bins, scaled)
                    return {"logs": values.tolist(), "summary": summary,
                            "wall_seconds": time.perf_counter()-start,
                            "geometry_bin_log_sums": [float(v.log()+values.max()) if v > 0 else None for v in mass],
                            "seeds": {s: seed(ident, s) for s in ("path", "label", "pair")},
                            "density_component_evaluations": n*(len(guide.components)+
                                (len(q.components) if m == "marginal" else 0)+
                                (len(q.components) if kind == "risk" else 0)),
                            "fft_paths": n, "cdf_calls": n,
                            "q_digest": row["q_digest"] if kind == "risk" else None}
                outputs[method].append(unit(identity, "N4", 2*config["batch_size"], 3., operation))
        all_logs = {m: torch.tensor([v for r in records for v in r["logs"]], dtype=torch.float64) for m, records in outputs.items()}
        walls = {m: sum(r["wall_seconds"] for r in records) for m, records in outputs.items()}
        record = {"cell": row["cell"], "parent": row["parent_training_rep"], "kind": kind,
                  "replication": replication, "prefix": prefix, "q_digest": row["q_digest"] if kind == "risk" else None,
                  "methods": {m: {"summary": log_moments(logs), "wall_seconds": walls[m],
                                   "blocks": [r["summary"] for r in outputs[m]],
                                   "sensitivity": sensitivity([r["summary"]["log_mean"] for r in outputs[m]], bootstrap_seed=seed(prefix+row['cell']['id']+kind+str(replication)+m, "bootstrap"))}
                              for m, logs in all_logs.items()},
                  "comparison": compare(all_logs["full"], all_logs["marginal"], walls["full"], walls["marginal"])}
        is_records.append(record)
        print(json.dumps({"phase": "N4", "cell": row["cell"]["id"], "kind": kind,
                          "replication": replication, "comparison": record["comparison"]}), flush=True)
        return record
    for row, ev, q, guide, marginal in contexts:
        for kind in ("mu", "risk"):
            for replication in range(2):
                comparisons.append(is_job(row, ev, q, guide, marginal, kind, replication, "N4"))
    extend = all(r["comparison"]["development_point_gain"] for r in comparisons if r["kind"] == "risk")
    if extend:
        for row in rows:
            if row["parent_training_rep"] == 0:
                continue
            context = next(c for c in contexts if c[0]["cell"]["id"] == row["cell"]["id"])
            _, ev, _, guide, marginal = context
            is_job(row, ev, proposal_from_parameters(row["q_parameters"]), guide, marginal, "risk", 0,
                   "N4extension"+str(row["parent_training_rep"]))
    coverage = []
    for row, ev, q, _, _ in contexts:
        q_density = CachedShiftDensity.create(q)
        for kind in ("mu", "risk"):
            for islands in (1, 4):
                units = []
                for rep in range(8):
                    deadline_check()
                    identity = f"N5-{row['cell']['id']}-{kind}-{islands}-{rep}"
                    def operation(evaluator: Any = ev, fixed_q: Any = q, k: str = kind, density: Any = q_density,
                                  ident: str = identity, allocation: int = islands) -> dict[str, Any]:
                        def potential(z: torch.Tensor) -> torch.Tensor:
                            logg = evaluator.log_probability(z)
                            return log_risk_potential(logg, density.log_q_over_p(z), defensive_mass=fixed_q.defensive_mass) if k == "risk" else logg
                        result = smc_unit(potential, dimension=evaluator.problem.local_dimension,
                                          identity=ident, islands=allocation, config=config, toy=False, evaluator=evaluator)
                        if k == "risk":
                            result["log_estimate"] -= math.log(fixed_q.defensive_mass)
                        return result
                    # Include terminal geometry's extra path and CDF calls.
                    units.append(unit(identity, "N5", 2*(512*63+512), 10., operation))
                logs = [r["log_estimate"] for r in units]
                record = {"cell": row["cell"], "kind": kind, "islands": islands,
                          "q_digest": row["q_digest"] if kind == "risk" else None,
                          "summary": log_moments(torch.tensor(logs, dtype=torch.float64)),
                          "sensitivity": sensitivity(logs, bootstrap_seed=seed(row['cell']['id']+kind+str(islands), "bootstrap")), "units": units}
                coverage.append(record)
                print(json.dumps({"phase": "N5", "cell": row["cell"]["id"], "kind": kind,
                                  "islands": islands, "summary": record["summary"]}), flush=True)
    result = {"schema": "npi.overnight-reference-recovery.result.v1", "status": "complete_development",
              "config": config, "source": source, "environment": environment, "toy": toy_results,
              "profile": profile, "is_comparisons": is_records, "extension_run": extend, "coverage": coverage,
              "allocation": allocation,
              "accounting": {p: journal.accounting(p) for p in config["limits"]},
              "p2_authorized": False, "production_qualification": False}
    write_json_atomic_nonoverwriting(directory/"result.json", result)
    return result


def audit(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload["p2_authorized"] or payload["production_qualification"]:
        raise ValueError("development results cannot unlock production")
    audit_snapshot(ROOT, payload["source"], payload["config"])
    directory = path.parent
    manifest = json.loads((directory/"manifest.json").read_text())
    journal = RecoveryJournal(directory, manifest)
    if hashlib.sha256((ROOT/payload["config"]["proposal_artifact_path"]).read_bytes()).hexdigest() != manifest["input_artifact_sha256"]:
        raise ValueError("input artifact changed")
    _, original_rows = inputs(payload["config"])
    for identity in manifest["expected_base_units"]:
        file = directory/f"{identity}.finish.json"
        if not file.exists() or journal._read(file)["status"] != "complete":
            raise ValueError("incomplete expected grid")
    for record in payload["toy"] + payload["coverage"]:
        logs = torch.tensor([r["log_estimate"] for r in record["units"]], dtype=torch.float64)
        if canonical_sha256(log_moments(logs)) != canonical_sha256(record["summary"]):
            raise ValueError("summary arithmetic mismatch")
        for whole in record["units"]:
            island_logs = torch.tensor([i["log_normalizer"] for i in whole["islands"]], dtype=torch.float64)
            reconstructed = float(torch.logsumexp(island_logs, 0))-math.log(len(island_logs))
            if record.get("kind") == "risk":
                binding = next(r for r in manifest["q_bindings"] if r["cell"]["id"] == record["cell"]["id"] and r["parent"] == 0)
                q = proposal_from_parameters(next(r["q_parameters"] for r in original_rows if r["q_digest"] == binding["q"]))
                reconstructed -= math.log(q.defensive_mass)
            if not math.isclose(reconstructed, whole["log_estimate"], rel_tol=0, abs_tol=1e-12):
                raise ValueError("island normalizer average mismatch")
    for record in payload["is_comparisons"]:
        reconstructed_logs = {}
        reconstructed_walls = {}
        for method in ("full", "marginal"):
            blocks = []
            for batch in range(32):
                ident = f"{record['prefix']}-{record['cell']['id']}-{record['kind']}-{record['replication']}-{method}-{batch}"
                unit_record = journal._read(directory/f"{ident}.finish.json")["result"]
                values = torch.tensor(unit_record["logs"], dtype=torch.float64)
                if len(values) != payload["config"]["batch_size"] or log_moments(values) != unit_record["summary"]:
                    raise ValueError("IID shard arithmetic/count mismatch")
                if unit_record["q_digest"] != record["q_digest"]:
                    raise ValueError("fixed q binding mismatch")
                if unit_record["seeds"] != {s: seed(ident, s) for s in ("path", "label", "pair")}:
                    raise ValueError("seed binding mismatch")
                blocks.append(unit_record)
            logs = torch.tensor([v for b in blocks for v in b["logs"]], dtype=torch.float64)
            if log_moments(logs) != record["methods"][method]["summary"]:
                raise ValueError("IID merged summary mismatch")
            reconstructed_logs[method] = logs
            reconstructed_walls[method] = sum(b["wall_seconds"] for b in blocks)
        if compare(reconstructed_logs["full"], reconstructed_logs["marginal"],
                   reconstructed_walls["full"], reconstructed_walls["marginal"]) != record["comparison"]:
            raise ValueError("comparison arithmetic mismatch")
    for phase, expected in payload["accounting"].items():
        if list(journal.accounting(phase)) != expected:
            raise ValueError("cumulative accounting mismatch")
    return {"status": "audit_pass", "p2_authorized": False, "production_qualification": False,
            "complete_base_units": len(manifest["expected_base_units"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT/"configs/post_audit/overnight_reference_recovery_v1.yaml")
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--stop-after-units", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.audit:
        print(json.dumps(audit(args.audit)))
        return
    config = yaml.safe_load(args.config.read_text())
    try:
        result = run(config, stop_after_units=args.stop_after_units)
        print(json.dumps({"status": result["status"], "accounting": result["accounting"]}))
    except Exception as error:
        directory = ROOT/config["run_directory"]
        if directory.exists():
            stamp = time.time_ns()
            write_json_atomic_nonoverwriting(directory/f"failure-{stamp}.json",
                {"status": "interrupted" if isinstance(error, InterruptedError) else "protocol_failure",
                 "type": type(error).__name__, "message": str(error), "p2_authorized": False})
        raise


if __name__ == "__main__":
    main()
