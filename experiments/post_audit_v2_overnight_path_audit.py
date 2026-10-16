"""Bounded original-simulator replay and honest operational qualification."""

from __future__ import annotations

import json
import math
import time
from statistics import NormalDist

import torch

from experiments.post_audit_r1_diagnostics import _problem
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_v2_independent_reference import inputs
from experiments.post_audit_v2_overnight_reference_recovery import PLAN, ROOT, TOYS, seed
from src.path_integral.gaussian_mixture_marginal import ShiftMixtureMarginal
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.structural_v2_conditional_reference import last_pair_cache
from src.path_integral.structural_v2_terminal_geometry import terminal_geometry
from src.path_integral.volterra_conditional_payoffs import evaluate_rbergomi_conditional_terminal
from src.path_integral.weighted_bank_mixture import proposal_from_parameters


def region_log_integral(name: str, power: int) -> float:
    oracle = TOYS[name]
    normal, terms = NormalDist(), []
    for a, m, s in zip(oracle.amplitudes, oracle.centers, oracle.scales, strict=True):
        partners = [(1., (0., 0.), math.inf)] if power == 1 else list(zip(oracle.amplitudes, oracle.centers, oracle.scales, strict=True))
        for b, n, t in partners:
            precision = 1+1/s**2+1/t**2
            vector = [x/s**2+y/t**2 for x, y in zip(m, n, strict=True)]
            mean = [x/precision for x in vector]
            sd = 1/math.sqrt(precision)
            constant = sum(x*x for x in m)/s**2+sum(x*x for x in n)/t**2
            mass = (normal.cdf((7.5-mean[0])/sd)-normal.cdf((2.5-mean[0])/sd))*(normal.cdf((2.-mean[1])/sd)-normal.cdf((-2.-mean[1])/sd))
            if mass > 0:
                terms.append(math.log(a*b)-math.log(precision)-.5*(constant-sum(x*x for x in vector)/precision)+math.log(mass))
    return float(torch.logsumexp(torch.tensor(terms, dtype=torch.float64), 0))


def main() -> None:
    torch.set_num_threads(1)
    directory = ROOT/"results/post_audit/overnight_reference_recovery_v1/session_20261008"
    result = json.loads((directory/"result.json").read_text())
    output = directory/"path_audit.json"
    if output.exists():
        raise FileExistsError(output)
    spec = {"scope": "first-batch-original-simulator-replay-not-new-performance-samples",
            "original_source": result["source"]["snapshot_sha256"], "parent": 0,
            "batch": 0, "work_cap": 1000000, "wall_cap": 120.}
    source = freeze_source(output, spec, extra_snapshot_paths=(PLAN,))
    metadata, rows = inputs(result["config"])
    timer, work = time.perf_counter(), 0
    records = []
    for comparison in result["is_comparisons"]:
        if comparison["parent"] != 0:
            continue
        row = next(r for r in rows if r["cell"] == comparison["cell"] and r["parent_training_rep"] == comparison["parent"])
        problem = _problem(metadata["config"]["model"], row["cell"], metadata["config"]["smc"]["steps"])
        q = proposal_from_parameters(row["q_parameters"])
        guide_parameters = json.loads((directory/f"N3-guide-{row['cell']['id']}.finish.json").read_text())["payload"]["result"]
        guide = proposal_from_parameters(guide_parameters)
        marginal = ShiftMixtureMarginal.from_full(guide, tuple(range(problem.local_dimension-2)))
        for method in ("full", "marginal"):
            ident = f"{comparison['prefix']}-{row['cell']['id']}-{comparison['kind']}-{comparison['replication']}-{method}-0"
            original = json.loads((directory/f"{ident}.finish.json").read_text())["payload"]["result"]
            n = result["config"]["batch_size"]
            needed = 4*n
            if work+needed > spec["work_cap"] or time.perf_counter()-timer > spec["wall_cap"]:
                raise TimeoutError("bounded path audit cap")
            work += needed
            if method == "full":
                draw = guide.sample(n, path_seed=seed(ident, "path"), label_seed=seed(ident, "label"))
                z = draw.samples
                logg = evaluate_rbergomi_conditional_terminal(problem, z).payoffs.log_left_probability
                logs = (2*logg-q.log_q_over_p(z) if comparison["kind"] == "risk" else logg)-draw.log_q_over_p
            else:
                a = marginal.sample(n, path_seed=seed(ident, "path"), label_seed=seed(ident, "label"))
                pair = torch.randn((n, 1, 2), dtype=torch.float64, generator=torch.Generator().manual_seed(seed(ident, "pair")))
                cache = last_pair_cache(problem, a, q)
                logs = (cache.log_inner_risk(pair)[:, 0] if comparison["kind"] == "risk" else cache.log_mu_mean())-marginal.log_q_over_p(a)
                z = torch.cat((a, pair[:, 0]), 1)
            expected = torch.tensor(original["logs"], dtype=torch.float64)
            torch.testing.assert_close(logs, expected, rtol=2e-13, atol=2e-12)
            records.append({"identity": ident, "maximum_absolute_log_error": float((logs-expected).abs().max()),
                            "contribution_weighted_geometry": terminal_geometry(problem, z, torch.softmax(logs, 0)),
                            "mean_log_contribution": float(torch.logsumexp(logs, 0))-math.log(n)})
    clock = []
    for phase in ("N2", "N3", "N4", "N5"):
        starts = list(directory.glob(phase+"-*.start.json"))
        finishes = list(directory.glob(phase+"-*.finish.json"))
        elapsed = max(p.stat().st_mtime for p in finishes)-min(p.stat().st_mtime for p in starts)
        clock.append({"phase": phase, "wall_with_io_seconds_from_original_file_times": elapsed,
                      "cap_seconds": result["config"]["limits"][phase]["wall"],
                      "operational_pass": elapsed <= result["config"]["limits"][phase]["wall"]})
    region = []
    for record in result["toy"]:
        truth = region_log_integral(record["toy"], record["power"])
        estimated = []
        for whole in record["units"]:
            terms = [i["log_normalizer"]+math.log(i["snapshots"][-1]["region_weight"])
                     for i in whole["islands"] if i["snapshots"][-1]["region_weight"] > 0]
            estimated.append(float(torch.logsumexp(torch.tensor(terms, dtype=torch.float64), 0))-math.log(len(whole["islands"])))
        logmean = float(torch.logsumexp(torch.tensor(estimated, dtype=torch.float64), 0))-math.log(len(estimated))
        region.append({"toy": record["toy"], "power": record["power"], "islands": record["islands"],
                       "true_endpoint_region_mass": math.exp(truth-TOYS[record["toy"]].log_endpoint(record["power"])),
                       "region_integral_estimate_over_truth": math.exp(logmean-truth),
                       "scope": "eight_run_development_not_bias_certificate"})
    report = {"status": "arithmetic_path_audit_pass_operational_N4_fail", "spec": spec, "source": source,
              "records": records, "phase_clock_review": clock, "toy_region": region,
              "work": work, "wall_seconds": time.perf_counter()-timer, "p2_authorized": False,
              "original_result_preserved": True,
              "operational_issue": "callback-only wall excluded checkpoint I/O; N4 exceeds phase cap",
              "performance_qualification": False,
              "scope": "saved-first-shard-diagnostic-not-complete-population-mode-certificate"}
    write_json_atomic_nonoverwriting(output, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("source", "records", "toy_region")}))


if __name__ == "__main__":
    main()
