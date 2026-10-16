"""Exact last-pair Rao--Blackwell probability reference, unchanged raw M2."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from experiments.post_audit_canonical_static_stratification import prepare
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_v2_independent_reference import inputs
from src.path_integral.gaussian_mixture_marginal import ShiftMixtureMarginal
from src.path_integral.paired_normalizer_difference import paired_normalizer_difference
from src.path_integral.recovery_checkpoint import RecoveryJournal
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.structural_v2_inventory import audit_snapshot
from src.path_integral.structural_v2_reference import (
    log_moments,
    merge_moments,
    precision_count,
    sensitivity,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = "docs/plans/CANONICAL_LAST_PAIR_PROBABILITY_REFERENCE_PILOT_2026-10-10_KO.md"
DIRECTORY = "results/post_audit/canonical_last_pair_probability_v1/session_20261010"
TARGETS = ["mu-full", "mu-rb"]+[f"risk-q{j}" for j in range(5)]


def seed(rep: int, stream: str) -> int:
    return int(hashlib.sha256((f"canonical-last-pair-mu-v1/{rep}/{stream}").encode()).hexdigest()[:15], 16)


def unit_values(rep: int, c: dict[str, Any]) -> dict[str, Any]:
    begin = time.perf_counter()
    streams = {s: seed(rep, s) for s in ("path", "label")}
    z = c["guide"].sample(32768, path_seed=streams["path"], label_seed=streams["label"]).samples
    values = []
    for batch in z.split(4096):
        u = batch[:, :-2]
        cache = c["evaluator"].last_pair(u, c["guide"])
        full_g = torch.special.log_ndtr((cache.a-cache.b*batch[:, -2])/torch.sqrt(cache.c_squared))
        full_r = c["density"].log_q_over_p(batch)
        rb = cache.log_mu_mean()-c["marginal"].log_q_over_p(u)
        values.append(torch.stack([full_g-full_r, rb]+[2*full_g-q.log_q_over_p(batch)-full_r for q in c["qdensities"]], 1))
    vector = torch.cat(values)
    return {"rep": rep, "seeds": streams, "actual_work": 32768,
            "moments": [log_moments(vector[:, j]) for j in range(7)], "wall": time.perf_counter()-begin,
            "q_digests": c["q_digests"]}


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(records) != 8 or {r["rep"] for r in records} != set(range(8)):
        raise ValueError("incomplete fixed paired pilot")
    return [{"target": target, "summary": merge_moments([r["moments"][j] for r in records]),
             "sensitivity": sensitivity([r["moments"][j]["log_mean"] for r in records], bootstrap_seed=seed(200, target)),
             "allocation": precision_count(merge_moments([r["moments"][j] for r in records]),
                 target_rse=.015, safety_factor=2, minimum=32*32768, maximum=8388608, multiple=32768)}
            for j, target in enumerate(TARGETS)]


def run() -> dict[str, Any]:
    torch.set_num_threads(1)
    input_config = yaml.safe_load((ROOT/"configs/post_audit/p1_coverage_envelope_recovery_v2.yaml").read_text())
    metadata, rows = inputs(input_config)
    config = {"input": input_config, "blocks": 8, "count_per_block": 32768,
              "scope": "canonical_exact_last_pair_mu_development"}
    directory = ROOT/DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config, extra_snapshot_paths=(PLAN,))
    write_json_atomic_nonoverwriting(directory/"source.json", source)
    manifest = {"config": config, "source": source["snapshot_sha256"], "expected_units": [f"block-{k}" for k in range(8)],
                "input_sha256": hashlib.sha256((ROOT/input_config["proposal_artifact_path"]).read_bytes()).hexdigest(),
                "limits": {"L": {"work": 300000, "wall": 600.}}}
    journal = RecoveryJournal(directory, manifest)
    c: dict[str, Any] = {}
    def preparation():
        c.update(prepare(metadata, rows, input_config))
        c["marginal"] = ShiftMixtureMarginal.from_full(c["guide"], tuple(range(62)))
        return {"actual_work": 64}
    journal.unit("preparation", phase="L", work=64, maximum_wall=30., operation=preparation)
    records = []
    for rep in range(8):
        def operation(k=rep):
            return unit_values(k, c)
        records.append(journal.unit(f"block-{rep}", phase="L", work=32768, maximum_wall=30., operation=operation))
    summaries = summarize(records)
    output = {"config": config, "source": source, "manifest": manifest, "summaries": summaries,
              "paired_mu_diagnostic": paired_normalizer_difference(
                  torch.tensor([r["moments"][0]["log_mean"] for r in records], dtype=torch.float64),
                  torch.tensor([r["moments"][1]["log_mean"] for r in records], dtype=torch.float64), comparisons=1),
              "work": journal.accounting("L")[0], "units": 8,
              "p1_qualified": False, "p2_qualified": False, "large_training_authorized": False,
              "phase_wall": time.time()-journal._read(directory/"phase-L.clock.json")["started_unix"]}
    write_json_atomic_nonoverwriting(directory/"result.json", output)
    if time.time()-journal._read(directory/"phase-L.clock.json")["started_unix"] > 600:
        write_json_atomic_nonoverwriting(directory/"final_io_failure.json", {"status": "wall_failure"})
        raise TimeoutError("summary/final I/O cap")
    return output


def audit(path: Path) -> dict[str, Any]:
    torch.set_num_threads(1)
    output = json.loads(path.read_text())
    config, manifest = output["config"], output["manifest"]
    audit_snapshot(ROOT, output["source"], config)
    if (manifest["source"] != output["source"]["snapshot_sha256"]
            or manifest["input_sha256"] != hashlib.sha256((ROOT/config["input"]["proposal_artifact_path"]).read_bytes()).hexdigest()
            or manifest["expected_units"] != [f"block-{k}" for k in range(8)]):
        raise ValueError("changed source/input/grid")
    metadata, rows = inputs(config["input"])
    c = prepare(metadata, rows, config["input"])
    c["marginal"] = ShiftMixtureMarginal.from_full(c["guide"], tuple(range(62)))
    journal = RecoveryJournal(path.parent, manifest)
    records = []
    for rep in range(8):
        finish = journal._read(path.parent/f"block-{rep}.finish.json")
        if finish["status"] != "complete":
            raise ValueError("failed block")
        r = finish["result"]
        if (r["rep"] != rep or r["seeds"] != {s: seed(rep, s) for s in ("path", "label")}
                or r["q_digests"] != c["q_digests"] or r["actual_work"] != 32768):
            raise ValueError("changed unit binding")
        if rep in (0, 7) and unit_values(rep, c)["moments"] != r["moments"]:
            raise ValueError("block replay mismatch")
        records.append(r)
    z = c["guide"].sample(1024, path_seed=seed(300, "path"), label_seed=seed(300, "label")).samples
    cache = c["evaluator"].last_pair(z[:, :-2], c["guide"])
    reconstructed = torch.special.log_ndtr((cache.a-cache.b*z[:, -2])/torch.sqrt(cache.c_squared))
    torch.testing.assert_close(reconstructed, c["evaluator"].log_probability(z), rtol=2e-11, atol=2e-10)
    diagnostic = paired_normalizer_difference(
        torch.tensor([r["moments"][0]["log_mean"] for r in records], dtype=torch.float64),
        torch.tensor([r["moments"][1]["log_mean"] for r in records], dtype=torch.float64), comparisons=1)
    if (summarize(records) != output["summaries"] or diagnostic != output["paired_mu_diagnostic"]
            or journal.accounting("L")[0] != output["work"]
            or output["work"] != 262208 or output["units"] != 8
            or any(output[k] for k in ("p1_qualified", "p2_qualified", "large_training_authorized"))):
        raise ValueError("changed summary/cost/gates")
    if (path.parent/"phase-L.violation.json").exists() or (path.parent/"final_io_failure.json").exists():
        raise ValueError("operational failure")
    return {"status": "source_q_seed_cdf_cache_density_cost_numerical_replay_pass_not_precision_certificate",
            "replayed_draws": 65536, "full_cache_comparison_draws": 1024,
            "audit_forward_evaluations": 64+65536+2048}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit")
    args = parser.parse_args()
    print(json.dumps(audit(ROOT/args.audit) if args.audit else
                     {k: v for k, v in run().items() if k in ("work", "units", "phase_wall")}))
