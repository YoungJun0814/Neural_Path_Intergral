"""Scoped adjacent mesh pilot with exact fine-law importance likelihood.

This pilot can run before P1 qualification; it cannot unlock large training.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import yaml

from experiments.post_audit_r1_diagnostics import _problem
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_v2_independent_reference import inputs
from src.path_integral.adjacent_latent_adapter import AdjacentLatentAdapter
from src.path_integral.cached_volterra_terminal import CachedShiftDensity, CachedVolterraTerminal
from src.path_integral.collective_variance_reference import CollectiveVarianceReference
from src.path_integral.recovery_checkpoint import RecoveryJournal
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.stable_gaussian import signed_log_normal_cdf_difference
from src.path_integral.volterra_excursion_guide import build_volterra_excursion_guide

ROOT = Path(__file__).resolve().parents[1]
PLAN = "docs/plans/P1_RECOVERY_P2_PARALLEL_EXECUTION_2026-10-09_KO.md"


def seed(identity, stream):
    return int(hashlib.sha256(("p2-adjacent-mesh-pilot-v1/"+identity+"/"+stream).encode()).hexdigest()[:15], 16)


def run():
    torch.set_num_threads(1)
    input_config = yaml.safe_load((ROOT/"configs/post_audit/p1_recovery_p2_parallel_v1.yaml").read_text())
    metadata, rows = inputs(input_config)
    config = {"scope": "mesh_pilot_not_production_or_continuum", "fine_steps": [32, 64], "count": 65536,
              "batch": 8192, "proposal": "half_guide_half_collective_covariance", "input": input_config}
    directory = ROOT/"results/post_audit/p2_adjacent_mesh_pilot_v1/session_20261009"
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config, extra_snapshot_paths=(PLAN,))
    write_json_atomic_nonoverwriting(directory/"source.json", source)
    manifest = {"config": config, "source": source["snapshot_sha256"], "limits": {"M": {"work": 1000000, "wall": 300.}}}
    journal = RecoveryJournal(directory, manifest)
    output = []
    for row in rows:
        if row["parent_training_rep"] != 0:
            continue
        for fine_steps in (32, 64):
            p = _problem(metadata["config"]["model"], row["cell"], fine_steps)
            coarse_p = replace(p, steps=fine_steps//2)
            key = row["cell"]["id"]+f"-{fine_steps}"
            cache: dict[str, Any] = {}
            def prepare(problem=p, coarse_problem=coarse_p, saved=cache):
                guide = build_volterra_excursion_guide(problem, **input_config["guide"])
                saved.update(fine=CachedVolterraTerminal.create(problem), coarse=CachedVolterraTerminal.create(coarse_problem),
                    adapter=AdjacentLatentAdapter.create(problem), guide=guide,
                    guide_density=CachedShiftDensity.create(guide), collective=CollectiveVarianceReference.create(problem))
                return {"actual_work": 2*problem.local_dimension}
            journal.unit(key+"-prepare", phase="M", work=512, maximum_wall=15., operation=prepare)
            parts = []
            for batch in range(8):
                identity = key+f"-{batch}"
                def operation(ident=identity, problem=p, cache=cache):
                    start = time.perf_counter()
                    count = 8192
                    z1 = cache["guide"].sample(count, path_seed=seed(ident, "guide-path"), label_seed=seed(ident, "guide-label")).samples
                    z2 = cache["collective"].sample(count, path_seed=seed(ident, "collective-path"), label_seed=seed(ident, "collective-label"))
                    label = torch.rand(count, dtype=torch.float64, generator=torch.Generator().manual_seed(seed(ident, "top-label"))) < .5
                    z = torch.where(label[:, None], z1, z2)
                    density = torch.logaddexp(cache["guide_density"].log_q_over_p(z), cache["collective"].log_q_over_p(z))-math.log(2)
                    eps = torch.randn((count, problem.steps//2), dtype=torch.float64,
                                      generator=torch.Generator().manual_seed(seed(ident, "augmentation")))
                    coarse_z = cache["adapter"].coarse(z, eps)
                    thresholds = []
                    for evaluator, coordinates in ((cache["fine"], z), (cache["coarse"], coarse_z)):
                        terminal, integrated, _ = evaluator.quantities(coordinates)
                        thresholds.append((math.log(problem.task.level)-terminal)/torch.sqrt((1-problem.rho**2)*integrated))
                    gf, gc = [torch.special.log_ndtr(x) for x in thresholds]
                    signs, log_abs = signed_log_normal_cdf_difference(*thresholds)
                    return {"fine": (gf-density).tolist(), "coarse": (gc-density).tolist(),
                        "sign": signs.tolist(), "difference_log_absolute": (log_abs-density).tolist(),
                        "wall": time.perf_counter()-start, "actual_work": 2*count,
                        "seeds": {s: seed(ident, s) for s in ("guide-path", "guide-label", "collective-path", "collective-label", "top-label", "augmentation")}}
                parts.append(journal.unit(identity, phase="M", work=16384, maximum_wall=15., operation=operation))
            fine = torch.tensor([v for b in parts for v in b["fine"]], dtype=torch.float64)
            coarse = torch.tensor([v for b in parts for v in b["coarse"]], dtype=torch.float64)
            signs = torch.tensor([v for b in parts for v in b["sign"]], dtype=torch.float64)
            logs = torch.tensor([v for b in parts for v in b["difference_log_absolute"]], dtype=torch.float64)
            scale = float(torch.max(torch.cat((fine, coarse, logs))))
            f, c, difference = torch.exp(fine-scale), torch.exp(coarse-scale), signs*torch.exp(logs-scale)
            mean_f, mean_c, mean_d = float(f.mean()), float(c.mean()), float(difference.mean())
            relative = mean_d/mean_f
            ratio_se = float((difference-relative*f).std(unbiased=True))/(math.sqrt(len(f))*mean_f)
            uncertainty_upper = abs(relative)+2.4977054744123737*ratio_se  # fixed 4-comparison normal approximation
            output.append({"cell": row["cell"], "fine_steps": fine_steps, "coarse_steps": fine_steps//2,
                "fine_mean": math.exp(scale)*mean_f, "coarse_mean": math.exp(scale)*mean_c,
                "signed_difference_mean": math.exp(scale)*mean_d,
                "fine_rse": float(f.std(unbiased=True))/(math.sqrt(len(f))*mean_f),
                "coarse_rse": float(c.std(unbiased=True))/(math.sqrt(len(c))*mean_c),
                "relative_fine_minus_coarse": relative, "relative_difference_se": ratio_se,
                "upper_absolute_relative_difference": uncertainty_upper,
                "small_difference_development_pass": uncertainty_upper <= .1,
                "maximum_fine_contribution_share": float(f.max()/f.sum()),
                "paired_mean_identity_error_scaled": abs(mean_f-mean_c-mean_d),
                "count": len(f), "scope": "paired_delta_method_mesh_pilot_not_continuum_certificate"})
            print(key+" mesh pilot complete", flush=True)
    phase_wall = time.time()-journal._read(directory/"phase-M.clock.json")["started_unix"]
    if phase_wall > 300:
        raise TimeoutError("mesh summary-inclusive cap")
    result = {"config": config, "source": source, "records": output,
              "phase_wall": phase_wall, "work": journal.accounting("M")[0],
              "coupling_oracle_implemented": True, "p2_qualified": False, "large_training_authorized": False}
    write_json_atomic_nonoverwriting(directory/"result.json", result)
    if time.time()-journal._read(directory/"phase-M.clock.json")["started_unix"] > 300:
        write_json_atomic_nonoverwriting(directory/"final_io_failure.json", {"status": "mesh_wall_failure"})
        raise TimeoutError("mesh final I/O cap")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"status": "scoped_mesh_pilot_complete", "work": result["work"], "p2_qualified": False}))
