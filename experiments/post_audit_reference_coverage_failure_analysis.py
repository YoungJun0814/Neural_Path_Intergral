"""Replay the maximum contribution from the failed fixed-count reference run."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from experiments.post_audit_high_eta_precision_validation import grid
from experiments.post_audit_p1_recovery_p2_parallel import identity
from experiments.post_audit_r1_diagnostics import _problem
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_v2_independent_reference import inputs
from src.path_integral.cached_volterra_terminal import CachedShiftDensity, CachedVolterraTerminal
from src.path_integral.collective_variance_reference import CollectiveVarianceReference
from src.path_integral.conditional_excursion_reference import ConditionalExcursionReference
from src.path_integral.coverage_envelope_reference import CoverageEnvelopeReference
from src.path_integral.recovery_checkpoint import RecoveryJournal
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.weighted_bank_mixture import proposal_from_parameters

ROOT = Path(__file__).resolve().parents[1]
INPUT = "results/post_audit/high_eta_precision_validation_v1/session_20261009/result.json"
OUTPUT = "results/post_audit/reference_coverage_failure_analysis_v2/session_20261010/result.json"


def run() -> dict[str, Any]:
    torch.set_num_threads(1)
    original_path = ROOT/INPUT
    original = json.loads(original_path.read_text())
    if original["decision"]["high_eta_parent0_development_precision_pass"]:
        raise ValueError("this replay is intended for the preserved failed validation")
    journal = RecoveryJournal(original_path.parent, original["manifest"])
    records = [(identity(j), journal._read(original_path.parent/(identity(j)+".finish.json"))["result"])
               for j in grid() if j["method"] == "excursion-iid"]
    best = max(((max(r["logs"]["mu"]), ident, r["logs"]["mu"].index(max(r["logs"]["mu"]))) for ident, r in records))
    maximum, ident, index = best
    record = next(r for unit, r in records if unit == ident)
    config = {"input": INPUT, "input_sha256": hashlib.sha256(original_path.read_bytes()).hexdigest(),
              "maximum_unit": ident, "maximum_sample_index": index, "scope": "diagnostic_replay_not_new_reference_samples"}
    output_path = ROOT/OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source = freeze_source(output_path, config, extra_snapshot_paths=("docs/plans/COVERAGE_ENVELOPE_RECOVERY_PROTOCOL_2026-10-10_KO.md",))
    metadata, rows = inputs(original["config"]["input"])
    row = next(r for r in rows if r["cell"]["id"] == "high-eta200-k1" and r["parent_training_rep"] == 0)
    p = _problem(metadata["config"]["model"], row["cell"], metadata["config"]["smc"]["steps"])
    reference = ConditionalExcursionReference.create(p)
    z = reference.sample(4096, path_seed=record["seeds"]["path"], label_seed=record["seeds"]["label"])
    x = z[index:index+1]
    evaluator = CachedVolterraTerminal.create(p)
    logg, logd = evaluator.log_probability(x), reference.log_q_over_p(x)
    q = proposal_from_parameters(row["q_parameters"])
    risk_log = 2*logg-CachedShiftDensity.create(q).log_q_over_p(x)-logd
    if float(logg-logd) != maximum or float(risk_log) != record["logs"]["risk"][index]:
        raise ValueError("original maximum raw contribution did not replay exactly")
    terminal, integrated, variance = evaluator.quantities(x)
    standardized = x @ reference.units.T
    new_d = CoverageEnvelopeReference.create(p).log_q_over_p(x)
    output = {"config": config, "source": source, "seeds": record["seeds"], "q_digest": row["q_digest"],
              "maximum_contribution": math.exp(maximum), "original_logg": float(logg), "original_log_density": float(logd),
              "maximum_standardized_volterra": float(standardized.max()),
              "number_monitor_times_above_two": int((standardized >= 2).sum()),
              "number_monitor_times_above_three": int((standardized >= 3).sum()),
              "number_monitor_times_above_four": int((standardized >= 4).sum()),
              "terminal_conditional_mean_logspot": float(terminal), "integrated_variance": float(integrated),
              "maximum_variance": float(variance.max()), "latent_squared_norm": float(x.square().sum()),
              "latent_sha256": hashlib.sha256(x.numpy().tobytes()).hexdigest(),
              "collective_log_density": float(CollectiveVarianceReference.create(p).log_q_over_p(x)),
              "envelope_log_density": float(new_d),
              "same_path_envelope_contribution": math.exp(float(logg-new_d)),
              "same_path_density_increase": math.exp(float(new_d-logd)),
              "interpretation": "proposal_support_concentration_gap_not_density_replay_error; single_path_not_variance_theorem",
              "potential_evaluations": 3*p.local_dimension+2,
              "replayed_generated_latents": 4096, "scope_of_cost": "3_geometry_probes_and_2_single_path_terminal_evaluations",
              "p1_qualified": False, "p2_qualified": False}
    write_json_atomic_nonoverwriting(output_path, output)
    return output


if __name__ == "__main__":
    result = run()
    print(json.dumps({k: v for k, v in result.items() if k not in ("source", "config")}, indent=2))
