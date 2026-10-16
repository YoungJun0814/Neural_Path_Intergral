"""Post-hoc DEVELOPMENT tail diagnosis, not fitting or confirmation evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from experiments.post_audit_r1_diagnostics import _problem
from src.path_integral.conditional_second_moment import log_auxiliary_second_moment
from src.path_integral.research_result_contract import canonical_digest
from src.path_integral.seed_ledger import SeedKey, SeedLedger
from src.path_integral.volterra_conditional_payoffs import evaluate_rbergomi_conditional_terminal
from src.path_integral.volterra_excursion_guide import (
    build_volterra_excursion_guide,
    volterra_monitoring_operator,
)
from src.path_integral.weighted_bank_mixture import proposal_from_parameters

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    raw = args.result.read_bytes()
    payload = json.loads(raw)
    config = payload["config"]
    original = json.loads((ROOT/config["proposal_artifact_path"]).read_bytes())
    ledger = SeedLedger.from_dict(payload["seed_ledger"])
    for cell in sorted({r["cell"]["id"] for r in payload["records"]}):
        rows = [r for r in payload["records"] if r["cell"]["id"] == cell]
        row = max(rows, key=lambda r: next(e for e in r["estimators"] if e["id"] == "risk-auxiliary")["summary"]["relative_se"])
        e = next(e for e in row["estimators"] if e["id"] == "risk-auxiliary")
        batch = max(range(len(e["batches"])), key=lambda i: e["batches"][i]["log_mean"]
            + math.log(e["batches"][i]["count"]*e["batches"][i]["maximum_fraction"]))
        old = next(o for o in original["records"] if o["cell"]["id"] == cell and o["parent_training_rep"] == row["parent_training_rep"])
        params = next(c["proposal_parameters"] for c in old["candidates"] if c["method"] == config["method"])
        q, r = proposal_from_parameters(params), proposal_from_parameters(e["r_parameters"])
        problem = _problem(original["config"]["model"], old["cell"], original["config"]["smc"]["steps"])

        def seed(stream: str, task: str = problem.task_id, batch_id: int = batch,
                 parent: int = row["parent_training_rep"], fit: int = row["auxiliary_training_rep"]) -> int:
            return ledger.lookup(SeedKey("r2-aux-risk-"+canonical_digest(config)[:16], "auxiliary-final",
                "risk-auxiliary", task, batch_id, parent, f"fit-{fit}-{stream}"))

        draw = r.sample(e["batches"][batch]["count"], path_seed=seed("path"), label_seed=seed("label"))
        value = evaluate_rbergomi_conditional_terminal(problem, draw.samples)
        logq, logr = q.log_q_over_p(draw.samples), draw.log_q_over_p
        logy = log_auxiliary_second_moment(value.payoffs.log_left_probability, logq, logr)
        expected_logmean = e["batches"][batch]["log_mean"]
        if abs(float(torch.logsumexp(logy, 0)-math.log(len(logy)))-expected_logmean) > 1e-10:
            raise ValueError("replayed batch differs from frozen result")
        top = torch.topk(logy, 5).indices
        b = volterra_monitoring_operator(problem)
        scores = draw.samples[top]@b.T/torch.linalg.vector_norm(b, dim=1)
        guide = build_volterra_excursion_guide(problem, amplitudes=[2., 4., 6., 8.], price_amplitudes=[0., 2., 4.])
        lift = guide.log_q_over_p(draw.samples[top])-logr[top]
        paths = problem.simulate_local(draw.samples[top])
        output = []
        for j, index in enumerate(top.tolist()):
            output.append({"log_g": float(value.payoffs.log_left_probability[index]),
                "q_natural_posterior": q.defensive_mass*math.exp(-float(logq[index])),
                "r_natural_posterior": r.defensive_mass*math.exp(-float(logr[index])),
                "peak_standardized_volterra": float(scores[j].max()),
                "peak_step": int(scores[j].argmax())+1,
                "integrated_variance": float(value.integrated_variance[index]),
                "largest_variance_cell_share": float(paths.variance[j, :-1].max()/paths.variance[j, :-1].sum()),
                "guide_log_density_lift_over_r": float(lift[j])})
        print(json.dumps({"cell": cell, "parent": row["parent_training_rep"], "fit": row["auxiliary_training_rep"],
            "selection": "post_hoc_worst_rse_development_row_only", "batch": batch,
            "result_sha256": hashlib.sha256(raw).hexdigest(), "points": output,
            "performance_claim": False}), flush=True)


if __name__ == "__main__":
    main()
