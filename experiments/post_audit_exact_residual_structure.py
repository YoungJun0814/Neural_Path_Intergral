"""Small deterministic structural oracle, not another rare-event MC campaign."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from typing import Any

import numpy as np
import torch
from scipy.integrate import quad
from scipy.special import log_ndtr, logsumexp, ndtr, roots_hermitenorm

from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_resource_sleep_recovery import ROOT, sha
from experiments.post_audit_small_grid_model_restart import prepare
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.residual_structural_oracle import (
    LastVarianceResidualSlice,
    bump_shift_moments,
    gaussian_log_rho,
    untouched_bump_floor,
)
from src.path_integral.structural_v2_inventory import audit_snapshot

DIRECTORY = "results/post_audit/exact_residual_structure_v1/session_20261021"
PLAN = "docs/plans/EXACT_RESIDUAL_STRUCTURAL_ORACLE_2026-10-21_KO.md"
THEORY = "docs/theory/EXACT_RESIDUAL_STRUCTURAL_CONTRACT_2026-10-21_KO.md"
PARENT = "results/post_audit/bounded_frozen_bank_contrast_v1/session_20261020"
FEATURE = "results/post_audit/frozen_q_path_attribution_v1/session_20261021"
EXPECTED = {PARENT: "323b00268d797b6e70222a748ce500ee4c5f7f1ff9da1e6724acd750cb3ee1bb",
    FEATURE: "eff4fd1406719163bdc333c62f79484377ba155ef3a6d58910c3739ba4dbd9f5"}


class Budget:
    def __init__(self) -> None:
        self.physical, self.scalar = 0, 0
        self.started = time.perf_counter()

    def charge(self, *, physical: int = 0, scalar: int = 0) -> None:
        if self.physical+physical > 100 or self.scalar+scalar > 50000 or time.perf_counter()-self.started > 900:
            raise TimeoutError("fixed structural-oracle cap exceeded")
        self.physical += physical
        self.scalar += scalar


def anchors() -> tuple[dict[str, Any], dict[str, Any]]:
    rows = []
    for name, digest in EXPECTED.items():
        path = ROOT/name/"result.json"
        value, receipt = [json.loads(p.read_text()) for p in (path, path.with_name("audit.json"))]
        if sha(path) != digest or receipt["result_sha256"] != digest:
            raise ValueError("immutable parent anchors changed")
        audit_snapshot(ROOT, value["source"], value["config"])
        rows.append(value)
    return rows[0], rows[1]


def exact_toy(budget: Budget) -> dict[str, Any]:
    base = bump_shift_moments((3., 0.), 1., (1.5, 0.))
    a, v = tuple(base["risk_target_mean"]), base["risk_target_variance"]
    rho_bad = gaussian_log_rho(a, v, (0., 4.), 1.)
    rho_matched = gaussian_log_rho(a, v, a, v)
    numeric = []
    # Independent scalar integration, separability only in this exact toy.
    for c, m in ((3., 1.5), (0., 0.)):
        def integrand(z: float, c: float = c, m: float = m) -> float:
            budget.charge(scalar=1)
            return math.exp(-(z-c)**2-.5*z*z-m*z+.5*m*m)/math.sqrt(2*math.pi)
        value, error = quad(integrand, -12, 12, epsabs=1e-13, epsrel=1e-10, limit=150)
        numeric.append({"value": value, "reported_error": error})
    check = abs(math.prod(r["value"] for r in numeric)/math.exp(base["log_m2"])-1)
    return {"center": [3., 0.], "width": 1., "q_mean": [1.5, 0.], "exact": base,
        "evaluation_bad_mean": [0., 4.], "log_evaluation_rho_bad": rho_bad,
        "log_evaluation_rho_matched": rho_matched, "N": 65536,
        "event_rse": math.sqrt(math.expm1(base["log_relative_second_moment"])/65536),
        "risk_evaluation_rse_bad": math.sqrt(math.expm1(rho_bad)/65536),
        "risk_evaluation_rse_matched": math.sqrt(max(0., math.expm1(rho_matched))/65536),
        "independent_scalar_m2_integrals": numeric, "m2_relative_error": check,
        "untouched_residual_log_floor": untouched_bump_floor((4.,), .75),
        "irrelevant_residual_floor": 1., "pass": check <= 1e-9 and abs(rho_matched) <= 1e-12,
        "not_defensive_mixture_population_certificate": True}


def scalar_moments(profile: LastVarianceResidualSlice, budget: Budget) -> dict[str, Any]:
    gh = []
    for n in (48, 96, 192):
        nodes, weights = roots_hermitenorm(n)
        budget.charge(scalar=n)
        logs = profile.log_probability(torch.tensor(nodes, dtype=torch.float64)).numpy()
        gh.append([float(np.exp(logsumexp(np.log(weights)+power*logs)-.5*math.log(2*math.pi))) for power in (1, 2)])
    adaptive, errors = [], []
    for power in (1, 2):
        def integrand(u: float, power: int = power) -> float:
            budget.charge(scalar=1)
            variance = math.exp(profile.log_initial_last_variance+profile.beta*u)
            value = ((profile.log_level-profile.preceding_terminal+.5*profile.dt*variance
                -profile.rho*math.sqrt(variance)*profile.last_dw)
                /math.sqrt((1-profile.rho**2)*(profile.preceding_integrated_variance+profile.dt*variance)))
            return math.exp(power*float(log_ndtr(value))-.5*u*u)/math.sqrt(2*math.pi)
        value, error = quad(integrand, -12, 12, epsabs=1e-13, epsrel=1e-9, limit=200)
        adaptive.append(value)
        errors.append(error)
    convergence = []
    for k in range(2):
        absolute = max(abs(gh[-1][k]-adaptive[k]), abs(gh[-1][k]-gh[-2][k]))
        relative = absolute/max(adaptive[k], 1e-300)
        convergence.append({"absolute": absolute, "relative": relative,
            "relative_resolved": relative <= 1e-5, "pass": relative <= 1e-5 or absolute <= 1e-12})
    mu, second = adaptive
    return {"gh_orders": [48, 96, 192], "gh": gh, "adaptive": adaptive,
        "adaptive_reported_errors_not_rigorous": errors, "absolute_omitted_tail_upper": float(2*ndtr(-12.)),
        "convergence": convergence, "resolved": all(c["pass"] for c in convergence),
        "conditional_relative_second_moment": second/mu**2 if mu > 0 and mu**2 > 0 else None,
        "conditional_variance": second-mu**2,
        "not_global_m2_or_production_quadrature_accuracy": True}


def density(spec: Any, z: torch.Tensor, weights: list[list[float]]) -> torch.Tensor:
    c = z @ spec.means.T-.5*spec.means.square().sum(1)
    return torch.stack([torch.logsumexp(c+torch.tensor(w, dtype=torch.float64).log(), 1) for w in weights], 1)


def context(cache: Any, spec: Any, z: torch.Tensor, weights: list[list[float]], budget: Budget) -> dict[str, Any]:
    budget.charge(physical=1)
    profile = LastVarianceResidualSlice.create(cache, z)
    values = torch.tensor([-4., -2., 0., 2., 4.], dtype=torch.float64)
    probes = []
    for index in (2*cache.problem.steps-3, 2*cache.problem.steps-1):
        points = z.repeat(5, 1)
        points[:, index] = values
        budget.charge(physical=5)
        full = cache.log_probability(points)
        sliced = (profile.log_probability(values) if index == 2*cache.problem.steps-3
            else profile.log_probability(z[0, 2*cache.problem.steps-3].repeat(5)))
        ratios = density(spec, points, weights)
        error = float((full-sliced).abs().max())
        variation = float(full.max()-full.min())
        invariant = bool(torch.equal(ratios, ratios[2].expand_as(ratios)))
        probes.append({"coordinate": index, "values": values.tolist(), "full_log_g": full.tolist(),
            "slice_log_g": sliced.tolist(), "maximum_log_error": error,
            "log_g_range": variation, "all11_density_ratios_exactly_invariant": invariant,
            "payoff_dependence_detected": bool(index == 2*cache.problem.steps-3 and variation > 1e-10),
            "pass": error <= 1e-8 and invariant and (index == 2*cache.problem.steps-3 or variation <= 1e-10)})
    return {"coordinates": z[0].tolist(), "profile": dataclasses.asdict(profile),
        "probes": probes, "conditional_moments": scalar_moments(profile, budget),
        "structural_probe_pass": all(p["pass"] for p in probes)}


def compute(parent: dict[str, Any], feature: dict[str, Any], budget: Budget) -> dict[str, Any]:
    prepared, _ = prepare({"model": parent["config"]["parent_model"], "cells": parent["config"]["cells"]})
    toy = exact_toy(budget)
    cells = []
    for cell, (cache, spec) in prepared.items():
        zeros = torch.where(spec.means.abs().max(0).values == 0)[0].tolist()
        if 61 not in zeros or 63 not in zeros:
            raise ValueError("frozen N32 coordinate support assumption failed")
        rows = []
        for role in ("zero", "discovery", "confirmation"):
            if role == "zero":
                z = torch.zeros((1, 64), dtype=torch.float64)
            else:
                record = next(r for r in feature[role] if r["cell"] == cell and r["block"] == 0)
                top = next(r for r in record["tops"] if r["q"] == 0)
                z = torch.tensor([top["z"]], dtype=torch.float64)
            rows.append({"role": role, **context(cache, spec, z, parent["family"][cell]["weights"], budget)})
        cells.append({"cell": cell, "exact_zero_mean_coordinates": zeros,
            "numerical_rank_default_tolerance": int(torch.linalg.matrix_rank(spec.means)),
            "singular_values": torch.linalg.svdvals(spec.means).tolist(), "contexts": rows})
        print("Completed fixed structural contexts: "+cell, flush=True)
    structural = (toy["pass"] and all(r["structural_probe_pass"] for c in cells for r in c["contexts"])
        and all(any(r["probes"][0]["payoff_dependence_detected"] for r in c["contexts"]) for c in cells))
    return {"toy": toy, "cells": cells, "structural_confirmed": structural,
        "all_context_quadratures_resolved": all(r["conditional_moments"]["resolved"] for c in cells for r in c["contexts"]),
        "decision": "STRUCTURAL_BLIND_COORDINATE_CONFIRMED" if structural else "INVALID_STRUCTURAL_ORACLE",
        "q_changed": False, "G_changed": False, "D3_authorized": False, "p2_qualified": False,
        "volterra_global_risk_floor_quantified": False}


def run() -> dict[str, Any]:
    torch.set_num_threads(1)
    parent, feature = anchors()
    path = ROOT/DIRECTORY
    if path.exists() and any(path.iterdir()):
        raise RuntimeError("one-shot structural evidence; no overwrite")
    path.mkdir(parents=True, exist_ok=True)
    config = {"anchors": EXPECTED, "contexts": ["zero", "discovery", "confirmation"],
        "probes": [-4., -2., 0., 2., 4.], "coordinates": [61, 63], "orders": [48, 96, 192],
        "physical_cap_including_audit": 100, "scalar_cap_including_audit": 50000, "wall_cap": 900}
    source = freeze_source(path/"run.json", config, extra_snapshot_paths=(PLAN, THEORY,
        PARENT+"/result.json", PARENT+"/audit.json", FEATURE+"/result.json", FEATURE+"/audit.json"))
    write_json_atomic_nonoverwriting(path/"protocol.json", {"config": config, "source": source})
    budget = Budget()
    try:
        summary = compute(parent, feature, budget)
    except Exception as e:
        write_json_atomic_nonoverwriting(path/"failure.json", {"error": type(e).__name__+": "+str(e),
            "physical_evaluations": budget.physical, "scalar_evaluations": budget.scalar})
        raise
    output = {"config": config, "source": source, "summary": summary,
        "physical_evaluations": budget.physical, "scalar_evaluations": budget.scalar,
        "wall": time.perf_counter()-budget.started}
    write_json_atomic_nonoverwriting(path/"result.json", output)
    return output


def audit() -> dict[str, Any]:
    torch.set_num_threads(1)
    parent, feature = anchors()
    path = ROOT/DIRECTORY
    result = json.loads((path/"result.json").read_text())
    audit_snapshot(ROOT, result["source"], result["config"])
    for name, value in result["source"]["snapshot_file_hashes"].items():
        if name.startswith(("src/", "experiments/")) and sha(ROOT/name) != value:
            raise ValueError("frozen source changed")
    prepared, _ = prepare({"model": parent["config"]["parent_model"], "cells": parent["config"]["cells"]})
    if (len(result["summary"]["cells"]) != 2
            or {c["cell"] for c in result["summary"]["cells"]} != set(prepared)):
        raise ValueError("complete structural cell family required")
    budget = Budget()
    if exact_toy(budget) != result["summary"]["toy"]:
        raise ValueError("complete analytic toy recomputation failed")
    for saved in result["summary"]["cells"]:
        cache, spec = prepared[saved["cell"]]
        if saved["exact_zero_mean_coordinates"] != torch.where(spec.means.abs().max(0).values == 0)[0].tolist():
            raise ValueError("zero support changed")
        if len(saved["contexts"]) != 3 or [r["role"] for r in saved["contexts"]] != ["zero", "discovery", "confirmation"]:
            raise ValueError("complete fixed context family required")
        for r in saved["contexts"]:
            if r["role"] == "zero":
                expected_coordinates = [0.]*64
            else:
                old = next(row for row in feature[r["role"]] if row["cell"] == saved["cell"] and row["block"] == 0)
                expected_coordinates = next(row["z"] for row in old["tops"] if row["q"] == 0)
            if r["coordinates"] != expected_coordinates or len(r["probes"]) != 2:
                raise ValueError("context coordinate binding changed")
            profile = LastVarianceResidualSlice(**r["profile"])
            for probe in r["probes"]:
                values = torch.tensor([-4., -2., 0., 2., 4.], dtype=torch.float64)
                is_live = probe["coordinate"] == 61
                budget.charge(scalar=5)
                sliced = profile.log_probability(values if is_live else torch.full_like(values, r["coordinates"][61]))
                points = torch.tensor([r["coordinates"]]*5, dtype=torch.float64)
                points[:, probe["coordinate"]] = values
                ratios = density(spec, points, parent["family"][saved["cell"]]["weights"])
                error = float((torch.tensor(probe["full_log_g"], dtype=torch.float64)-sliced).abs().max())
                valid = error <= 1e-8 and bool(torch.equal(ratios, ratios[2].expand_as(ratios)))
                if probe["slice_log_g"] != sliced.tolist() or probe["maximum_log_error"] != error or not valid or not probe["pass"]:
                    raise ValueError("saved scalar probe/density identity changed")
            if scalar_moments(profile, budget) != r["conditional_moments"]:
                raise ValueError("all scalar quadratures changed")
            if r["role"] == "zero":
                rebuilt = context(cache, spec, torch.zeros((1, 64), dtype=torch.float64),
                    parent["family"][saved["cell"]]["weights"], budget)
                if rebuilt != {k: v for k, v in r.items() if k != "role"}:
                    raise ValueError("representative full-law replay mismatch")
    if (result["physical_evaluations"] != 66 or budget.physical != 22
            or result["scalar_evaluations"]+budget.scalar > 50000 or result["wall"] > 900):
        raise ValueError("structural oracle accounting mismatch")
    structural = (result["summary"]["toy"]["pass"] and all(r["structural_probe_pass"]
        for c in result["summary"]["cells"] for r in c["contexts"])
        and all(any(r["probes"][0]["payoff_dependence_detected"] for r in c["contexts"])
            for c in result["summary"]["cells"]))
    if result["summary"]["structural_confirmed"] != structural or result["summary"]["D3_authorized"]:
        raise ValueError("structural decision changed")
    return {"status": "exact_toy_all_scalar_oracles_and_two_zero_context_full_replays_pass",
        "result_sha256": sha(path/"result.json"), "production_physical": result["physical_evaluations"],
        "audit_physical": budget.physical, "audit_scalar": budget.scalar,
        "total_scalar": result["scalar_evaluations"]+budget.scalar,
        "decision": result["summary"]["decision"], "p2_qualified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.audit:
        receipt = audit()
        write_json_atomic_nonoverwriting(ROOT/DIRECTORY/"audit.json", receipt)
        print(json.dumps(receipt))
    else:
        result = run()
        print(json.dumps({k: result[k] for k in ("physical_evaluations", "scalar_evaluations", "wall")}))
