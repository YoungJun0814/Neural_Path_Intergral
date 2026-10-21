"""Post-hoc implementation review; never overwrites the invalid original gate."""

from __future__ import annotations

import json
import time
from typing import Any

import torch

from experiments.post_audit_exact_residual_structure import (
    DIRECTORY,
    ROOT,
    Budget,
    anchors,
    context,
    density,
    exact_toy,
    scalar_moments,
)
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_resource_sleep_recovery import sha
from experiments.post_audit_small_grid_model_restart import prepare
from src.path_integral.gaussian_span_marginalization import GaussianMixtureShiftSpec
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.residual_structural_oracle import LastVarianceResidualSlice
from src.path_integral.structural_v2_inventory import audit_snapshot

ADDENDUM = "docs/plans/RESIDUAL_ORACLE_NUMERICAL_REVIEW_ADDENDUM_2026-10-21_KO.md"


def density_invariance_review(spec: GaussianMixtureShiftSpec, points: torch.Tensor,
                              weights: list[list[float]], coordinate: int) -> dict[str, Any]:
    if (points.shape != (5, spec.dimension) or points.dtype != torch.float64
            or points.device.type != "cpu" or not torch.isfinite(points).all()
            or not 0 <= coordinate < spec.dimension):
        raise ValueError("five finite CPU float64 points and a valid coordinate required")
    fixed = points.clone()
    fixed[:, coordinate] = fixed[2, coordinate]
    if not torch.equal(fixed, fixed[2].expand_as(fixed)) or not bool((spec.means[:, coordinate] == 0).all()):
        raise ValueError("exact algebraic invariance assumptions violated")
    w = torch.tensor(weights, dtype=torch.float64)
    if (w.ndim != 2 or w.shape[1] != len(spec.means) or not torch.isfinite(w).all()
            or not bool((w > 0).all()) or not torch.allclose(w.sum(1), torch.ones(len(w), dtype=torch.float64), rtol=0, atol=1e-12)):
        raise ValueError("positive normalized complete mixture weights required")
    ratios = density(spec, points, weights)
    maximum = float((ratios-ratios[2]).abs().max())
    scale = (1+float((points.abs() @ spec.means.abs().T).max())
             +.5*float(spec.means.square().sum(1).max())+float(w.log().abs().max()))
    tolerance = 512*torch.finfo(torch.float64).eps*scale
    if maximum > tolerance:
        raise ValueError("dense evaluation discrepancy exceeds roundoff diagnostic scale")
    return {"coordinate": coordinate, "exact_zero_means_and_fixed_other_coordinates": True,
        "maximum_dense_log_ratio_difference": maximum, "float64_diagnostic_tolerance": tolerance,
        "not_rigorous_interval_bound": True, "number_of_weight_vectors": len(weights)}


def review() -> dict[str, Any]:
    torch.set_num_threads(1)
    parent, feature = anchors()
    path = ROOT/DIRECTORY
    original = json.loads((path/"result.json").read_text())
    if original["summary"]["decision"] != "INVALID_STRUCTURAL_ORACLE" or original["physical_evaluations"] != 66:
        raise ValueError("review is specific to the preserved invalid original run")
    audit_snapshot(ROOT, original["source"], original["config"])
    for name, digest in original["source"]["snapshot_file_hashes"].items():
        if name.startswith(("src/", "experiments/", "tests/")) and sha(ROOT/name) != digest:
            raise ValueError("original frozen numerical source changed")
    prepared, _ = prepare({"model": parent["config"]["parent_model"], "cells": parent["config"]["cells"]})
    config = {"original_sha256": sha(path/"result.json"), "posthoc": True,
        "new_physical_cap": 22, "combined_physical_cap": 100, "combined_scalar_cap": 50000,
        "wall_cap_computation_only": 900}
    if (path/"numerical_review_protocol.json").exists():
        raise RuntimeError("one-shot review already started; no overwrite")
    source = freeze_source(path/"numerical_review.json", config,
        extra_snapshot_paths=(ADDENDUM, DIRECTORY+"/result.json", DIRECTORY+"/protocol.json"))
    write_json_atomic_nonoverwriting(path/"numerical_review_protocol.json", {"config": config, "source": source})
    budget = Budget()
    toy = exact_toy(budget)
    toy_scalar = budget.scalar
    if toy != original["summary"]["toy"] or not toy["pass"]:
        raise ValueError("analytic toy replay failed")
    rows, canonical_zero_scalar = [], None
    if len(original["summary"]["cells"]) != 2 or {c["cell"] for c in original["summary"]["cells"]} != set(prepared):
        raise ValueError("complete original cell family required")
    for c in original["summary"]["cells"]:
        cache, spec = prepared[c["cell"]]
        zeros = torch.where(spec.means.abs().max(0).values == 0)[0].tolist()
        if c["exact_zero_mean_coordinates"] != zeros or c["numerical_rank_default_tolerance"] != int(torch.linalg.matrix_rank(spec.means)):
            raise ValueError("original dictionary structure changed")
        if c["singular_values"] != torch.linalg.svdvals(spec.means).tolist():
            raise ValueError("original numerical singular values changed")
        if [r["role"] for r in c["contexts"]] != ["zero", "discovery", "confirmation"]:
            raise ValueError("fixed original contexts required")
        for r in c["contexts"]:
            if r["role"] == "zero":
                expected = [0.]*64
            else:
                old = next(x for x in feature[r["role"]] if x["cell"] == c["cell"] and x["block"] == 0)
                expected = next(x["z"] for x in old["tops"] if x["q"] == 0)
            if r["coordinates"] != expected or [p["coordinate"] for p in r["probes"]] != [61, 63]:
                raise ValueError("original path/coordinate binding changed")
            profile = LastVarianceResidualSlice(**r["profile"])
            checks = []
            for probe in r["probes"]:
                values = torch.tensor([-4., -2., 0., 2., 4.], dtype=torch.float64)
                points = torch.tensor([expected]*5, dtype=torch.float64)
                points[:, probe["coordinate"]] = values
                check = density_invariance_review(spec, points, parent["family"][c["cell"]]["weights"], probe["coordinate"])
                budget.charge(scalar=5)
                sliced = profile.log_probability(values if probe["coordinate"] == 61 else torch.full_like(values, expected[61]))
                full = torch.tensor(probe["full_log_g"], dtype=torch.float64)
                error, variation = float((full-sliced).abs().max()), float(full.max()-full.min())
                live = probe["coordinate"] == 61 and variation > 1e-10
                if (probe["values"] != values.tolist() or probe["slice_log_g"] != sliced.tolist()
                        or error != probe["maximum_log_error"] or error > 1e-8
                        or variation != probe["log_g_range"] or live != probe["payoff_dependence_detected"]
                        or (probe["coordinate"] == 63 and variation > 1e-10)):
                    raise ValueError("original payoff probe identity failed")
                expected_old_pass = error <= 1e-8 and probe["all11_density_ratios_exactly_invariant"] and (probe["coordinate"] == 61 or variation <= 1e-10)
                if probe["pass"] != expected_old_pass:
                    raise ValueError("original strict-gate bookkeeping changed")
                checks.append({**check, "full_law_scalar_maximum_log_error": error,
                    "payoff_log_range": variation, "original_bitwise_test": probe["all11_density_ratios_exactly_invariant"]})
            before = budget.scalar
            moments = scalar_moments(profile, budget)
            if moments != r["conditional_moments"]:
                raise ValueError("original quadrature changed")
            if c["cell"] == "canonical-h005-k1" and r["role"] == "zero":
                canonical_zero_scalar = budget.scalar-before
            if r["role"] == "zero":
                if budget.physical+11 > 22:
                    raise ValueError("supplementary replay physical cap exceeded")
                rebuilt = context(cache, spec, torch.zeros((1, 64), dtype=torch.float64), parent["family"][c["cell"]]["weights"], budget)
                if rebuilt != {k: v for k, v in r.items() if k != "role"}:
                    raise ValueError("prespecified full-law replay failed")
            rows.append({"cell": c["cell"], "role": r["role"], "checks": checks,
                "both_conditional_moments_relative_resolved": all(x["relative_resolved"] for x in moments["convergence"]),
                "absolute_resolution_only": not all(x["relative_resolved"] for x in moments["convergence"])})
    if canonical_zero_scalar is None or budget.physical != 22:
        raise ValueError("failed-audit accounting reconstruction unavailable")
    # The original audit: toy + zero probes + zero quadrature + full zero replay,
    # then five scalar evaluations before failing at discovery's first probe.
    failed_scalar = toy_scalar+2*canonical_zero_scalar+15
    total_scalar = original["scalar_evaluations"]+failed_scalar+budget.scalar
    if total_scalar > 50000 or original["physical_evaluations"]+11+budget.physical > 100:
        raise ValueError("combined original and supplementary budget exceeded")
    if not all(any(r["cell"] == c and r["checks"][0]["payoff_log_range"] > 1e-10 for r in rows) for c in prepared):
        raise ValueError("cell-wise live residual dependence not detected")
    return {"decision": "POSTHOC_STRUCTURAL_IDENTITY_VERIFIED_ORIGINAL_GATE_INVALID", "posthoc": True,
        "original_decision_preserved": original["summary"]["decision"], "original_result_sha256": config["original_sha256"],
        "original_audit_failure": "ValueError: saved scalar probe/density identity changed",
        "original_failed_audit_accounting": {"physical": 11, "scalar": failed_scalar, "reconstructed_not_original_receipt": True},
        "source": source, "config": config, "contexts": rows, "new_physical": budget.physical,
        "new_scalar": budget.scalar, "combined_physical": 99, "combined_scalar": total_scalar,
        "wall_computation_only": time.perf_counter()-budget.started,
        "q_changed": False, "G_changed": False, "P2_qualified": False, "D3_authorized": False,
        "population_effect_or_global_floor_certified": False}


if __name__ == "__main__":
    receipt = review()
    write_json_atomic_nonoverwriting(ROOT/DIRECTORY/"numerical_review.json", receipt)
    print(json.dumps({k: receipt[k] for k in ("decision", "combined_physical", "combined_scalar", "wall_computation_only")}))
