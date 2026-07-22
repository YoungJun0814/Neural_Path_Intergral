"""Route B theorem-to-code diagnostics on calibrated V6 cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from experiments.g11_v6_reference import _load_manifest
from src.path_integral import (
    SeedKey,
    SeedLedger,
    TimePiecewiseTwoDriverControl,
    barrier_obligation_diagnostics,
    coefficient_moment_diagnostics,
    correction_rate_observation,
    direction_regularity_diagnostics,
    evaluate_rbergomi_dcs_adjacent,
    evaluate_rbergomi_threshold_coupling,
    identify_rate_window,
    rank_one_price_control_span,
    simulate_coupled_rbergomi_mixture,
    slope_lower_tail_diagnostics,
    terminal_slope_inverse_moment_bound,
)
from src.path_integral.provenance import runtime_provenance, source_provenance
from src.physics_engine import RBergomiSimulator

_SCHEMA = "npi.g11.v6-theory-diagnostics.config.v1"


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA:
        raise ValueError("unsupported V6 theory-diagnostics config")
    expected = {
        "schema",
        "protocol_id",
        "phase",
        "frozen",
        "estimand",
        "target_probability",
        "hierarchy",
        "proposal",
        "sampling",
        "analysis",
        "gates",
    }
    if set(payload) != expected:
        raise ValueError("malformed V6 theory-diagnostics config fields")
    if payload["phase"] not in ("development", "qualification"):
        raise ValueError("unsupported theory-diagnostics phase")
    if payload["phase"] == "qualification" and payload["frozen"] is not True:
        raise ValueError("qualification theory diagnostics must be frozen")
    if payload["estimand"] != "fixed_finest_grid":
        raise ValueError("theory diagnostics must retain the finite-grid estimand")
    return payload, hashlib.sha256(raw).hexdigest()


def _selected_cells(manifest, target_probability: float):
    grouped = {}
    for cell in manifest.cells:
        key = (cell.hurst, cell.task)
        distance = abs(math.log(cell.nominal_probability / target_probability))
        previous = grouped.get(key)
        if previous is None or distance < previous[0]:
            grouped[key] = (distance, cell)
    return tuple(value[1] for key, value in sorted(grouped.items()))


def _task(cell):
    from src.path_integral import DiscreteBarrierHitTask, TerminalThresholdTask

    if cell.task == "terminal_left_tail":
        return TerminalThresholdTask(cell.event_threshold)
    return DiscreteBarrierHitTask(cell.event_threshold)


def _direction(sample) -> torch.Tensor:
    schedules = sample.all_expert_controls[0]
    if float(torch.linalg.vector_norm(schedules[:, :, 1])) == 0.0:
        return torch.full(
            (schedules.shape[1],),
            1.0 / math.sqrt(schedules.shape[1]),
            dtype=schedules.dtype,
            device=schedules.device,
        )
    return rank_one_price_control_span(
        schedules, step_dt=sample.paths.fine.step_dt
    ).direction


def run(config_path: Path, manifest_path: Path, *, smoke: bool = False) -> dict[str, Any]:
    config, config_hash = _load_config(config_path)
    manifest = _load_manifest(manifest_path)
    cells = _selected_cells(manifest, float(config["target_probability"]))
    if smoke:
        cells = cells[:2]
    by_hurst = {}
    for cell in cells:
        by_hurst.setdefault(cell.hurst, []).append(cell)
    levels = tuple(int(value) for value in config["hierarchy"]["adjacent_levels"])
    if levels != tuple(range(levels[0], levels[-1] + 1)) or levels[0] < 1:
        raise ValueError("adjacent theory levels must be positive and consecutive")
    if smoke:
        levels = levels[:2]
    replicates = 1 if smoke else int(config["sampling"]["replicates"])
    paths = 128 if smoke else int(config["sampling"]["paths_per_level"])
    analysis = config["analysis"]
    ledger = SeedLedger()
    records = []
    rate_observations = {}
    failures = []
    for hurst, hurst_cells in sorted(by_hurst.items()):
        anchor = hurst_cells[0]
        simulator = RBergomiSimulator(
            H=anchor.hurst,
            eta=anchor.eta,
            xi=anchor.xi,
            rho=anchor.rho,
            device="cpu",
        )
        controls = tuple(
            TimePiecewiseTwoDriverControl(
                tuple(tuple(float(value) for value in segment) for segment in schedule),
                maturity=anchor.maturity,
            )
            for schedule in config["proposal"]["controls"]
        )
        weights = torch.tensor(config["proposal"]["weights"], dtype=torch.float64)
        for replicate in range(replicates):
            for level in levels:
                try:
                    seeds = {
                        stream: ledger.allocate(
                            SeedKey(
                                str(config["protocol_id"]),
                                "theory-diagnostic",
                                f"h{hurst:.3f}",
                                "shared-tasks",
                                level,
                                replicate,
                                stream,
                            )
                        )
                        for stream in ("proposal", "labels")
                    }
                    torch.manual_seed(seeds["proposal"])
                    fine_steps = int(config["hierarchy"]["coarsest_steps"]) * 2**level
                    sample = simulate_coupled_rbergomi_mixture(
                        simulator,
                        controls,
                        weights,
                        spot=anchor.spot,
                        maturity=anchor.maturity,
                        fine_steps=fine_steps,
                        num_paths=paths,
                        dtype=torch.float64,
                        label_generator=torch.Generator().manual_seed(seeds["labels"]),
                        engine=str(config["sampling"]["engine"]),
                    )
                    direction = _direction(sample)
                    direction_summary = direction_regularity_diagnostics(
                        direction,
                        declared_coarse_weights=direction.reshape(-1, 2).sum(dim=1),
                    )
                    for cell in hurst_cells:
                        task = _task(cell)
                        evaluation = evaluate_rbergomi_dcs_adjacent(
                            sample, task=task, rho=simulator.rho
                        )
                        threshold = evaluate_rbergomi_threshold_coupling(
                            evaluation.fine.log_spot_intercept,
                            evaluation.fine.log_spot_slope,
                            evaluation.coarse.log_spot_intercept,
                            evaluation.coarse.log_spot_slope,
                            fine_step_dt=sample.paths.fine.step_dt,
                            coarse_step_dt=sample.paths.coarse.step_dt,
                            task=task,
                            denominator_floor=float(analysis["denominator_floor"]),
                        )
                        if cell.task == "terminal_left_tail":
                            slope_sample = evaluation.fine.log_spot_slope[:, -1]
                            coefficients = coefficient_moment_diagnostics(
                                evaluation.fine.log_spot_intercept[:, -1],
                                evaluation.fine.log_spot_slope[:, -1],
                                evaluation.coarse.log_spot_intercept[:, -1],
                                evaluation.coarse.log_spot_slope[:, -1],
                                mesh_size=sample.paths.fine.step_dt,
                                orders=tuple(float(value) for value in analysis["inverse_orders"]),
                            )
                            barrier = None
                            analytic_inverse_bounds = [
                                asdict(
                                    terminal_slope_inverse_moment_bound(
                                        direction,
                                        step_dt=sample.paths.fine.step_dt,
                                        maturity=cell.maturity,
                                        hurst=cell.hurst,
                                        xi=cell.xi,
                                        eta=cell.eta,
                                        rho=cell.rho,
                                        order=float(order),
                                    )
                                )
                                for order in analysis["inverse_orders"]
                            ]
                        else:
                            finite = threshold.finite_threshold
                            path_index = torch.arange(finite.numel(), device=finite.device)
                            slope_sample = evaluation.fine.log_spot_slope[
                                path_index, threshold.fine_active_index
                            ][finite]
                            coefficients = None
                            analytic_inverse_bounds = None
                            barrier = barrier_obligation_diagnostics(
                                threshold,
                                active_time_cutoff=float(analysis["active_time_fraction"])
                                * cell.maturity,
                            )
                        slope = slope_lower_tail_diagnostics(
                            slope_sample,
                            inverse_orders=tuple(
                                float(value) for value in analysis["inverse_orders"]
                            ),
                            lower_tail_floors=tuple(
                                float(value) for value in analysis["lower_tail_floors"]
                            ),
                        )
                        observation = correction_rate_observation(
                            level=level,
                            replicate=replicate,
                            threshold_difference=evaluation.threshold_difference,
                            raw_correction=evaluation.raw_correction,
                            dcs_correction=evaluation.marginalized_correction,
                            raw_work_units=paths * fine_steps,
                            dcs_work_units=paths * fine_steps * max(1.0, math.log2(fine_steps)),
                        )
                        rate_observations.setdefault(cell.cell_id, []).append(observation)
                        records.append(
                            {
                                "cell_id": cell.cell_id,
                                "level": level,
                                "replicate": replicate,
                                "fine_steps": fine_steps,
                                "fine_step_dt": sample.paths.fine.step_dt,
                                "direction": asdict(direction_summary),
                                "slope_lower_tail": asdict(slope),
                                "terminal_analytic_inverse_moment_bounds": (
                                    analytic_inverse_bounds
                                ),
                                "coefficient_moments": (
                                    None if coefficients is None else asdict(coefficients)
                                ),
                                "barrier_obligations": None if barrier is None else asdict(barrier),
                                "rate_observation": asdict(observation),
                                "pathwise": {
                                    "maximum_good_event_bound_violation": (
                                        threshold.maximum_good_event_bound_violation
                                    ),
                                    "maximum_exact_decomposition_violation": (
                                        threshold.maximum_exact_decomposition_violation
                                    ),
                                    "maximum_coordinate_mismatch": (
                                        evaluation.maximum_coordinate_mismatch
                                    ),
                                    "maximum_path_reconstruction_error": max(
                                        evaluation.fine.maximum_path_reconstruction_error,
                                        evaluation.coarse.maximum_path_reconstruction_error,
                                    ),
                                },
                            }
                        )
                except Exception as error:
                    failures.append(
                        {
                            "hurst": hurst,
                            "level": level,
                            "replicate": replicate,
                            "type": type(error).__name__,
                            "message": str(error),
                        }
                    )
    rate_results = {}
    for cell_id, observations in rate_observations.items():
        if smoke:
            rate_results[cell_id] = {
                "identified": False,
                "reason": "smoke run has insufficient levels and clusters",
            }
        else:
            rate_results[cell_id] = asdict(
                identify_rate_window(
                    observations,
                    bootstrap_repetitions=int(analysis["bootstrap_repetitions"]),
                    bootstrap_seed=17_230_701,
                    minimum_levels=int(analysis["minimum_rate_levels"]),
                    endpoint_slope_margin=float(analysis["endpoint_slope_margin"]),
                    maximum_variance_cv=float(analysis["maximum_variance_cv"]),
                )
            )
    tolerance = float(config["gates"]["pathwise_tolerance"])
    gates = {
        "no_failures": not failures,
        "complete_diagnostic_matrix": len(records)
        == len(cells) * len(levels) * replicates,
        "direction_geometry": all(
            record["direction"]["positive"]
            and record["direction"]["unit_normalized"]
            and record["direction"]["coarse_consistent"]
            for record in records
        ),
        "pathwise_exactness": all(
            max(record["pathwise"].values()) <= tolerance for record in records
        ),
        "all_empirical_inverse_moments_finite": all(
            record["slope_lower_tail"]["finite"] for record in records
        ),
        "terminal_analytic_inverse_moment_bounds_finite": all(
            record["terminal_analytic_inverse_moment_bounds"] is None
            or all(
                math.isfinite(bound["upper_bound"])
                for bound in record["terminal_analytic_inverse_moment_bounds"]
            )
            for record in records
        ),
        "empirical_diagnostics_not_proof": True,
    }
    provenance = source_provenance()
    formal = {
        "frozen_config": bool(config["frozen"]),
        "frozen_manifest": manifest.frozen,
        "clean_source": not bool(provenance["dirty_worktree"]),
        "non_smoke": not smoke,
    }
    return {
        "schema": "npi.g11.v6-theory-diagnostics.v1",
        "protocol_id": config["protocol_id"],
        "config_sha256": config_hash,
        "manifest_sha256": manifest.sha256,
        "smoke": smoke,
        "claim_scope": (
            "terminal inverse-slope moments have an analytic bound; coefficient and "
            "correction-rate outputs remain falsification diagnostics, not a model-rate proof"
        ),
        "records": records,
        "rate_results": rate_results,
        "failures": failures,
        "gates": gates,
        "formal_readiness": formal,
        "diagnostics_qualified": all(gates.values()) and all(formal.values()),
        "seed_ledger": ledger.to_dict(),
        "seed_ledger_sha256": ledger.sha256,
        "environment": runtime_provenance(dtype="torch.float64"),
        **provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/g11_v6/theory_diagnostics_development.yaml"),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    result = run(arguments.config, arguments.manifest, smoke=arguments.smoke)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"qualified": result["diagnostics_qualified"], **result["gates"]}))


if __name__ == "__main__":
    main()
