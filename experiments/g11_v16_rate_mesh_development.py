"""Deterministic fixed-rank mesh and nested-Galerkin rate-action audit."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import yaml

from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.cameron_martin_basis import build_blp_cameron_martin_basis
from src.path_integral.cameron_martin_modes import (
    ActionMode,
    ActionSolverConfig,
    ModeSearchConfig,
    find_action_modes,
)
from src.path_integral.finite_grid_small_noise import RBergomiFiniteGridRateAction
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.rate_mesh_audit import (
    omitted_mode_gradient_norm,
    pad_channel_coefficients,
)
from src.path_integral.v15_result_audit import file_sha256, git_source_provenance

ROOT = Path(__file__).resolve().parents[1]


def _problem(config: dict, *, steps: int) -> RBergomiBaselineProblem:
    model = config["model"]
    return RBergomiBaselineProblem(
        task_id=f"{config['task_id']}-n{steps}",
        task=TerminalThresholdTask(level=float(config["threshold"])),
        spot=float(model["spot"]),
        maturity=float(model["maturity"]),
        steps=steps,
        hurst=float(model["hurst"]),
        eta=float(model["eta"]),
        xi=float(model["xi"]),
        rho=float(model["rho"]),
    )


def _search_config(config: dict, *, seed: int) -> ModeSearchConfig:
    return ModeSearchConfig(
        methods=("lbfgs", "trust-ncg"),
        random_starts=int(config["random_starts"]),
        random_seed=seed,
        start_scale=float(config["start_scale"]),
        solver=ActionSolverConfig(
            maximum_iterations=int(config["maximum_iterations"]),
            gradient_tolerance=float(config["gradient_tolerance"]),
        ),
    )


def _mode_record(action: RBergomiFiniteGridRateAction, mode: ActionMode) -> dict:
    coefficients = mode.coefficients
    evaluated = action.evaluate(coefficients)
    eigenvalues = mode.hessian_eigenvalues
    return {
        "action": float(evaluated.value),
        "energy": float(evaluated.energy),
        "conditional_cost": float(evaluated.conditional_cost),
        "correlated_return": float(evaluated.correlated_return),
        "integrated_variance": float(evaluated.integrated_variance),
        "gradient_norm": float(mode.gradient_norm),
        "minimum_hessian_eigenvalue": (
            float(torch.min(eigenvalues)) if eigenvalues is not None else None
        ),
        "support_count": int(mode.support_count),
        "methods": list(mode.methods),
        "coefficients": [float(value) for value in coefficients],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root_seed = int(config["seed"])
    fixed_modes = int(config["fixed_modes_per_channel"])

    mesh_records = []
    previous = None
    for index, steps_value in enumerate(config["mesh_steps"]):
        steps = int(steps_value)
        problem = _problem(config, steps=steps)
        basis = build_blp_cameron_martin_basis(
            steps=steps,
            modes_per_driver=fixed_modes,
        )
        action = RBergomiFiniteGridRateAction(problem, basis)
        supplied = () if previous is None else (previous,)
        modes = find_action_modes(
            action,
            basis.rank,
            config=_search_config(config, seed=root_seed + 1009 * index),
            supplied_starts=supplied,
        )
        if not modes.modes:
            raise RuntimeError(f"no converged fixed-rank mode at steps={steps}")
        best = modes.modes[0]
        record = {
            "steps": steps,
            "modes_per_channel": fixed_modes,
            **_mode_record(action, best),
        }
        if previous is not None:
            record["coefficient_change_from_previous_mesh"] = float(
                torch.linalg.vector_norm(best.coefficients - previous)
            )
        mesh_records.append(record)
        previous = best.coefficients.detach().clone()

    for left, right in zip(mesh_records[:-1], mesh_records[1:], strict=True):
        right["adjacent_action_change"] = abs(float(right["action"]) - float(left["action"]))

    finest_steps = int(config["mesh_steps"][-1])
    rank_records = []
    previous_coefficients = None
    previous_modes = None
    for index, modes_value in enumerate(config["rank_modes_per_channel"]):
        modes_per_channel = int(modes_value)
        problem = _problem(config, steps=finest_steps)
        basis = build_blp_cameron_martin_basis(
            steps=finest_steps,
            modes_per_driver=modes_per_channel,
        )
        action = RBergomiFiniteGridRateAction(problem, basis)
        supplied = ()
        if previous_coefficients is not None and previous_modes is not None:
            supplied = (
                pad_channel_coefficients(
                    previous_coefficients,
                    old_modes=previous_modes,
                    new_modes=modes_per_channel,
                ),
            )
        modes = find_action_modes(
            action,
            basis.rank,
            config=_search_config(config, seed=root_seed + 100_003 + 1009 * index),
            supplied_starts=supplied,
        )
        if not modes.modes:
            raise RuntimeError(f"no converged Galerkin mode at rank={basis.rank}")
        best = modes.modes[0]
        record = {
            "steps": finest_steps,
            "modes_per_channel": modes_per_channel,
            **_mode_record(action, best),
        }
        if previous_coefficients is not None and previous_modes is not None:
            padded = pad_channel_coefficients(
                previous_coefficients,
                old_modes=previous_modes,
                new_modes=modes_per_channel,
            )
            point = padded.detach().requires_grad_(True)
            value = action(point)
            (gradient,) = torch.autograd.grad(value, point)
            record["previous_rank_omitted_gradient_norm"] = omitted_mode_gradient_norm(
                gradient.detach(),
                retained_modes=previous_modes,
                expanded_modes=modes_per_channel,
            )
            record["action_decrease_from_previous_rank"] = float(rank_records[-1]["action"]) - float(
                record["action"]
            )
        rank_records.append(record)
        previous_coefficients = best.coefficients.detach().clone()
        previous_modes = modes_per_channel

    mesh_changes = [
        float(record["adjacent_action_change"])
        for record in mesh_records[1:]
        if float(record["adjacent_action_change"]) > 0.0
    ]
    observed_mesh_rate = None
    if len(mesh_changes) >= 2:
        observed_mesh_rate = -math.log2(mesh_changes[-1] / mesh_changes[-2])
    payload = {
        "schema": "npi.g11.v16-rate-mesh-audit.v1",
        "config_binding": {
            "path": config_path.relative_to(ROOT).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "source_provenance": git_source_provenance(ROOT),
        "claim_boundary": {
            "deterministic_galerkin_diagnostic": True,
            "fixed_rank_mesh_theorem_proved": False,
            "rank_to_infinity_theorem_proved": False,
            "blp_to_continuum_transport_theorem_proved": False,
        },
        "coordinate_contract": {
            "local_channels": 2,
            "interpretation": "orthonormal_within_cell_shapes_of_one_volatility_brownian_motion",
            "coefficient_energy_is_cm_energy": True,
        },
        "fixed_rank_mesh": mesh_records,
        "observed_last_adjacent_mesh_rate": observed_mesh_rate,
        "finest_mesh_rank_study": rank_records,
    }
    output = ROOT / str(config["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "mesh": len(mesh_records), "ranks": len(rank_records)}, indent=2))


if __name__ == "__main__":
    main()
