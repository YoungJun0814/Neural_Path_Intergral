"""Generic Gaussian-mixture theorem oracle for G11 M1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from src.path_integral.gaussian_span_marginalization import (
    GaussianMixtureShiftSpec,
    build_orthonormal_control_span,
    evaluate_marginal_likelihood,
    evaluate_marginalized_function,
    linear_threshold_conditional_probability,
    sample_gaussian_mixture,
)
from src.path_integral.provenance import runtime_provenance, source_provenance
from src.path_integral.seed_ledger import SeedKey, SeedLedger


def _load(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("expected a G11 Gaussian-oracle schema-version-1 config")
    return config, hashlib.sha256(raw).hexdigest()


def _basis(dimension: int, rank: int, generator: torch.Generator) -> torch.Tensor:
    if rank == 0:
        return torch.empty((dimension, 0), dtype=torch.float64)
    matrix = torch.randn((dimension, rank), dtype=torch.float64, generator=generator)
    basis, _upper = torch.linalg.qr(matrix, mode="reduced")
    return basis


def _z_score(values: torch.Tensor, truth: float) -> float:
    standard_error = float(values.std(unbiased=True)) / math.sqrt(values.numel())
    difference = float(values.mean()) - truth
    if standard_error == 0.0:
        return 0.0 if difference == 0.0 else math.copysign(math.inf, difference)
    return difference / standard_error


def run(config_path: Path, *, smoke: bool) -> dict[str, Any]:
    started = time.perf_counter()
    config, digest = _load(config_path)
    ledger = SeedLedger()

    def seed(role: str, case: int, stream: str) -> int:
        return ledger.allocate(
            SeedKey(
                config["protocol_id"],
                role,
                "gaussian_oracle",
                "linear_threshold",
                0,
                case,
                stream,
            )
        )

    factor = config["factorization"]
    cases = 20 if smoke else int(factor["cases"])
    batch = 16 if smoke else int(factor["batch_size"])
    dimensions = [int(value) for value in factor["dimensions"]]
    component_options = [int(value) for value in factor["components"]]
    maximum_rank = int(factor["maximum_rank"])
    maximum_component_error = 0.0
    maximum_mixture_error = 0.0
    maximum_full_bound_violation = 0.0
    maximum_residual_bound_violation = 0.0
    maximum_residual_projection = 0.0
    for case in range(cases):
        generator = torch.Generator().manual_seed(
            seed("factorization", case, "parameters")
        )
        dimension = dimensions[case % len(dimensions)]
        components = component_options[case % len(component_options)]
        rank = case % (min(maximum_rank, dimension) + 1)
        means = 0.9 * torch.randn(
            (components, dimension), dtype=torch.float64, generator=generator
        )
        means[0] = 0.0
        weights = torch.rand(components, dtype=torch.float64, generator=generator) + 0.05
        weights = weights / torch.sum(weights)
        spec = GaussianMixtureShiftSpec(means, weights)
        span = build_orthonormal_control_span(spec, _basis(dimension, rank, generator))
        sample = sample_gaussian_mixture(
            spec,
            batch,
            gaussian_generator=torch.Generator().manual_seed(
                seed("factorization", case, "proposal")
            ),
            label_generator=torch.Generator().manual_seed(
                seed("factorization", case, "labels")
            ),
        )
        density = evaluate_marginal_likelihood(sample.target_coordinates, spec, span)
        maximum_component_error = max(
            maximum_component_error, density.maximum_component_reconstruction_error
        )
        maximum_mixture_error = max(
            maximum_mixture_error, density.maximum_mixture_reconstruction_error
        )
        maximum_full_bound_violation = max(
            maximum_full_bound_violation, density.maximum_full_bound_violation
        )
        maximum_residual_bound_violation = max(
            maximum_residual_bound_violation,
            density.maximum_residual_bound_violation,
        )
        maximum_residual_projection = max(
            maximum_residual_projection,
            density.maximum_sample_residual_projection,
        )

    analytic_config = config["analytic"]
    analytic_cases = 2 if smoke else int(analytic_config["cases"])
    paths = 5_000 if smoke else int(analytic_config["paths_per_case"])
    analytic_outputs: list[dict[str, Any]] = []
    for case in range(analytic_cases):
        dimension = dimensions[case % len(dimensions)]
        components = max(2, component_options[(case + 1) % len(component_options)])
        rank = 1 + case % min(maximum_rank, dimension)
        case_generator = torch.Generator().manual_seed(
            seed("analytic", case, "parameters")
        )
        means = 0.65 * torch.randn(
            (components, dimension), dtype=torch.float64, generator=case_generator
        )
        means[0] = 0.0
        weights = torch.rand(components, dtype=torch.float64, generator=case_generator) + 0.1
        weights = weights / torch.sum(weights)
        spec = GaussianMixtureShiftSpec(means, weights)
        span = build_orthonormal_control_span(
            spec, _basis(dimension, rank, case_generator)
        )
        sample = sample_gaussian_mixture(
            spec,
            paths,
            gaussian_generator=torch.Generator().manual_seed(
                seed("analytic", case, "proposal")
            ),
            label_generator=torch.Generator().manual_seed(
                seed("analytic", case, "labels")
            ),
        )
        density = evaluate_marginal_likelihood(sample.target_coordinates, spec, span)
        event_normal = torch.randn(
            dimension, dtype=torch.float64, generator=case_generator
        )
        event_normal = event_normal / torch.linalg.vector_norm(event_normal)
        threshold = -1.1 + 0.2 * case
        raw_value = (
            sample.target_coordinates @ event_normal <= threshold
        ).to(torch.float64)
        conditional = linear_threshold_conditional_probability(
            density.residual, span, event_normal, threshold
        )
        evaluation = evaluate_marginalized_function(
            density,
            raw_value=raw_value,
            conditional_target_value=conditional,
        )
        truth = float(torch.special.ndtr(torch.tensor(threshold, dtype=torch.float64)))
        paired = evaluation.marginalized_contribution - evaluation.raw_contribution
        outer_truth = 1.0
        output = {
            "case": case,
            "dimension": dimension,
            "components": components,
            "rank": rank,
            "paths": paths,
            "threshold": threshold,
            "truth": truth,
            "raw_mean": float(evaluation.raw_contribution.mean()),
            "marginalized_mean": float(evaluation.marginalized_contribution.mean()),
            "raw_variance": float(evaluation.raw_contribution.var(unbiased=True)),
            "marginalized_variance": float(
                evaluation.marginalized_contribution.var(unbiased=True)
            ),
            "raw_mean_z": _z_score(evaluation.raw_contribution, truth),
            "marginalized_mean_z": _z_score(
                evaluation.marginalized_contribution, truth
            ),
            "paired_mean_z": _z_score(paired, 0.0),
            "outer_normalization_z": _z_score(
                density.residual_likelihood, outer_truth
            ),
            "variance_nonincrease": float(
                evaluation.marginalized_contribution.var(unbiased=True)
            )
            <= float(evaluation.raw_contribution.var(unbiased=True)),
            "maximum_component_reconstruction_error": (
                density.maximum_component_reconstruction_error
            ),
            "maximum_mixture_reconstruction_error": (
                density.maximum_mixture_reconstruction_error
            ),
            "maximum_defensive_bound_violation": max(
                density.maximum_full_bound_violation,
                density.maximum_residual_bound_violation,
            ),
        }
        analytic_outputs.append(output)

    gates_config = config["gates"]
    gates = {
        "factorization_cases": cases
        >= (20 if smoke else int(gates_config["minimum_factorization_cases"])),
        "density_reconstruction": max(
            maximum_component_error, maximum_mixture_error
        )
        <= float(gates_config["maximum_density_reconstruction_error"]),
        "defensive_bound": max(
            maximum_full_bound_violation, maximum_residual_bound_violation
        )
        <= float(gates_config["maximum_defensive_bound_violation"]),
        "analytic_means": all(
            max(abs(case["raw_mean_z"]), abs(case["marginalized_mean_z"]))
            <= float(gates_config["maximum_mean_z"])
            for case in analytic_outputs
        ),
        "paired_means": all(
            abs(case["paired_mean_z"])
            <= float(gates_config["maximum_paired_mean_z"])
            for case in analytic_outputs
        ),
        "outer_normalization": all(
            abs(case["outer_normalization_z"])
            <= float(gates_config["maximum_outer_normalization_z"])
            for case in analytic_outputs
        ),
        "rao_blackwell": all(
            bool(case["variance_nonincrease"]) for case in analytic_outputs
        ),
    }
    elapsed = time.perf_counter() - started
    return {
        "protocol_id": config["protocol_id"],
        "protocol_sha256": digest,
        "seed_ledger_sha256": ledger.sha256,
        "seed_count": len(ledger),
        "smoke": smoke,
        "theory_contract": {
            "target": "finite-dimensional standard Gaussian expectation",
            "proposal": "randomized defensive deterministic-shift mixture",
            "conditional_law": "target Gaussian after residual marginal correction",
            "self_normalized": False,
        },
        "factorization": {
            "cases": cases,
            "batch_size": batch,
            "maximum_component_reconstruction_error": maximum_component_error,
            "maximum_mixture_reconstruction_error": maximum_mixture_error,
            "maximum_full_bound_violation": maximum_full_bound_violation,
            "maximum_residual_bound_violation": maximum_residual_bound_violation,
            "maximum_residual_projection": maximum_residual_projection,
        },
        "analytic_cases": analytic_outputs,
        "gates": gates,
        "passed": all(gates.values()),
        "work_ledger": {
            "warmup_seconds": 0.0,
            "compilation_seconds": 0.0,
            "calibration_seconds": 0.0,
            "pilot_seconds": 0.0,
            "online_seconds": elapsed,
            "audit_seconds": 0.0,
        },
        "environment": runtime_provenance(dtype="torch.float64"),
        **source_provenance(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/g11_gaussian_oracle.yaml")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    result = run(arguments.config, smoke=arguments.smoke)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({"passed": result["passed"], "gates": result["gates"]}, indent=2))


if __name__ == "__main__":
    main()
