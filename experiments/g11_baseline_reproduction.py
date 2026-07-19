"""Analytic reproduction for crude, IS, mixture, and smoothing baselines."""

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

from src.path_integral import SeedKey, SeedLedger
from src.path_integral.gaussian_span_marginalization import (
    GaussianMixtureShiftSpec,
    build_orthonormal_control_span,
    evaluate_marginal_likelihood,
    evaluate_marginalized_function,
    sample_gaussian_mixture,
)
from src.path_integral.numerical_smoothing_reference import (
    scipy_scaled_normal_cdf,
)
from src.path_integral.provenance import runtime_provenance, source_provenance
from src.path_integral.stable_gaussian import scaled_normal_cdf


def _load(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("expected a baseline-reproduction schema-version-1 config")
    return config, hashlib.sha256(raw).hexdigest()


def _z_score(values: torch.Tensor, truth: float) -> float:
    standard_error = float(torch.std(values, unbiased=True)) / math.sqrt(values.numel())
    difference = float(torch.mean(values)) - truth
    if standard_error == 0.0:
        return 0.0 if difference == 0.0 else math.copysign(math.inf, difference)
    return difference / standard_error


def run(config_path: Path, *, smoke: bool) -> dict[str, Any]:
    started = time.perf_counter()
    config, config_hash = _load(config_path)
    gaussian = config["gaussian"]
    dimension = int(gaussian["dimension"])
    threshold = float(gaussian["threshold"])
    shift = float(gaussian["shift"])
    paths = 10_000 if smoke else int(gaussian["paths"])
    truth = float(torch.special.ndtr(torch.tensor(threshold, dtype=torch.float64)))
    ledger = SeedLedger()
    methods: dict[str, Any] = {}
    specifications = {
        "crude_mc": (
            torch.zeros((1, dimension), dtype=torch.float64),
            torch.ones(1, dtype=torch.float64),
        ),
        "single_shift_is": (
            torch.tensor(
                [[shift] + [0.0] * (dimension - 1)], dtype=torch.float64
            ),
            torch.ones(1, dtype=torch.float64),
        ),
        "defensive_mixture_is": (
            torch.tensor(
                [[0.0] * dimension, [shift] + [0.0] * (dimension - 1)],
                dtype=torch.float64,
            ),
            torch.tensor(
                [
                    float(gaussian["defensive_weight"]),
                    1.0 - float(gaussian["defensive_weight"]),
                ],
                dtype=torch.float64,
            ),
        ),
    }
    for replicate, (name, (means, weights)) in enumerate(specifications.items()):
        spec = GaussianMixtureShiftSpec(means, weights)
        sample = sample_gaussian_mixture(
            spec,
            paths,
            gaussian_generator=torch.Generator().manual_seed(
                ledger.allocate(
                    SeedKey(
                        config["protocol_id"],
                        "baseline",
                        "gaussian",
                        name,
                        0,
                        replicate,
                        "proposal",
                    )
                )
            ),
            label_generator=torch.Generator().manual_seed(
                ledger.allocate(
                    SeedKey(
                        config["protocol_id"],
                        "baseline",
                        "gaussian",
                        name,
                        0,
                        replicate,
                        "labels",
                    )
                )
            ),
        )
        span = build_orthonormal_control_span(
            spec, torch.empty((dimension, 0), dtype=torch.float64)
        )
        density = evaluate_marginal_likelihood(sample.target_coordinates, spec, span)
        event = (sample.target_coordinates[:, 0] <= threshold).to(torch.float64)
        values = event * density.full_likelihood
        methods[name] = {
            "mean": float(torch.mean(values)),
            "variance": float(torch.var(values, unbiased=True)),
            "z_score": _z_score(values, truth),
        }

    natural_spec = GaussianMixtureShiftSpec(
        torch.zeros((1, dimension), dtype=torch.float64),
        torch.ones(1, dtype=torch.float64),
    )
    event_basis = torch.zeros((dimension, 1), dtype=torch.float64)
    event_basis[0, 0] = 1.0
    span = build_orthonormal_control_span(natural_spec, event_basis)
    sample = sample_gaussian_mixture(
        natural_spec,
        paths,
        gaussian_generator=torch.Generator().manual_seed(
            ledger.allocate(
                SeedKey(
                    config["protocol_id"],
                    "baseline",
                    "gaussian",
                    "natural_smoothing",
                    0,
                    0,
                    "proposal",
                )
            )
        ),
        label_generator=torch.Generator().manual_seed(
            ledger.allocate(
                SeedKey(
                    config["protocol_id"],
                    "baseline",
                    "gaussian",
                    "natural_smoothing",
                    0,
                    0,
                    "labels",
                )
            )
        ),
    )
    density = evaluate_marginal_likelihood(sample.target_coordinates, natural_spec, span)
    raw = (sample.target_coordinates[:, 0] <= threshold).to(torch.float64)
    conditional = torch.full_like(raw, truth)
    smoothed = evaluate_marginalized_function(
        density, raw_value=raw, conditional_target_value=conditional
    ).marginalized_contribution
    methods["natural_conditional_smoothing"] = {
        "mean": float(torch.mean(smoothed)),
        "variance": float(torch.var(smoothed, unbiased=True)),
        "z_score": _z_score(smoothed, truth),
    }
    log_scale = torch.linspace(-2.0, 2.0, 101, dtype=torch.float64)
    arguments = torch.linspace(-12.0, 12.0, 101, dtype=torch.float64)
    scipy_error = float(
        torch.max(
            torch.abs(
                scaled_normal_cdf(log_scale, arguments)
                - scipy_scaled_normal_cdf(log_scale, arguments)
            )
        )
    )
    maximum_z = max(abs(item["z_score"]) for item in methods.values())
    gates = {
        "all_analytic_means_within_z_gate": maximum_z
        <= float(config["gates"]["maximum_absolute_z"]),
        "conditional_smoothing_zero_variance": methods[
            "natural_conditional_smoothing"
        ]["variance"]
        <= 1e-30,
        "independent_scipy_specialization": scipy_error
        <= float(config["gates"]["maximum_scipy_specialization_error"]),
    }
    elapsed = time.perf_counter() - started
    return {
        "schema": "npi.g11.baseline-reproduction.v1",
        "protocol_id": config["protocol_id"],
        "config_sha256": config_hash,
        "smoke": smoke,
        "truth": truth,
        "paths": paths,
        "methods": methods,
        "maximum_absolute_z": maximum_z,
        "maximum_scipy_specialization_error": scipy_error,
        "seed_ledger_sha256": ledger.sha256,
        "seed_count": len(ledger),
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
        "--config", type=Path, default=Path("configs/g11_baseline_reproduction.yaml")
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
    print(json.dumps({"passed": result["passed"], **result["gates"]}, sort_keys=True))


if __name__ == "__main__":
    main()
