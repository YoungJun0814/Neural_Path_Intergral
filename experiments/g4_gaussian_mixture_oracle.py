"""Gaussian two-tail falsification gate for exact path-integral mixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy.optimize import minimize_scalar

from src.path_integral import (
    gaussian_single_drift_second_moment,
    gaussian_symmetric_mixture_log_q_over_p,
    gaussian_symmetric_mixture_second_moment,
    gaussian_two_tail_probability,
    log_mixture_q_over_p,
)


def _load(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("expected a G4 Gaussian schema-version-1 config")
    return payload


def _z_score(estimate: float, standard_error: float, reference: float) -> float:
    return (estimate - reference) / max(standard_error, 1e-300)


def _simulate(
    *,
    drift: float,
    horizon: float,
    threshold: float,
    paths: int,
    seed: int,
) -> dict[str, float]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    labels = torch.randint(0, 2, (paths,), generator=generator)
    signs = torch.where(labels == 0, 1.0, -1.0).double()
    proposal_terminal = torch.randn(paths, generator=generator, dtype=torch.float64)
    proposal_terminal = proposal_terminal * math.sqrt(horizon)
    target_terminal = proposal_terminal + signs * drift * horizon
    component_log = torch.stack(
        (
            drift * target_terminal - 0.5 * drift * drift * horizon,
            -drift * target_terminal - 0.5 * drift * drift * horizon,
        ),
        dim=-1,
    )
    weights = torch.full((2,), 0.5, dtype=torch.float64)
    log_mixture = log_mixture_q_over_p(component_log, weights)
    selected_log = torch.gather(component_log, 1, labels[:, None]).squeeze(1)
    event = (torch.abs(target_terminal) >= threshold).double()
    balance_likelihood = torch.exp(-log_mixture)
    component_likelihood = torch.exp(-selected_log)
    balance = event * balance_likelihood
    component = event * component_likelihood

    def summarize(values: torch.Tensor, prefix: str) -> dict[str, float]:
        variance = float(values.var(unbiased=True))
        standard_error = math.sqrt(variance / paths)
        return {
            f"{prefix}_estimate": float(values.mean()),
            f"{prefix}_standard_error": standard_error,
            f"{prefix}_second_moment": float(values.square().mean()),
        }

    likelihood_variance = float(balance_likelihood.var(unbiased=True))
    return {
        **summarize(balance, "balance"),
        **summarize(component, "component"),
        "likelihood_mean": float(balance_likelihood.mean()),
        "likelihood_standard_error": math.sqrt(likelihood_variance / paths),
        "event_fraction": float(event.mean()),
        "maximum_log_likelihood": float((-log_mixture).max()),
        "minimum_log_likelihood": float((-log_mixture).min()),
    }


def run(config_path: Path, *, smoke: bool = False) -> dict[str, Any]:
    config = _load(config_path)
    horizon = float(config["model"]["horizon"])
    threshold = float(config["model"]["threshold"])
    bound = float(config["optimization"]["maximum_absolute_drift"])
    tolerance = float(config["optimization"]["scalar_tolerance"])
    probability = gaussian_two_tail_probability(horizon, threshold)

    single = minimize_scalar(
        lambda value: gaussian_single_drift_second_moment(
            value, horizon=horizon, threshold=threshold
        ),
        bounds=(-bound, bound),
        method="bounded",
        options={"xatol": tolerance},
    )
    mixture = minimize_scalar(
        lambda value: gaussian_symmetric_mixture_second_moment(
            value, horizon=horizon, threshold=threshold
        ),
        bounds=(0.0, bound),
        method="bounded",
        options={"xatol": tolerance},
    )
    if not single.success or not mixture.success:
        raise RuntimeError("scalar oracle optimization failed")

    density_grid = torch.linspace(-8.0, 8.0, 4001, dtype=torch.float64)
    drift = float(mixture.x)
    component_log = torch.stack(
        (
            drift * density_grid - 0.5 * drift * drift * horizon,
            -drift * density_grid - 0.5 * drift * drift * horizon,
        ),
        dim=-1,
    )
    primitive_density = log_mixture_q_over_p(
        component_log, torch.full((2,), 0.5, dtype=torch.float64)
    )
    analytic_density = torch.from_numpy(
        np.asarray(
            gaussian_symmetric_mixture_log_q_over_p(
                density_grid.numpy(), drift=drift, horizon=horizon
            )
        )
    )
    maximum_density_error = float(torch.max(torch.abs(primitive_density - analytic_density)))

    efficacy_paths = (
        100_000 if smoke else int(config["simulation"]["efficacy_paths"])
    )
    component_paths = (
        100_000 if smoke else int(config["simulation"]["component_audit_paths"])
    )
    efficacy = _simulate(
        drift=drift,
        horizon=horizon,
        threshold=threshold,
        paths=efficacy_paths,
        seed=int(config["simulation"]["efficacy_seed"]),
    )
    audit_drift = float(config["simulation"]["component_audit_drift"])
    component_audit = _simulate(
        drift=audit_drift,
        horizon=horizon,
        threshold=threshold,
        paths=component_paths,
        seed=int(config["simulation"]["component_audit_seed"]),
    )
    efficacy["balance_bias_z"] = _z_score(
        efficacy["balance_estimate"], efficacy["balance_standard_error"], probability
    )
    efficacy["normalization_z"] = _z_score(
        efficacy["likelihood_mean"], efficacy["likelihood_standard_error"], 1.0
    )
    component_audit["component_bias_z"] = _z_score(
        component_audit["component_estimate"],
        component_audit["component_standard_error"],
        probability,
    )
    component_audit["balance_bias_z"] = _z_score(
        component_audit["balance_estimate"],
        component_audit["balance_standard_error"],
        probability,
    )

    second_moment_ratio = float(single.fun) / float(mixture.fun)
    gates = config["gates"]
    passed = {
        "single_optimum_is_zero": abs(float(single.x))
        <= float(gates["maximum_single_optimum_absolute_drift"]),
        "mixture_second_moment_improves": second_moment_ratio
        >= float(gates["minimum_second_moment_ratio"]),
        "density_matches_analytic": maximum_density_error
        <= float(gates["maximum_density_error"]),
        "balance_estimate_matches": abs(efficacy["balance_bias_z"])
        <= float(gates["maximum_absolute_z"]),
        "mixture_likelihood_normalizes": abs(efficacy["normalization_z"])
        <= float(gates["maximum_absolute_z"]),
        "component_estimate_matches_at_audit_drift": abs(
            component_audit["component_bias_z"]
        )
        <= float(gates["maximum_absolute_z"]),
    }
    return {
        "protocol_id": config["protocol_id"],
        "protocol_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "protocol_frozen": bool(config.get("frozen", False)),
        "smoke": smoke,
        "analytic_probability": probability,
        "single_constant": {
            "optimal_drift": float(single.x),
            "second_moment": float(single.fun),
        },
        "symmetric_mixture": {
            "optimal_absolute_drift": drift,
            "second_moment": float(mixture.fun),
            "second_moment_ratio_vs_single": second_moment_ratio,
            "maximum_density_error": maximum_density_error,
        },
        "efficacy_simulation": efficacy,
        "component_unbiasedness_audit": {
            "drift": audit_drift,
            **component_audit,
        },
        "gates": passed,
        "passed": all(passed.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/g4_gaussian_mixture_oracle.yaml")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    result = run(args.config, smoke=args.smoke)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(args.output)


if __name__ == "__main__":
    main()
