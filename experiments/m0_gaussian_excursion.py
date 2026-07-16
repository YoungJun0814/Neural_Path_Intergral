"""M0 dynamic-programming oracle validation for a Gaussian path event."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import yaml

from src.path_integral import (
    GaussianExcursionSpec,
    build_gaussian_excursion_oracle,
    simulate_gaussian_excursion,
)


def _load(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("expected an M0 schema-version-1 config")
    if payload.get("frozen") is not True:
        raise ValueError("M0 protocol must be frozen")
    return payload


def run(config_path: Path, *, smoke: bool = False) -> dict[str, Any]:
    config = _load(config_path)
    raw_spec, oracle_config, search, validation = (
        config["spec"],
        config["oracle"],
        config["constant_search"],
        config["validation"],
    )
    spec = GaussianExcursionSpec(
        steps=int(raw_spec["steps"]),
        maturity=float(raw_spec["maturity"]),
        hit_barrier=float(raw_spec["hit_barrier"]),
        stress_level=float(raw_spec["stress_level"]),
        minimum_occupation=float(raw_spec["minimum_occupation"]),
    )
    requested_points = list(oracle_config["state_points"][:2] if smoke else oracle_config["state_points"])
    oracles = [
        build_gaussian_excursion_oracle(
            spec,
            state_minimum=float(oracle_config["state_minimum"]),
            state_maximum=float(oracle_config["state_maximum"]),
            state_points=int(points),
            control_bound=float(oracle_config["control_bound"]),
        )
        for points in requested_points
    ]
    references = [oracle.reference_probability for oracle in oracles]
    convergence_differences = [
        abs(right - left) for left, right in zip(references, references[1:], strict=False)
    ]

    search_paths = 10_000 if smoke else int(search["paths"])
    candidates: list[dict[str, float]] = []
    for control in search["controls"]:
        sample = simulate_gaussian_excursion(
            spec,
            num_paths=search_paths,
            seed=int(search["seed"]),
            constant_control=float(control),
        )
        candidates.append(
            {
                "control": float(control),
                "estimate": sample.estimate,
                "second_moment": sample.second_moment,
            }
        )
    selected_control = min(candidates, key=lambda item: item["second_moment"])["control"]

    paths = 20_000 if smoke else int(validation["paths_per_seed"])
    seeds = list(validation["seeds"][:2] if smoke else validation["seeds"])
    runs: list[dict[str, float | int | str]] = []
    for seed in seeds:
        for method, kwargs in (
            ("natural", {}),
            ("constant", {"constant_control": selected_control}),
            ("projected_oracle", {"oracle": oracles[-1]}),
        ):
            sample = simulate_gaussian_excursion(
                spec, num_paths=paths, seed=int(seed), **kwargs
            )
            runs.append(
                {
                    "method": method,
                    "seed": int(seed),
                    "estimate": sample.estimate,
                    "standard_error": sample.standard_error,
                    "second_moment": sample.second_moment,
                    "event_fraction": float(sample.event.mean()),
                    "mean_likelihood": float(sample.likelihood.mean()),
                    "mean_control_energy": sample.mean_control_energy,
                }
            )

    def mean(method: str, key: str) -> float:
        return statistics.mean(
            float(run[key]) for run in runs if run["method"] == method
        )

    natural_estimate = mean("natural", "estimate")
    oracle_estimate = mean("projected_oracle", "estimate")
    combined_se = math.sqrt(
        sum(
            float(run["standard_error"]) ** 2
            for run in runs
            if run["method"] in ("natural", "projected_oracle")
        )
    ) / len(seeds)
    reference_z = (oracle_estimate - references[-1]) / max(combined_se, 1e-300)
    second_moment_vrf = mean("constant", "second_moment") / mean(
        "projected_oracle", "second_moment"
    )
    convergence_pass = len(convergence_differences) == 1 or (
        convergence_differences[-1] < convergence_differences[-2]
    )
    gates = {
        "grid_convergence": convergence_pass,
        "natural_reference_agreement": abs(natural_estimate - references[-1])
        <= 4.0 * mean("natural", "standard_error") + 0.001,
        "oracle_reference_z": abs(reference_z)
        <= float(validation["maximum_reference_z"]),
        "second_moment_below_constant": second_moment_vrf
        >= float(validation["minimum_second_moment_vrf"]),
    }
    return {
        "protocol_id": config["protocol_id"],
        "protocol_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "smoke": smoke,
        "reference_probabilities": references,
        "convergence_differences": convergence_differences,
        "constant_search": candidates,
        "selected_constant_control": selected_control,
        "runs": runs,
        "metrics": {
            "reference_z": reference_z,
            "second_moment_vrf_vs_constant": second_moment_vrf,
        },
        "gates": gates,
        "passed": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/m0_gaussian_excursion.yaml")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    result = run(args.config, smoke=args.smoke)
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(args.output)


if __name__ == "__main__":
    main()
