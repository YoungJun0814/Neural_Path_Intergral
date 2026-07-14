"""Frozen-law validation for the controlled two-driver rBergomi BLP scheme."""

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
from scipy.stats import chi2

from src.physics_engine import RBergomiSimulator


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("expected a G2 rBergomi schema-version-1 config")
    return payload


def _constant_control(first: float, second: float):
    def control(
        _time: float,
        spot: torch.Tensor,
        _variance: torch.Tensor,
        _volterra: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack(
            (torch.full_like(spot, first), torch.full_like(spot, second)), dim=-1
        )

    return control


def _mean_se(values: torch.Tensor) -> tuple[float, float]:
    return float(values.mean()), float(values.std(unbiased=True) / math.sqrt(values.numel()))


def _difference_z(
    first_mean: float,
    first_se: float,
    second_mean: float,
    second_se: float,
) -> float:
    return (first_mean - second_mean) / max(
        math.sqrt(first_se * first_se + second_se * second_se), 1e-300
    )


def _run_regime(
    *,
    H: float,
    rho: float,
    config: dict[str, Any],
    paths: int,
    seed: int,
    dt: float,
) -> dict[str, Any]:
    model = config["model"]
    payoffs = config["payoffs"]
    proposal = config["proposal"]
    simulator = RBergomiSimulator(
        H=H,
        eta=float(model["eta"]),
        xi=float(model["xi"]),
        rho=rho,
        device="cpu",
    )
    torch.manual_seed(seed)
    start = time.perf_counter()
    natural = simulator.simulate_controlled_two_driver(
        S0=float(model["spot"]),
        T=float(model["maturity"]),
        dt=dt,
        num_paths=paths,
        mu=float(model["mu"]),
        control_fn=None,
        record_augmented=False,
    )
    natural_seconds = time.perf_counter() - start

    control_values = tuple(float(value) for value in proposal["control"])
    torch.manual_seed(seed + 1_000_003)
    start = time.perf_counter()
    controlled = simulator.simulate_controlled_two_driver(
        S0=float(model["spot"]),
        T=float(model["maturity"]),
        dt=dt,
        num_paths=paths,
        mu=float(model["mu"]),
        control_fn=_constant_control(control_values[0], control_values[1]),
        record_augmented=False,
    )
    controlled_seconds = time.perf_counter() - start

    likelihood = torch.exp(controlled.log_likelihood)
    likelihood_mean, likelihood_se = _mean_se(likelihood)
    weighted_spot = likelihood * controlled.spot[:, -1]
    weighted_spot_mean, weighted_spot_se = _mean_se(weighted_spot)
    natural_spot_mean, natural_spot_se = _mean_se(natural.spot[:, -1])

    weighted_variance = likelihood * controlled.variance[:, -1]
    weighted_variance_mean, weighted_variance_se = _mean_se(weighted_variance)
    natural_variance_mean, natural_variance_se = _mean_se(natural.variance[:, -1])

    hard_natural = (natural.spot[:, -1] <= float(payoffs["hard_barrier"])).double()
    hard_controlled = (
        controlled.spot[:, -1] <= float(payoffs["hard_barrier"])
    ).double()
    hard_contribution = hard_controlled * likelihood
    hard_natural_mean, hard_natural_se = _mean_se(hard_natural)
    hard_controlled_mean, hard_controlled_se = _mean_se(hard_contribution)

    soft_natural = torch.sigmoid(
        (float(payoffs["soft_barrier"]) - natural.spot[:, -1])
        / float(payoffs["soft_scale"])
    )
    soft_controlled = torch.sigmoid(
        (float(payoffs["soft_barrier"]) - controlled.spot[:, -1])
        / float(payoffs["soft_scale"])
    )
    soft_contribution = soft_controlled * likelihood
    soft_natural_mean, soft_natural_se = _mean_se(soft_natural)
    soft_controlled_mean, soft_controlled_se = _mean_se(soft_contribution)
    contribution_ess = float(
        soft_contribution.sum().square()
        / soft_contribution.square().sum().clamp_min(1e-300)
    )

    finite = all(
        bool(torch.isfinite(value).all())
        for value in (
            natural.spot,
            natural.variance,
            controlled.spot,
            controlled.variance,
            controlled.log_likelihood,
        )
    )
    return {
        "H": H,
        "rho": rho,
        "dt": dt,
        "paths": paths,
        "seed": seed,
        "finite": finite,
        "likelihood": {
            "mean": likelihood_mean,
            "se": likelihood_se,
            "normalization_z": (likelihood_mean - 1.0) / max(likelihood_se, 1e-300),
        },
        "spot": {
            "natural_mean": natural_spot_mean,
            "natural_se": natural_spot_se,
            "weighted_mean": weighted_spot_mean,
            "weighted_se": weighted_spot_se,
            "weighted_target_z": (weighted_spot_mean - float(model["spot"]))
            / max(weighted_spot_se, 1e-300),
            "natural_target_z": (natural_spot_mean - float(model["spot"]))
            / max(natural_spot_se, 1e-300),
        },
        "forward_variance": {
            "natural_mean": natural_variance_mean,
            "natural_se": natural_variance_se,
            "natural_target_z": (natural_variance_mean - float(model["xi"]))
            / max(natural_variance_se, 1e-300),
            "weighted_mean": weighted_variance_mean,
            "weighted_se": weighted_variance_se,
            "weighted_target_z": (weighted_variance_mean - float(model["xi"]))
            / max(weighted_variance_se, 1e-300),
        },
        "hard_payoff": {
            "natural_mean": hard_natural_mean,
            "natural_se": hard_natural_se,
            "weighted_mean": hard_controlled_mean,
            "weighted_se": hard_controlled_se,
            "difference_z": _difference_z(
                hard_controlled_mean,
                hard_controlled_se,
                hard_natural_mean,
                hard_natural_se,
            ),
        },
        "soft_payoff": {
            "natural_mean": soft_natural_mean,
            "natural_se": soft_natural_se,
            "weighted_mean": soft_controlled_mean,
            "weighted_se": soft_controlled_se,
            "difference_z": _difference_z(
                soft_controlled_mean,
                soft_controlled_se,
                soft_natural_mean,
                soft_natural_se,
            ),
            "contribution_ess": contribution_ess,
            "contribution_ess_fraction": contribution_ess / paths,
        },
        "timing": {
            "natural_seconds": natural_seconds,
            "controlled_seconds": controlled_seconds,
            "controlled_cost_per_path": controlled_seconds / paths,
        },
    }


def run(config_path: Path, *, smoke: bool = False) -> dict[str, Any]:
    config = _load(config_path)
    model = config["model"]
    evaluation = config["evaluation"]
    H_grid = list(model["H_grid"])
    rho_grid = list(model["rho_grid"])
    if smoke:
        H_grid = H_grid[:2]
        rho_grid = rho_grid[:2]
    paths = 2_000 if smoke else int(evaluation["paths_per_law"])
    regime_root = int(config["seeds"]["regime_root"])
    regimes: list[dict[str, Any]] = []
    for H_index, H in enumerate(H_grid):
        for rho_index, rho in enumerate(rho_grid):
            regimes.append(
                _run_regime(
                    H=float(H),
                    rho=float(rho),
                    config=config,
                    paths=paths,
                    seed=regime_root + 100 * H_index + rho_index,
                    dt=float(model["dt"]),
                )
            )

    refinement_paths = 2_000 if smoke else int(evaluation["refinement_paths"])
    refinement_grid = list(evaluation["refinement_dt_grid"])
    if smoke:
        refinement_grid = refinement_grid[:2]
    refinement = [
        _run_regime(
            H=0.10,
            rho=-0.70,
            config=config,
            paths=refinement_paths,
            seed=int(config["seeds"]["refinement_root"]) + index,
            dt=float(candidate_dt),
        )
        for index, candidate_dt in enumerate(refinement_grid)
    ]
    finest_soft = refinement[-1]["soft_payoff"]
    assert isinstance(finest_soft, dict)
    refinement_relative_changes = []
    refinement_z = []
    for candidate in refinement[:-1]:
        soft = candidate["soft_payoff"]
        assert isinstance(soft, dict)
        refinement_relative_changes.append(
            abs(float(soft["weighted_mean"]) - float(finest_soft["weighted_mean"]))
            / max(abs(float(finest_soft["weighted_mean"])), 1e-300)
        )
        refinement_z.append(
            _difference_z(
                float(soft["weighted_mean"]),
                float(soft["weighted_se"]),
                float(finest_soft["weighted_mean"]),
                float(finest_soft["weighted_se"]),
            )
        )

    z_families: dict[str, list[float]] = {
        "likelihood_normalization": [],
        "weighted_spot": [],
        "weighted_forward_variance": [],
        "hard_payoff_difference": [],
        "soft_payoff_difference": [],
    }
    for regime in regimes:
        z_families["likelihood_normalization"].append(
            float(regime["likelihood"]["normalization_z"])
        )
        z_families["weighted_spot"].append(
            float(regime["spot"]["weighted_target_z"])
        )
        z_families["weighted_forward_variance"].append(
            float(regime["forward_variance"]["weighted_target_z"])
        )
        z_families["hard_payoff_difference"].append(
            float(regime["hard_payoff"]["difference_z"])
        )
        z_families["soft_payoff_difference"].append(
            float(regime["soft_payoff"]["difference_z"])
        )
    z_values = [value for family in z_families.values() for value in family]
    family_tests = {
        name: {
            "count": len(values),
            "directional_z": sum(values) / math.sqrt(len(values)),
            "chi_square": sum(value * value for value in values),
            "chi_square_p": float(
                chi2.sf(sum(value * value for value in values), df=len(values))
            ),
            "maximum_absolute_z": max(abs(value) for value in values),
        }
        for name, values in z_families.items()
    }
    maximum_absolute_z = max(abs(value) for value in z_values)
    finite_pass = all(bool(regime["finite"]) for regime in regimes)
    law_z_mode = str(evaluation.get("law_z_mode", "per_regime"))
    if law_z_mode == "per_regime":
        z_pass = maximum_absolute_z <= float(evaluation["maximum_absolute_z"])
    elif law_z_mode == "family_global":
        z_pass = (
            maximum_absolute_z <= float(evaluation["maximum_absolute_z_alarm"])
            and all(
                abs(float(test["directional_z"]))
                <= float(evaluation["maximum_absolute_family_directional_z"])
                and float(test["chi_square_p"])
                >= float(evaluation["minimum_family_chi_square_p"])
                for test in family_tests.values()
            )
        )
    else:
        raise ValueError("law_z_mode must be 'per_regime' or 'family_global'")
    ess_pass = all(
        float(regime["soft_payoff"]["contribution_ess_fraction"])
        >= float(evaluation["minimum_contribution_ess_fraction"])
        for regime in regimes
    )
    refinement_pass = (
        max(refinement_relative_changes, default=0.0)
        <= float(evaluation["maximum_refinement_relative_change"])
        and max((abs(value) for value in refinement_z), default=0.0)
        <= float(evaluation["maximum_absolute_z"])
    )
    gate_pass = finite_pass and z_pass and ess_pass and refinement_pass
    return {
        "protocol_id": config["protocol_id"],
        "protocol_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "protocol_frozen": bool(config.get("frozen", False)),
        "smoke": smoke,
        "regimes": regimes,
        "refinement": {
            "runs": refinement,
            "relative_changes_vs_finest": refinement_relative_changes,
            "z_vs_finest": refinement_z,
        },
        "summary": {
            "law_z_mode": law_z_mode,
            "maximum_absolute_z": maximum_absolute_z,
            "family_tests": family_tests,
            "minimum_contribution_ess_fraction": min(
                float(regime["soft_payoff"]["contribution_ess_fraction"])
                for regime in regimes
            ),
        },
        "gates": {
            "finite": finite_pass,
            "law_z": z_pass,
            "contribution_ess": ess_pass,
            "refinement": refinement_pass,
            "g2_law_pass": gate_pass,
            "confirmatory_pass": bool(config.get("frozen", False)) and not smoke and gate_pass,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/g2_rbergomi_law_development.yaml")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    result = run(args.config, smoke=args.smoke)
    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.output is None:
        print(serialized)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
        print(args.output)


if __name__ == "__main__":
    main()
