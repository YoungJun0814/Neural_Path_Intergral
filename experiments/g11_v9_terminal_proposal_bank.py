"""Train the fresh, terminal-only V9 defensive DCS proposal bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from src.path_integral.dcs_proposal_bank import (
    DCSBankCell,
    DCSBankTrainingConfig,
    train_dcs_proposal_bank,
)
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.provenance import runtime_provenance, source_provenance

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "npi.g11.v9-terminal-proposal-bank.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(config, dict) or config.get("schema") != SCHEMA:
        raise ValueError("unexpected V9 terminal proposal-bank config schema")
    return config, hashlib.sha256(raw).hexdigest()


def _validate(config: dict[str, Any]) -> None:
    binding = config.get("claim_contract")
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise ValueError("V9 proposal bank requires one claim-contract binding")
    path = ROOT / str(binding["path"])
    if not path.is_file() or _sha256(path) != str(binding["sha256"]):
        raise ValueError("V9 proposal-bank claim binding mismatch")
    claim = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config["model"] != claim["model"] or config["cells"] != claim["cells"]:
        raise ValueError("V9 proposal-bank model/cells differ from the claim contract")
    if len(config["cells"]) != 12 or any(
        cell["task"] != "terminal_left_tail" for cell in config["cells"]
    ):
        raise ValueError("V9 proposal bank is exactly the 12-cell terminal roster")
    if config.get("current_namespace_outcomes_inspected_before_freeze") is not False:
        raise ValueError("V9 proposal-bank namespace was not outcome-blind at freeze")
    source_commit = str(config["source_parent_commit"])
    current = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()
    if subprocess.run(
        ("git", "merge-base", "--is-ancestor", source_commit, current),
        cwd=ROOT,
        check=False,
    ).returncode:
        raise ValueError("V9 proposal-bank source commit is not an ancestor")
    source_paths = (
        "experiments/g11_v9_terminal_proposal_bank.py",
        "src/path_integral/dcs_proposal_bank.py",
        "src/path_integral/path_functionals.py",
        "src/training/rbergomi_piecewise_cem.py",
    )
    if subprocess.run(
        ("git", "diff", "--quiet", source_commit, "--", *source_paths),
        cwd=ROOT,
        check=False,
    ).returncode or subprocess.check_output(
        ("git", "status", "--porcelain", "--", *source_paths), cwd=ROOT, text=True
    ).strip():
        raise ValueError("V9 proposal-bank implementation differs from its bound source")


def _cells(config: dict[str, Any]) -> tuple[DCSBankCell, ...]:
    return tuple(
        DCSBankCell(
            cell_id=str(cell["cell_id"]),
            task=TerminalThresholdTask(float(cell["threshold"])),
            hurst=float(cell["hurst"]),
        )
        for cell in config["cells"]
    )


def run(config: dict[str, Any], config_sha256: str) -> dict[str, Any]:
    _validate(config)
    training = config["training"]
    bank = train_dcs_proposal_bank(
        _cells(config),
        spot=float(config["model"]["spot"]),
        maturity=float(config["model"]["maturity"]),
        steps=int(config["model"]["steps"]),
        eta=float(config["model"]["eta"]),
        xi=float(config["model"]["xi"]),
        rho=float(config["model"]["rho"]),
        base_seed=int(config["base_seed"]),
        config=DCSBankTrainingConfig(
            segments=int(training["segments"]),
            replicates=int(training["replicates"]),
            paths_per_iteration=int(training["paths_per_iteration"]),
            maximum_iterations=int(training["maximum_iterations"]),
            elite_quantile=float(training["elite_quantile"]),
            smoothing=float(training["smoothing"]),
            minimum_elite_paths=int(training["minimum_elite_paths"]),
            control_bound=float(training["control_bound"]),
            target_level_repetitions=int(training["target_level_repetitions"]),
            minimum_price_driver_magnitude=float(training["minimum_price_driver_magnitude"]),
            initial_control=tuple(
                (float(pair[0]), float(pair[1])) for pair in training["initial_control"]
            ),
            mixture_scales=tuple(float(value) for value in training["mixture_scales"]),
            mixture_weights=tuple(float(value) for value in training["mixture_weights"]),
        ),
    )
    return {
        "schema": "npi.g11.v9-terminal-proposal-bank-result.v1",
        "protocol_id": config["protocol_id"],
        "namespace": config["namespace"],
        "config_sha256": config_sha256,
        **source_provenance(),
        "environment": runtime_provenance(dtype="torch.float64"),
        "entries": [asdict(entry) for entry in bank.entries],
        "entry_count": len(bank.entries),
        "seed_count": len(bank.entries) * int(training["replicates"]),
        "bank_sha256": bank.bank_sha256,
        "total_training_cost": asdict(bank.total_training_cost),
        "amortization_query_counts": list(config["amortization_query_counts"]),
        "decision": {
            "bank_construction_complete": True,
            "development_benchmark_authorized": True,
            "qualification_authorized": False,
            "performance_claim_authorized": False,
            "submission_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    config, digest = load_config(args.config)
    result = run(config, digest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
