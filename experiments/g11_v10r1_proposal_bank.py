"""Train an immutable, clean-source V10R1 full-latent proposal bank."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from src.path_integral.baselines.cem import CEMTrainingConfig
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.provenance import runtime_provenance, source_provenance
from src.path_integral.v10r1_proposal_bank import (
    V10R1BankCell,
    bank_entry_to_dict,
    train_v10r1_proposal_bank,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "npi.g11.v10r1-full-latent-proposal-bank.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(config, dict) or config.get("schema") != SCHEMA:
        raise ValueError("unexpected V10R1 proposal-bank schema")
    return config, hashlib.sha256(raw).hexdigest()


def validate_config(config: dict[str, Any], *, root: Path = ROOT) -> None:
    binding = config.get("claim_contract")
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise ValueError("V10R1 bank requires one exact claim-contract binding")
    claim_path = root / str(binding["path"])
    if not claim_path.is_file() or _sha256(claim_path) != str(binding["sha256"]):
        raise ValueError("V10R1 claim-contract binding mismatch")
    claim = yaml.safe_load(claim_path.read_text(encoding="utf-8"))
    if claim.get("schema") != "npi.g11.v10r1-terminal-claim-contract.v1":
        raise ValueError("unexpected V10R1 claim-contract schema")
    if config.get("model") != claim.get("model") or config.get("cells") != claim.get(
        "cells"
    ):
        raise ValueError("V10R1 bank model/cells differ from the claim contract")
    if int(config.get("replicates", 0)) != int(claim.get("proposal_replicates", 0)):
        raise ValueError("V10R1 bank replicate count differs from the claim contract")
    cells = config.get("cells")
    if not isinstance(cells, list) or len(cells) < 1:
        raise ValueError("V10R1 bank requires a nonempty terminal cell roster")
    if any(cell.get("task") != "terminal_left_tail" for cell in cells):
        raise ValueError("V10R1 bank is terminal-only")
    if config.get("current_namespace_outcomes_inspected_before_freeze") is not False:
        raise ValueError("V10R1 bank namespace was not outcome-blind at freeze")
    if int(config.get("replicates", 0)) < 2:
        raise ValueError("V10R1 requires at least two independent training replicates")
    if config.get("training", {}).get("time_bins", None) is not None:
        raise ValueError("V10R1 CEM must train the complete 3N mean without time binning")


def cells_from_config(config: dict[str, Any]) -> tuple[V10R1BankCell, ...]:
    return tuple(
        V10R1BankCell(
            cell_id=str(cell["cell_id"]),
            task=TerminalThresholdTask(float(cell["threshold"])),
            hurst=float(cell["hurst"]),
        )
        for cell in config["cells"]
    )


def cem_from_config(config: dict[str, Any]) -> CEMTrainingConfig:
    training = config["training"]
    return CEMTrainingConfig(
        iterations=int(training["iterations"]),
        samples_per_iteration=int(training["samples_per_iteration"]),
        elite_fraction=float(training["elite_fraction"]),
        smoothing=float(training["smoothing"]),
        defensive_weight=float(training["defensive_weight"]),
        max_mean_norm=float(training["max_mean_norm"]),
        time_bins=None,
    )


def run(config: dict[str, Any], config_sha256: str, *, root: Path = ROOT) -> dict[str, Any]:
    validate_config(config, root=root)
    provenance = source_provenance()
    if provenance["dirty_worktree"] is not False:
        raise RuntimeError("V10R1 bank execution requires a clean committed source tree")
    model = config["model"]
    bank = train_v10r1_proposal_bank(
        cells_from_config(config),
        replicates=int(config["replicates"]),
        spot=float(model["spot"]),
        maturity=float(model["maturity"]),
        steps=int(model["steps"]),
        eta=float(model["eta"]),
        xi=float(model["xi"]),
        rho=float(model["rho"]),
        base_seed=int(config["base_seed"]),
        cem_config=cem_from_config(config),
    )
    seed_payload = json.dumps(bank.training_seeds, separators=(",", ":")).encode()
    return {
        "schema": "npi.g11.v10r1-full-latent-proposal-bank-result.v1",
        "protocol_id": config["protocol_id"],
        "namespace": config["namespace"],
        "config_path": str(config["config_path"]),
        "config_sha256": config_sha256,
        **provenance,
        "environment": runtime_provenance(dtype="torch.float64"),
        "entries": [bank_entry_to_dict(entry) for entry in bank.entries],
        "entry_count": len(bank.entries),
        "training_seeds": list(bank.training_seeds),
        "seed_set_sha256": hashlib.sha256(seed_payload).hexdigest(),
        "bank_sha256": bank.bank_sha256,
        "total_training_cost": asdict(bank.total_training_cost),
        "decision": {
            "bank_construction_complete": True,
            "bank_audit_required": True,
            "development_authorized": False,
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
        raise FileExistsError("V10R1 proposal-bank outputs are immutable")
    config, digest = load_config(args.config)
    config["config_path"] = args.config.resolve().relative_to(ROOT).as_posix()
    result = run(config, digest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
