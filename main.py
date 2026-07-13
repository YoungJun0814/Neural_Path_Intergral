"""
Main entry point for the Neural Path Integral pipeline.

Usage:
    python main.py simulate --config configs/default.yaml
    python main.py calibrate --config configs/default.yaml
    python main.py train_ipm --config configs/default.yaml

Each subcommand reads a YAML config and logs seed, git hash, and timing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# -----------------------------------------------------------------------------
# Subcommands
# -----------------------------------------------------------------------------


def cmd_simulate(cfg: dict) -> int:
    """Run a forward simulation under the specified model."""
    from src.utils import pick_device  # local imports keep --help torch-free

    device = pick_device(cfg.get("device"))
    sim_cfg = cfg["simulate"]
    model_type = sim_cfg.get("model_type", "heston")
    t0 = time.perf_counter()

    if model_type == "rbergomi":
        from src.physics_engine import RBergomiSimulator

        rb = cfg.get("rbergomi", {})
        rb_simulator = RBergomiSimulator(
            H=rb.get("H", 0.07),
            eta=rb.get("eta", 1.9),
            xi=rb.get("xi", 0.235),
            rho=rb.get("rho", -0.9),
            kappa_hybrid=rb.get("kappa_hybrid", 1),
            device=device,
        )
        S, v = rb_simulator.simulate(
            S0=sim_cfg["S0"],
            T=sim_cfg["T"],
            dt=sim_cfg["dt"],
            num_paths=sim_cfg["num_paths"],
            mu=cfg.get("heston", {}).get("mu", 0.0),
        )
    else:
        from src.physics_engine import MarketSimulator

        heston_cfg = cfg["heston"]
        market_simulator = MarketSimulator(
            mu=heston_cfg["mu"],
            kappa=heston_cfg["kappa"],
            theta=heston_cfg["theta"],
            xi=heston_cfg["xi"],
            rho=heston_cfg["rho"],
            jump_lambda=heston_cfg.get("jump_lambda", 0.0),
            jump_mean=heston_cfg.get("jump_mean", 0.0),
            jump_std=heston_cfg.get("jump_std", 0.0),
            vol_jump_mean=heston_cfg.get("vol_jump_mean", 0.0),
            device=device,
        )
        S, v = market_simulator.simulate(
            S0=sim_cfg["S0"],
            v0=sim_cfg["v0"],
            T=sim_cfg["T"],
            dt=sim_cfg["dt"],
            num_paths=sim_cfg["num_paths"],
            model_type=model_type,
        )

    elapsed = time.perf_counter() - t0
    S_T = S[:, -1]
    out = {
        "model": model_type,
        "num_paths": int(sim_cfg["num_paths"]),
        "T": sim_cfg["T"],
        "S_T_mean": float(S_T.mean().item()),
        "S_T_std": float(S_T.std().item()),
        "S_T_min": float(S_T.min().item()),
        "S_T_max": float(S_T.max().item()),
        "elapsed_sec": elapsed,
    }
    print(json.dumps(out, indent=2))
    return 0


def cmd_calibrate(cfg: dict) -> int:
    """Train the NeuralCalibrator on synthetic Heston samples (baseline)."""
    from src.ai_calibrator import NeuralCalibrator
    from src.utils import pick_device

    device = pick_device(cfg.get("device"))
    cal_cfg = cfg.get("calibrate", {})
    model = NeuralCalibrator(
        input_dim=cal_cfg.get("input_dim", 5),
        hidden_dim=cal_cfg.get("hidden_dim", 64),
    ).to(device)
    print(
        f"Built NeuralCalibrator with {sum(p.numel() for p in model.parameters())} params on {device}."
    )
    print("Full training pipeline arrives in Phase 3 (see IMPROVEMENT_PLAN.md M3).")
    return 0


def cmd_train_ipm(cfg: dict, config_path: Path) -> int:
    """Train the NeuralSDESimulator on real return data (distribution matching)."""
    # Late import so `python main.py --help` stays fast.
    from train_driftnet import main as train_main

    print("Delegating to train_driftnet.main() ...")
    train_main(config_path)
    return 0


SUBCMDS = {
    "simulate": cmd_simulate,
    "calibrate": cmd_calibrate,
    "train_ipm": cmd_train_ipm,
}


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="driftnet", description="Neural Path Integral pipeline")
    p.add_argument(
        "mode",
        choices=list(SUBCMDS.keys()),
        help="Subcommand to run.",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed from config.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.config.exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 2

    # Late imports so `python main.py --help` does not require torch/yaml.
    import torch  # noqa: WPS433
    import yaml  # noqa: WPS433

    from src.utils import git_hash, set_seed

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    set_seed(seed)
    print(f"[driftnet] mode={args.mode} seed={seed} git={git_hash()} torch={torch.__version__}")
    # train_ipm needs the config path so train_driftnet can pick up config overrides
    if args.mode == "train_ipm":
        return cmd_train_ipm(cfg, args.config)
    if args.mode == "simulate":
        return cmd_simulate(cfg)
    if args.mode == "calibrate":
        return cmd_calibrate(cfg)
    raise ValueError(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    sys.exit(main())
