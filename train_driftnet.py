"""DriftNet training — distribution matching to real daily returns.

This is the Phase-1 rewrite of the legacy trainer.  Key differences vs. the
pre-Phase-1 script:

1. All hyper-parameters come from ``configs/default.yaml`` (section
   ``train_ipm``).
2. Target kurtosis defaults to the *empirical* value of the loaded returns
   (hard-coded 6.0 removed).  The old behaviour is preserved by setting
   ``train_ipm.target_kurtosis: 6.0`` in the config.
3. Optional MMD loss term (``train_ipm.mmd_weight > 0``) is added next to
   the moment-matching term.
4. Uses ``src.utils.set_seed`` and ``src.utils.pick_device`` so runs are
   reproducible and device-portable (CUDA / MPS / CPU).

Invocation::

    python main.py train_ipm --config configs/default.yaml
    # or directly
    python train_driftnet.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

# Allow running as a script from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.losses.distribution_match import mmd_loss, moment_match_loss, standardized_moments
from src.neural_engine import NeuralSDESimulator
from src.utils import pick_device, set_seed

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


def load_returns(data_path: str) -> np.ndarray:
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        if "Returns" in df.columns:
            return df["Returns"].dropna().values.astype(np.float32)
        if "returns" in df.columns:
            return df["returns"].dropna().values.astype(np.float32)
        # fallback: first numeric column
        col = df.select_dtypes(include=[np.number]).columns[0]
        return df[col].dropna().values.astype(np.float32)
    # Synthetic heavy-tailed returns
    rng = np.random.default_rng(42)
    return (rng.standard_t(df=5, size=5000) * 0.01 + 0.0004).astype(np.float32)


# -----------------------------------------------------------------------------
# Config helpers
# -----------------------------------------------------------------------------


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_target_kurtosis(cfg_value: float | None, real_returns: torch.Tensor) -> torch.Tensor:
    if cfg_value is None:
        _, _, _, kurt = standardized_moments(real_returns)
        return kurt.detach()
    return torch.tensor(float(cfg_value), device=real_returns.device)


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------


def main(config_path: str | Path = "configs/default.yaml") -> None:
    cfg = load_config(Path(config_path))
    train_cfg = cfg.get("train_ipm", {})

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = pick_device(cfg.get("device"))

    data_path = train_cfg.get("data_path", "data/processed/spy_returns.csv")
    real = torch.tensor(load_returns(data_path), dtype=torch.float32, device=device)

    # Targets (all computed from data unless overridden)
    t_mean, t_std, t_skew, t_kurt_data = standardized_moments(real)
    t_kurt = resolve_target_kurtosis(train_cfg.get("target_kurtosis"), real)

    hidden_dim = int(train_cfg.get("hidden_dim", 64))
    n_layers = int(train_cfg.get("n_layers", 3))
    lr = float(train_cfg.get("learning_rate", 1e-4))
    epochs = int(train_cfg.get("epochs", 200))
    batch_paths = int(train_cfg.get("batch_size_paths", 1000))
    dt = float(train_cfg.get("dt", 1.0 / 252.0))
    T_horizon = float(train_cfg.get("T_horizon", 0.5))
    S0 = float(train_cfg.get("S0", 100.0))

    w_mean = float(train_cfg.get("w_mean", 50.0))
    w_std = float(train_cfg.get("w_std", 100.0))
    w_skew = float(train_cfg.get("w_skew", 0.5))
    w_kurt = float(train_cfg.get("w_kurt", 0.01))
    mmd_weight = float(train_cfg.get("mmd_weight", 0.0))

    vol_head = str(train_cfg.get("vol_head", "prior"))

    print("=" * 64)
    print("DriftNet training (distribution matching)")
    print("=" * 64)
    print(f"device={device}  seed={seed}  vol_head={vol_head}")
    print(
        f"Target moments (from data): mean={t_mean.item():.6f} "
        f"std={t_std.item():.6f} skew={t_skew.item():.4f} kurt_data={t_kurt_data.item():.3f}"
    )
    print(f"Effective target kurt: {t_kurt.item():.3f} (None → data)")

    simulator = NeuralSDESimulator(
        hidden_dim=hidden_dim, n_layers=n_layers, device=device, vol_head=vol_head
    )
    optimizer = torch.optim.Adam(simulator.parameters(), lr=lr)

    save_path = Path(train_cfg.get("model_save_path", "checkpoints/driftnet_phase1.pth"))
    save_path.parent.mkdir(parents=True, exist_ok=True)

    loss_history: list[float] = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        S, _v = simulator.simulate(S0=S0, T=T_horizon, dt=dt, num_paths=batch_paths, training=True)
        log_returns = torch.log(S[:, 1:] / S[:, :-1]).flatten()

        loss_mm = moment_match_loss(
            log_returns,
            real,
            weights=(w_mean, w_std, w_skew, w_kurt),
            targets=(t_mean.item(), t_std.item(), t_skew.item(), float(t_kurt.item())),
        )
        loss = loss_mm
        if mmd_weight > 0.0:
            # Sub-sample for O(n²) tractability
            idx_m = torch.randperm(log_returns.numel(), device=device)[:512]
            idx_r = torch.randperm(real.numel(), device=device)[:512]
            loss = loss + mmd_weight * mmd_loss(log_returns[idx_m], real[idx_r])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(simulator.parameters()), 1.0)
        optimizer.step()

        loss_history.append(float(loss.item()))
        if epoch % max(1, epochs // 20) == 0 or epoch == 1:
            with torch.no_grad():
                m, s, sk, k = standardized_moments(log_returns.detach())
            print(
                f"Epoch {epoch:4d}/{epochs} | loss={loss.item():.5f} "
                f"mean={m.item():.5f} std={s.item():.5f} "
                f"skew={sk.item():.3f} kurt={k.item():.3f} "
                f"vol_ann={s.item() * math.sqrt(1.0 / dt) * 100:.1f}%"
            )

    simulator.save(str(save_path))
    print(f"Saved model → {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DriftNet distribution-matching trainer.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()
    main(args.config)
