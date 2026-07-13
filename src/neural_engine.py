"""
Neural SDE Engine — learned P-measure dynamics + Girsanov-consistent IS control.

See ``docs/formulation.md`` for the exact math.  Key design decisions in Phase 1:

* ``DriftNet`` and ``DiffNet`` approximate the P-measure drift μ(·) and
  diffusion σ(·) of the price process.
* ``VolNet`` keeps the Heston-style mean-reversion prior with a learnable
  *correction*; ``VolNetFree`` drops the prior and lets the network represent
  the entire (a, b) pair.  Both are available.
* ``NeuralSDESimulator.simulate_controlled`` now includes the ρξ√v·u drift
  correction on the variance process (``docs/formulation.md §2.2``).  The
  log-weight is ``log(dP/dQ) = −∫u dW^Q − ½∫u² dt`` with dW^Q the Q-Brownian
  increments that we simulate.
* ``NeuralImportanceSampler.train_step`` implements the variance-minimization
  objective (``docs/formulation.md §3.1``) with proper Girsanov weights.  The
  loss is the second moment of ``g(S_T) · dP/dQ``.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# 1.  Drift / diffusion networks for S
# -----------------------------------------------------------------------------


class DriftNet(nn.Module):
    """μ(S, v, t[, A]) — bounded by tanh to keep gradients tame.

    The 4th input ``A`` is a running average of S (useful for path-dependent
    controls, e.g. Asian options).  When the caller does not supply it we pass
    ``A = S`` as a neutral default.
    """

    def __init__(self, hidden_dim: int = 64, n_layers: int = 3, u_bound: float = 1.0) -> None:
        super().__init__()
        self.u_bound = float(u_bound)
        layers: list[nn.Module] = [nn.Linear(4, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        S: torch.Tensor,
        v: torch.Tensor,
        t: float | torch.Tensor,
        avg_S: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if avg_S is None:
            avg_S = S
        S_n = torch.log(S + 1e-8)
        v_n = torch.log(torch.clamp(v, min=1e-8))
        t_n = (t if torch.is_tensor(t) else torch.tensor(float(t), device=S.device)).expand_as(S)
        A_n = torch.log(avg_S + 1e-8)
        x = torch.stack([S_n, v_n, t_n, A_n], dim=-1)
        raw = self.net(x).squeeze(-1)
        return self.u_bound * torch.tanh(raw)


class DiffNet(nn.Module):
    """σ(S, v, t) — softplus guarantees strict positivity."""

    def __init__(self, hidden_dim: int = 64, n_layers: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(3, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers += [nn.Linear(hidden_dim, 1), nn.Softplus()]
        self.net = nn.Sequential(*layers)

    def forward(self, S: torch.Tensor, v: torch.Tensor, t: float | torch.Tensor) -> torch.Tensor:
        S_n = torch.log(S + 1e-8)
        v_n = torch.log(torch.clamp(v, min=1e-8))
        t_n = (t if torch.is_tensor(t) else torch.tensor(float(t), device=S.device)).expand_as(S)
        x = torch.stack([S_n, v_n, t_n], dim=-1)
        return self.net(x).squeeze(-1)


# -----------------------------------------------------------------------------
# 2.  Variance networks
# -----------------------------------------------------------------------------


class VolNet(nn.Module):
    """Heston-prior volatility dynamics with neural correction.

    a(S,v,t) = κ(θ − v) + ε·ψ_a(S,v,t)
    b(S,v,t) = ψ_b(S,v,t) · √v
    """

    def __init__(
        self, hidden_dim: int = 64, n_layers: int = 3, correction_scale: float = 0.1
    ) -> None:
        super().__init__()
        self.correction_scale = float(correction_scale)

        drift_layers: list[nn.Module] = [nn.Linear(3, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            drift_layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        drift_layers.append(nn.Linear(hidden_dim, 1))
        self.drift_net = nn.Sequential(*drift_layers)

        diff_layers: list[nn.Module] = [nn.Linear(3, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            diff_layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        diff_layers += [nn.Linear(hidden_dim, 1), nn.Softplus()]
        self.diff_net = nn.Sequential(*diff_layers)

        self.kappa = nn.Parameter(torch.tensor(2.0))
        self.theta = nn.Parameter(torch.tensor(0.04))

    def forward(self, S: torch.Tensor, v: torch.Tensor, t: float | torch.Tensor):
        S_n = torch.log(S + 1e-8)
        v_n = torch.log(torch.clamp(v, min=1e-8))
        t_n = (t if torch.is_tensor(t) else torch.tensor(float(t), device=S.device)).expand_as(S)
        x = torch.stack([S_n, v_n, t_n], dim=-1)

        heston_drift = self.kappa * (torch.clamp(self.theta, min=1e-4) - v)
        a = heston_drift + self.correction_scale * self.drift_net(x).squeeze(-1)
        b = self.diff_net(x).squeeze(-1) * torch.sqrt(torch.clamp(v, min=1e-8))
        return a, b


class VolNetFree(nn.Module):
    """Unconstrained variance dynamics — useful as a flexibility benchmark.

    a = ψ_a(S,v,t),   b = ψ_b(S,v,t)   (both fully learned; b > 0 via softplus)

    Because the mean-reversion prior is removed, training is harder and may
    yield negative-drift solutions for small v.  Clip in the simulator.
    """

    def __init__(self, hidden_dim: int = 64, n_layers: int = 3) -> None:
        super().__init__()
        drift_layers: list[nn.Module] = [nn.Linear(3, hidden_dim), nn.SiLU()]
        diff_layers: list[nn.Module] = [nn.Linear(3, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            drift_layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
            diff_layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        drift_layers.append(nn.Linear(hidden_dim, 1))
        diff_layers += [nn.Linear(hidden_dim, 1), nn.Softplus()]
        self.drift_net = nn.Sequential(*drift_layers)
        self.diff_net = nn.Sequential(*diff_layers)

    def forward(self, S: torch.Tensor, v: torch.Tensor, t: float | torch.Tensor):
        S_n = torch.log(S + 1e-8)
        v_n = torch.log(torch.clamp(v, min=1e-8))
        t_n = (t if torch.is_tensor(t) else torch.tensor(float(t), device=S.device)).expand_as(S)
        x = torch.stack([S_n, v_n, t_n], dim=-1)
        return self.drift_net(x).squeeze(-1), self.diff_net(x).squeeze(-1)


# -----------------------------------------------------------------------------
# 3.  Neural SDE simulator
# -----------------------------------------------------------------------------


class NeuralSDESimulator:
    """Stochastic-volatility Neural SDE with learnable correlation ρ.

    Dynamics (P-measure)::

        dS = μ_net(S,v,t) · S · dt + σ_net(S,v,t) · S · dW^S
        dv = a(S,v,t) dt + b(S,v,t) dW^v
        d<W^S, W^v> = ρ dt  (ρ = tanh(_rho_raw))
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 3,
        device: str | torch.device = "cuda",
        rho_init: float = -0.7,
        v0: float = 0.04,
        vol_head: str = "prior",  # "prior" → VolNet, "free" → VolNetFree
    ) -> None:
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.drift_net = DriftNet(hidden_dim, n_layers).to(self.device)
        self.diff_net = DiffNet(hidden_dim, n_layers).to(self.device)
        if vol_head == "prior":
            self.vol_net: nn.Module = VolNet(hidden_dim, n_layers).to(self.device)
        elif vol_head == "free":
            self.vol_net = VolNetFree(hidden_dim, n_layers).to(self.device)
        else:
            raise ValueError(f"vol_head must be 'prior' or 'free', got {vol_head!r}")
        self.vol_head = vol_head

        rho_raw_init = self._inverse_tanh(rho_init)
        self._rho_raw = nn.Parameter(
            torch.tensor(rho_raw_init, device=self.device, dtype=torch.float32)
        )
        self.v0 = float(v0)

    @staticmethod
    def _inverse_tanh(y: float) -> float:
        y = max(-0.999, min(0.999, float(y)))
        return 0.5 * math.log((1.0 + y) / (1.0 - y))

    @property
    def rho(self) -> torch.Tensor:
        return torch.tanh(self._rho_raw)

    def parameters(self):
        yield from self.drift_net.parameters()
        yield from self.diff_net.parameters()
        yield from self.vol_net.parameters()
        yield self._rho_raw

    # ------------------------------------------------------------------
    # 3.1 Uncontrolled (P-measure) simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        S0: float,
        T: float,
        dt: float,
        num_paths: int,
        v0: float | None = None,
        training: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if v0 is None:
            v0 = self.v0
        if T <= 0.0 or dt <= 0.0:
            raise ValueError(f"T and dt must be positive; got T={T}, dt={dt}")
        if num_paths <= 0:
            raise ValueError(f"num_paths must be positive; got {num_paths}")
        if v0 < 0.0:
            raise ValueError(f"v0 must be nonnegative; got {v0}")
        n_steps = max(1, int(math.ceil(T / dt)))
        step_dt = T / n_steps

        curr_S = torch.full((num_paths,), float(S0), device=self.device)
        curr_v = torch.full((num_paths,), float(v0), device=self.device)
        S_list = [curr_S]
        v_list = [curr_v]

        for net in (self.drift_net, self.diff_net, self.vol_net):
            net.train(training)
        ctx = torch.enable_grad() if training else torch.no_grad()

        with ctx:
            sqrt_dt = math.sqrt(step_dt)
            rho = self.rho
            sqrt_one_minus_rho2 = torch.sqrt(torch.clamp(1.0 - rho**2, min=1e-8))

            for k in range(1, n_steps + 1):
                t_curr = (k - 1) * step_dt
                z1 = torch.randn(num_paths, device=self.device)
                z2 = torch.randn(num_paths, device=self.device)
                dW_S = z1 * sqrt_dt
                dW_v = (rho * z1 + sqrt_one_minus_rho2 * z2) * sqrt_dt

                mu = self.drift_net(curr_S, curr_v, t_curr)
                sigma = self.diff_net(curr_S, curr_v, t_curr)
                a, b = self.vol_net(curr_S, curr_v, t_curr)

                dS = mu * curr_S * step_dt + sigma * curr_S * dW_S
                dv = a * step_dt + b * dW_v
                next_S = torch.clamp(curr_S + dS, min=1e-8)
                next_v = torch.clamp(curr_v + dv, min=1e-8)

                S_list.append(next_S)
                v_list.append(next_v)
                curr_S, curr_v = next_S, next_v

        return torch.stack(S_list, dim=1), torch.stack(v_list, dim=1)

    # ------------------------------------------------------------------
    # 3.2 Girsanov-consistent controlled simulation
    # ------------------------------------------------------------------

    def simulate_controlled(
        self,
        S0: float,
        T: float,
        dt: float,
        num_paths: int,
        control_fn: Callable[..., torch.Tensor],
        v0: float | None = None,
        training: bool = True,
        apply_v_drift_correction: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Simulate under Q-measure with the given control and return:

        ``(S, v, log_weight)`` where ``log_weight = log(dP/dQ)`` along each path.

        Notation follows ``docs/formulation.md §2.2``::

            dS = (μ + σ·u) S dt + σ S dW^{Q,S}
            dv = [a + ρ·b·u] dt + b · dW^{Q,v}          ← v-drift correction

        ``b`` plays the role of ``ξ√v`` in the Heston notation; with the VolNet
        convention ``b = ψ_b · √v``, so the correction is ρ · ψ_b · √v · u.
        """
        if v0 is None:
            v0 = self.v0
        if T <= 0.0 or dt <= 0.0:
            raise ValueError(f"T and dt must be positive; got T={T}, dt={dt}")
        if num_paths <= 0:
            raise ValueError(f"num_paths must be positive; got {num_paths}")
        if v0 < 0.0:
            raise ValueError(f"v0 must be nonnegative; got {v0}")
        n_steps = max(1, int(math.ceil(T / dt)))
        step_dt = T / n_steps
        sqrt_dt = math.sqrt(step_dt)

        curr_S = torch.full((num_paths,), float(S0), device=self.device)
        curr_v = torch.full((num_paths,), float(v0), device=self.device)
        S_list = [curr_S]
        v_list = [curr_v]

        int_u_dW = torch.zeros(num_paths, device=self.device)
        int_u_sq_dt = torch.zeros(num_paths, device=self.device)
        running_int_S = torch.zeros(num_paths, device=self.device)

        for net in (self.drift_net, self.diff_net, self.vol_net):
            net.train(training)
        if hasattr(control_fn, "train"):
            control_fn.train(training)  # type: ignore[attr-defined]

        rho = self.rho
        sqrt_one_minus_rho2 = torch.sqrt(torch.clamp(1.0 - rho**2, min=1e-8))

        for k in range(1, n_steps + 1):
            t_curr = (k - 1) * step_dt
            z1 = torch.randn(num_paths, device=self.device)
            z2 = torch.randn(num_paths, device=self.device)
            dw_S_Q = z1 * sqrt_dt
            dw_perp_Q = z2 * sqrt_dt

            # Running average (for path-dependent controls)
            avg_S = running_int_S / t_curr if t_curr > 1e-9 else curr_S

            mu = self.drift_net(curr_S, curr_v, t_curr)
            sigma = self.diff_net(curr_S, curr_v, t_curr)
            a, b = self.vol_net(curr_S, curr_v, t_curr)
            u_t = control_fn(t_curr, curr_S, curr_v, avg_S)

            # Price dynamics under Q
            dS = (mu + sigma * u_t) * curr_S * step_dt + sigma * curr_S * dw_S_Q
            next_S = torch.clamp(curr_S + dS, min=1e-8)

            # Variance dynamics under Q — with correction
            v_drift = a
            if apply_v_drift_correction:
                v_drift = v_drift + rho * b * u_t
            dW_v_Q = rho * dw_S_Q + sqrt_one_minus_rho2 * dw_perp_Q
            dv = v_drift * step_dt + b * dW_v_Q
            next_v = torch.clamp(curr_v + dv, min=1e-8)

            running_int_S = running_int_S + curr_S * step_dt

            # log dP/dQ = -∫ u dW^Q − ½ ∫ u² dt
            int_u_dW = int_u_dW + u_t * z1 * sqrt_dt
            int_u_sq_dt = int_u_sq_dt + (u_t**2) * step_dt

            S_list.append(next_S)
            v_list.append(next_v)
            curr_S, curr_v = next_S, next_v

        S = torch.stack(S_list, dim=1)
        v = torch.stack(v_list, dim=1)
        log_weight = -int_u_dW - 0.5 * int_u_sq_dt
        return S, v, log_weight

    # ------------------------------------------------------------------
    # 3.3 Training helpers
    # ------------------------------------------------------------------

    def train_step(
        self,
        target_prices,
        strikes,
        T: float,
        S0: float,
        r: float,
        optimizer: torch.optim.Optimizer,
        *,
        target_kurtosis: float = 6.0,
        kurtosis_weight: float = 0.1,
        num_paths: int = 2000,
        dt: float | None = None,
    ) -> float:
        """Single training step: MSE(option prices) + λ·(kurtosis − target)²."""
        for net in (self.drift_net, self.diff_net, self.vol_net):
            net.train()
        optimizer.zero_grad()

        dt = float(dt) if dt is not None else min(0.01, T / 3.0)
        S, _v = self.simulate(S0, T, dt, num_paths, training=True)
        S_T = S[:, -1]

        strikes_t = torch.as_tensor(strikes, device=self.device, dtype=torch.float32)
        payoffs = torch.clamp(S_T.unsqueeze(1) - strikes_t.unsqueeze(0), min=0.0)
        model_prices = payoffs.mean(dim=0) * math.exp(-r * T)

        log_returns = torch.log(S[:, 1:] / S[:, :-1]).flatten()
        z = (log_returns - log_returns.mean()) / (log_returns.std() + 1e-8)
        model_kurt = (z**4).mean()
        kurt_loss = (model_kurt - target_kurtosis) ** 2

        target_t = torch.as_tensor(target_prices, device=self.device, dtype=torch.float32)
        price_loss = ((model_prices - target_t) ** 2).mean()

        loss = price_loss + kurtosis_weight * kurt_loss
        loss.backward()
        optimizer.step()
        return float(loss.item())

    # ------------------------------------------------------------------
    # 3.4 Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(
            {
                "drift_net": self.drift_net.state_dict(),
                "diff_net": self.diff_net.state_dict(),
                "vol_net": self.vol_net.state_dict(),
                "vol_head": self.vol_head,
                "_rho_raw": self._rho_raw.data,
                "v0": self.v0,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.drift_net.load_state_dict(ckpt["drift_net"])
        self.diff_net.load_state_dict(ckpt["diff_net"])
        if "vol_net" in ckpt:
            self.vol_net.load_state_dict(ckpt["vol_net"])
        if "_rho_raw" in ckpt:
            self._rho_raw.data = ckpt["_rho_raw"].to(self.device)
        self.v0 = float(ckpt.get("v0", 0.04))


# -----------------------------------------------------------------------------
# 4.  Neural Importance Sampler (variance-minimization IS)
# -----------------------------------------------------------------------------


class NeuralImportanceSampler:
    """Learns a control u(t, S, v, A) that minimizes Var[g(S_T)·dP/dQ].

    The simulator may be either ``NeuralSDESimulator`` (learned P-dynamics) or
    ``src.physics_engine.MarketSimulator`` (analytic P-dynamics).  Both expose
    a ``simulate_controlled`` method that returns ``(S, v, log_weight, …)``.

    The VM loss (``docs/formulation.md §3.1``) is the second moment of the
    re-weighted estimator::

        L_VM(u) = E^Q[ (g(S_T) · dP/dQ)^2 ]

    Minimizing L_VM minimizes the variance of the IS estimator, because
    E^Q[g·dP/dQ] = E^P[g] is independent of u.

    ``train_step`` uses reparameterized pathwise gradients and therefore
    requires a payoff differentiable almost everywhere (vanilla put/call
    payoffs are admissible). A hard event indicator requires the explicit
    score-function treatment in ``src.training.markov_control``; applying this
    helper directly to an indicator is not theoretically justified.
    """

    def __init__(
        self,
        simulator,
        hidden_dim: int = 64,
        n_layers: int = 3,
        u_bound: float = 1.0,
        init_near_zero: bool = True,
    ) -> None:
        self.sim = simulator
        self.device = getattr(simulator, "device", torch.device("cpu"))
        self.control_net = DriftNet(hidden_dim, n_layers, u_bound=u_bound).to(self.device)
        if init_near_zero:
            for p in self.control_net.parameters():
                nn.init.normal_(p, mean=0.0, std=1e-3)

    # ------------------------------------------------------------------
    def get_control_fn(self) -> Callable[..., torch.Tensor]:
        def control_fn(t, S, v, A=None):
            return self.control_net(S, v, t, A)

        return control_fn

    def parameters(self):
        return self.control_net.parameters()

    # ------------------------------------------------------------------
    def train_step(
        self,
        *,
        S0: float,
        T: float,
        dt: float,
        num_paths: int,
        optimizer: torch.optim.Optimizer,
        payoff_fn: Callable[[torch.Tensor], torch.Tensor],
        v0: float | None = None,
        discount: float = 0.0,
        kl_weight: float = 0.0,
    ) -> dict:
        """One VM gradient step.

        Parameters
        ----------
        payoff_fn : callable
            ``payoff_fn(S_T) -> tensor`` of shape ``(num_paths,)``.  For
            barriers, pre-compose with a knock-out mask.
        discount : float
            ``-r·T`` factor built into the payoff if desired; we do *not*
            apply it inside the loss so the optimizer still sees the raw
            second moment.
        kl_weight : float
            Optional KL regularizer ``λ · E^Q[½∫u² dt]`` (``docs/formulation.md §3.2``).
        """
        self.control_net.train()
        optimizer.zero_grad()

        control_fn = self.get_control_fn()
        # Simulator may return 3 or 5 element tuple; accept both.
        out = (
            self.sim.simulate_controlled(
                S0=S0,
                T=T,
                dt=dt,
                num_paths=num_paths,
                control_fn=control_fn,
                v0=v0,
            )
            if self._simulator_accepts_v0()
            else self.sim.simulate_controlled(
                S0=S0,
                T=T,
                dt=dt,
                num_paths=num_paths,
                control_fn=control_fn,
            )
        )
        S = out[0]
        log_weight = out[2]  # position 2 in both NeuralSDE & MarketSimulator returns

        S_T = S[:, -1]
        payoffs = payoff_fn(S_T)
        weights = torch.exp(log_weight)
        reweighted = payoffs * weights * math.exp(discount)

        loss = (reweighted**2).mean()
        if kl_weight > 0.0:
            # E^Q[½ ∫ u² dt] is accumulated in −log_weight as −0.5·∫u² dt − ∫u dW^Q
            # but the second term has mean 0, so KL ≈ −E^Q[log_weight] + E^Q[½ ∫u² dt]
            # (equal in expectation since ∫u dW^Q has zero Q-mean).
            kl = -log_weight.mean()
            loss = loss + kl_weight * kl

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            mean_estimate = reweighted.mean().item()
            ess = (weights.sum() ** 2 / (weights**2).sum()).item()
        return {
            "loss": float(loss.item()),
            "mean_estimate": float(mean_estimate),
            "ess": float(ess),
            "num_paths": int(num_paths),
        }

    def _simulator_accepts_v0(self) -> bool:
        import inspect

        try:
            return "v0" in inspect.signature(self.sim.simulate_controlled).parameters
        except (TypeError, ValueError):
            return False

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save(self.control_net.state_dict(), path)

    def load(self, path: str) -> None:
        self.control_net.load_state_dict(torch.load(path, map_location=self.device))
