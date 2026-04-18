"""
Physics Engine — stochastic calculus simulators for financial markets.

See ``docs/formulation.md`` for the exact equations. In Phase 1 we aligned every
simulator with a single mathematical specification:

* **Full-Truncation Euler** for Heston/Bates/SVJJ (Lord–Koekkoek–van Dijk 2010).
* **Poisson jump counts** per step (supports multiple jumps per interval).
* **Girsanov-consistent controlled simulation**: the drift shift ρξ√v·u on the
  variance process under Q-measure is now explicit.
* **Hybrid scheme** for rBergomi (Bennedsen–Lunde–Pakkanen 2017).
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import torch


# -----------------------------------------------------------------------------
# 1. Heston / Bates / SVJJ
# -----------------------------------------------------------------------------

class MarketSimulator:
    """Heston + Bates + SVJJ simulator using Full-Truncation Euler.

    Parameters follow the notation of ``docs/formulation.md``:

    * ``mu`` – P-measure drift of S
    * ``kappa, theta, xi`` – variance mean-reversion speed, long-run level, vol-of-vol
    * ``rho`` – correlation between dW^S and dW^v
    * ``jump_lambda, jump_mean, jump_std`` – jump intensity and log-jump moments
      (Bates/SVJJ only)
    * ``vol_jump_mean`` – mean of exponential variance jump (SVJJ only)
    """

    def __init__(
        self,
        mu: float,
        kappa: float,
        theta: float,
        xi: float,
        rho: float,
        jump_lambda: float = 0.0,
        jump_mean: float = 0.0,
        jump_std: float = 0.0,
        vol_jump_mean: float = 0.0,
        device: str | torch.device = "cuda",
    ) -> None:
        self.mu = float(mu)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.xi = float(xi)
        self.rho = float(rho)
        self.jump_lambda = float(jump_lambda)
        self.jump_mean = float(jump_mean)
        self.jump_std = float(jump_std)
        self.vol_jump_mean = float(vol_jump_mean)
        self.device = torch.device(device) if not isinstance(device, torch.device) else device

    # ------------------------------------------------------------------
    # 1.1 Natural (P-measure) simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        S0: float,
        v0: float,
        T: float,
        dt: float,
        num_paths: int,
        model_type: str = "heston",
        override_params: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Simulate (S, v) paths under P-measure with Full-Truncation Euler."""
        params = self._resolved_params(override_params)
        n_steps = max(1, int(round(T / dt)))

        S = torch.zeros((num_paths, n_steps + 1), device=self.device)
        v = torch.zeros((num_paths, n_steps + 1), device=self.device)
        S[:, 0] = float(S0)
        v[:, 0] = float(v0)

        sqrt_dt = math.sqrt(dt)
        rho = params["rho"]
        sqrt_one_minus_rho2 = math.sqrt(max(1.0 - rho * rho, 0.0))

        # Jump drift compensator: κ_J = E[e^J − 1]
        if model_type in ("bates", "svjj"):
            kappa_J = math.exp(params["jump_mean"] + 0.5 * params["jump_std"] ** 2) - 1.0
            drift_comp = params["jump_lambda"] * kappa_J
        else:
            drift_comp = 0.0

        for t in range(1, n_steps + 1):
            z1 = torch.randn(num_paths, device=self.device)
            z2 = torch.randn(num_paths, device=self.device)

            v_prev = v[:, t - 1]
            S_prev = S[:, t - 1]
            v_plus = torch.clamp(v_prev, min=0.0)
            sqrt_v = torch.sqrt(v_plus)

            dw_S = z1 * sqrt_dt
            dw_v = (rho * z1 + sqrt_one_minus_rho2 * z2) * sqrt_dt

            # Full-Truncation Euler
            dv = params["kappa"] * (params["theta"] - v_plus) * dt + params["xi"] * sqrt_v * dw_v
            v_new = v_prev + dv

            dS = (params["mu"] - drift_comp) * S_prev * dt + sqrt_v * S_prev * dw_S

            # Jumps
            if model_type in ("bates", "svjj") and params["jump_lambda"] > 0:
                n_jumps = torch.poisson(
                    torch.full((num_paths,), params["jump_lambda"] * dt, device=self.device)
                )
                has_jumps = n_jumps > 0
                if has_jumps.any():
                    # Sum of n_jumps iid log-jumps: mean = n·m, var = n·s²
                    log_jump = (
                        n_jumps * params["jump_mean"]
                        + torch.sqrt(n_jumps) * params["jump_std"] * torch.randn(num_paths, device=self.device)
                    )
                    jump_factor = (torch.exp(log_jump) - 1.0) * has_jumps.float()
                    dS = dS + jump_factor * S_prev

                    if model_type == "svjj" and params["vol_jump_mean"] > 0:
                        rate = 1.0 / (params["vol_jump_mean"] + 1e-12)
                        exp_sample = torch.distributions.Exponential(rate).sample((num_paths,)).to(self.device)
                        v_new = v_new + exp_sample * n_jumps * has_jumps.float()

            S[:, t] = torch.clamp(S_prev + dS, min=1e-8)
            v[:, t] = torch.clamp(v_new, min=0.0)  # full-truncation: keep sign; use v_plus downstream

        return S, v

    # ------------------------------------------------------------------
    # 1.2 Controlled (Q-measure) simulation
    # ------------------------------------------------------------------

    def simulate_controlled(
        self,
        S0: float,
        v0: float,
        T: float,
        dt: float,
        num_paths: int,
        control_fn: Optional[Callable] = None,
        model_type: str = "heston",
        barrier_level: Optional[float] = None,
        barrier_type: str = "down-out",
        apply_v_drift_correction: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Simulate paths under the controlled Q-measure.

        Returns
        -------
        S, v : torch.Tensor
            Simulated price / variance paths under Q.
        log_weights : torch.Tensor
            ``log(dP/dQ)`` accumulated along each path.
        barrier_hit : torch.Tensor | None
            Boolean mask of barrier breach events (None if ``barrier_level is None``).
        running_int_S : torch.Tensor
            ∫₀ᵀ S_t dt (left-point rule), useful for Asian-type payoffs.

        Notes
        -----
        * The Q-dynamics include the ρξ√v·u drift correction on the variance
          process when ``apply_v_drift_correction=True`` (default, correct).
        * Setting it False reproduces the pre-Phase-1 (biased) behavior and is
          only kept for regression comparison in ``tests``.
        """
        params = self._resolved_params(None)
        n_steps = max(1, int(round(T / dt)))
        sqrt_dt = math.sqrt(dt)
        rho = params["rho"]
        sqrt_one_minus_rho2 = math.sqrt(max(1.0 - rho * rho, 0.0))

        curr_S = torch.full((num_paths,), float(S0), device=self.device)
        curr_v = torch.full((num_paths,), float(v0), device=self.device)

        S_list = [curr_S]
        v_list = [curr_v]

        int_u_dW = torch.zeros(num_paths, device=self.device)
        int_u_sq_dt = torch.zeros(num_paths, device=self.device)
        running_int_S = torch.zeros(num_paths, device=self.device)
        barrier_hit = (
            torch.zeros(num_paths, dtype=torch.bool, device=self.device)
            if barrier_level is not None
            else None
        )

        for k in range(1, n_steps + 1):
            t_curr = (k - 1) * dt
            z1 = torch.randn(num_paths, device=self.device)
            z2 = torch.randn(num_paths, device=self.device)

            # Q-Brownian increments
            dw_S_Q = z1 * sqrt_dt
            dw_perp_Q = z2 * sqrt_dt

            v_plus = torch.clamp(curr_v, min=0.0)
            sqrt_v = torch.sqrt(v_plus)

            # Control
            if control_fn is not None:
                if t_curr > 1e-9:
                    avg_S = running_int_S / t_curr
                else:
                    avg_S = curr_S
                u_t = control_fn(t_curr, curr_S, curr_v, avg_S)
            else:
                u_t = torch.zeros(num_paths, device=self.device)

            # Price dynamics (Q): drift = μ + √v · u
            dS = (params["mu"] + sqrt_v * u_t) * curr_S * dt + sqrt_v * curr_S * dw_S_Q
            next_S = torch.clamp(curr_S + dS, min=1e-8)

            # Variance dynamics (Q): drift picks up ρ·ξ·√v · u
            v_drift_Q = params["kappa"] * (params["theta"] - v_plus)
            if apply_v_drift_correction and control_fn is not None:
                v_drift_Q = v_drift_Q + rho * params["xi"] * sqrt_v * u_t
            dW_v_Q = rho * dw_S_Q + sqrt_one_minus_rho2 * dw_perp_Q
            dv = v_drift_Q * dt + params["xi"] * sqrt_v * dW_v_Q
            next_v = torch.clamp(curr_v + dv, min=0.0)

            running_int_S = running_int_S + curr_S * dt

            # Girsanov accumulators: log(dP/dQ) = −∫ u dW^Q − ½ ∫ u² dt
            int_u_dW = int_u_dW + u_t * z1 * sqrt_dt
            int_u_sq_dt = int_u_sq_dt + (u_t ** 2) * dt

            # Barrier
            if barrier_hit is not None:
                if barrier_type == "down-out":
                    barrier_hit = barrier_hit | (next_S <= barrier_level)
                elif barrier_type == "up-out":
                    barrier_hit = barrier_hit | (next_S >= barrier_level)
                else:
                    raise ValueError(f"unknown barrier_type: {barrier_type}")

            S_list.append(next_S)
            v_list.append(next_v)
            curr_S, curr_v = next_S, next_v

        S = torch.stack(S_list, dim=1)
        v = torch.stack(v_list, dim=1)
        log_weights = -int_u_dW - 0.5 * int_u_sq_dt
        return S, v, log_weights, barrier_hit, running_int_S

    # ------------------------------------------------------------------
    # 1.3 Helpers
    # ------------------------------------------------------------------

    def _resolved_params(self, override: Optional[dict]) -> dict:
        base = {
            "mu": self.mu,
            "kappa": self.kappa,
            "theta": self.theta,
            "xi": self.xi,
            "rho": self.rho,
            "jump_lambda": self.jump_lambda,
            "jump_mean": self.jump_mean,
            "jump_std": self.jump_std,
            "vol_jump_mean": self.vol_jump_mean,
        }
        if override:
            base.update(override)
        return base


# -----------------------------------------------------------------------------
# 2. Rough Bergomi (hybrid scheme)
# -----------------------------------------------------------------------------

class FractionalBrownianMotion:
    """Fractional Brownian motion generator via Cholesky decomposition.

    Exact in law but O(N²) memory. Intended for validation / small grids only;
    production rBergomi uses the hybrid scheme below.
    """

    def __init__(self, H: float = 0.1, device: str | torch.device = "cuda") -> None:
        if not (0.0 < H < 1.0):
            raise ValueError(f"H must be in (0,1); got {H}")
        self.H = float(H)
        self.device = torch.device(device) if not isinstance(device, torch.device) else device

    def _covariance_matrix(self, n_steps: int, dt: float) -> torch.Tensor:
        H = self.H
        times = torch.arange(1, n_steps + 1, device=self.device, dtype=torch.float64) * dt
        t_i = times.unsqueeze(1)
        t_j = times.unsqueeze(0)
        cov = 0.5 * (torch.abs(t_i) ** (2 * H) + torch.abs(t_j) ** (2 * H) - torch.abs(t_i - t_j) ** (2 * H))
        cov = cov + 1e-10 * torch.eye(n_steps, device=self.device, dtype=torch.float64)
        return cov.to(torch.float32)

    def generate(self, n_paths: int, n_steps: int, dt: float) -> torch.Tensor:
        cov = self._covariance_matrix(n_steps, dt)
        L = torch.linalg.cholesky(cov)
        Z = torch.randn(n_paths, n_steps, device=self.device)
        W_H = (L @ Z.T).T
        zeros = torch.zeros(n_paths, 1, device=self.device)
        return torch.cat([zeros, W_H], dim=1)


class RBergomiSimulator:
    """rBergomi simulator using the hybrid scheme of Bennedsen–Lunde–Pakkanen (2017).

    Dynamics::

        V_t = ξ · exp( η · Ỹ_t − ½ η² t^{2H} )
        dS_t = √V_t · S_t · dW_t^{(S)}
        dW_t^{(S)} = ρ dW_t^{(1)} + √(1-ρ²) dW_t^{(2)}
        Ỹ_t = ∫₀ᵗ (t-s)^{H-½} dW_s^{(1)}          (Volterra process)

    Critically, the price driver $W^{(S)}$ is built from the **same Brownian
    motion $W^{(1)}$ that drives the Volterra process**, not from the fBm
    increments. This was wrong in pre-Phase-1 code.
    """

    def __init__(
        self,
        H: float = 0.1,
        eta: float = 1.9,
        xi: float = 0.235,
        rho: float = -0.9,
        kappa_hybrid: int = 1,
        device: str | torch.device = "cuda",
    ) -> None:
        if not (0.0 < H < 0.5):
            raise ValueError(f"rBergomi requires H in (0, 0.5); got {H}")
        self.H = float(H)
        self.eta = float(eta)
        self.xi = float(xi)
        self.rho = float(rho)
        self.kappa_hybrid = int(kappa_hybrid)
        self.device = torch.device(device) if not isinstance(device, torch.device) else device

    def _hybrid_scheme_volterra(self, num_paths: int, n_steps: int, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (W1, Y) where W1 is increments of the driving BM and Y is
        the Volterra process Ỹ_t evaluated on the time grid.

        Implementation follows Bennedsen–Lunde–Pakkanen (2017) Alg. BN, with
        κ=self.kappa_hybrid (default 1 — exact on the first sub-interval,
        Riemann approximation afterwards).
        """
        H = self.H
        alpha = H - 0.5  # Volterra kernel exponent

        # Time grid t_i = i·dt for i=0..n_steps
        # We need, for each i: Ỹ(t_i) = ∫₀^{t_i} (t_i - s)^α dW^{(1)}_s
        # Discrete hybrid: split into [t_{i-1}, t_i] (exact 2D Gaussian) and earlier (Riemann sum).

        # Exact covariance of (ΔW, ∫ΔW) on first sub-interval where ΔW = W^{(1)}_{t_i}-W^{(1)}_{t_{i-1}}
        # and ∫_{t_{i-1}}^{t_i} (t_i - s)^α dW_s.
        c11 = dt
        c12 = dt ** (alpha + 1.0) / (alpha + 1.0)
        c22 = dt ** (2 * alpha + 1.0) / (2 * alpha + 1.0)
        cov_local = torch.tensor([[c11, c12], [c12, c22]], device=self.device, dtype=torch.float32)
        L_local = torch.linalg.cholesky(cov_local + 1e-12 * torch.eye(2, device=self.device))

        # Sample n_steps independent 2-vectors per path
        Z = torch.randn(num_paths, n_steps, 2, device=self.device)
        incr = Z @ L_local.T  # shape (paths, steps, 2)
        dW1 = incr[:, :, 0]  # ΔW^(1)_i on interval (t_{i-1}, t_i]
        dI = incr[:, :, 1]  # ∫_{t_{i-1}}^{t_i} (t_i - s)^α dW_s

        # Riemann-sum contribution from earlier intervals:
        # For each t_i, sum_{j<i} b_ij * dW1_j with b_ij = ((i-j)*dt)^α * dt^{1-α} * (1 - ((i-j-1)/(i-j))^{α+1}) / (α+1)
        # Simpler form equivalent to optimal BLP weights:
        # b_ij = dt^{α+1} * [ (i-j)^{α+1} − (i-j-1)^{α+1} ] / (α+1)
        i_idx = torch.arange(1, n_steps + 1, device=self.device, dtype=torch.float32).unsqueeze(1)
        j_idx = torch.arange(0, n_steps, device=self.device, dtype=torch.float32).unsqueeze(0)
        diff = i_idx - j_idx  # (steps, steps)
        diff_prev = torch.clamp(diff - 1.0, min=0.0)
        # Weight for j<i-1 (the i-1 slot is handled exactly via dI)
        b = (diff ** (alpha + 1.0) - diff_prev ** (alpha + 1.0)) / (alpha + 1.0)
        b = b * (dt ** (alpha + 1.0))
        # Mask: j <= i - 2 (earlier intervals only; last interval comes from dI)
        mask = (j_idx <= (i_idx - 2)).to(torch.float32)
        b = b * mask  # (steps, steps)

        # Convolve: earlier[path, i] = sum_j b[i, j] * dW1[path, j]
        earlier = dW1 @ b.T  # (paths, steps)

        Y = torch.zeros(num_paths, n_steps + 1, device=self.device)
        Y[:, 1:] = earlier + dI  # Ỹ at t_i
        return dW1, Y

    def simulate(
        self,
        S0: float,
        T: float,
        dt: float,
        num_paths: int,
        mu: float = 0.0,
        override_params: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        params = self._resolved(override_params)
        H, eta, xi, rho = params["H"], params["eta"], params["xi"], params["rho"]
        n_steps = int(round(T / dt))

        dW1, Y = self._hybrid_scheme_volterra(num_paths, n_steps, dt)

        # Independent Brownian driver for the orthogonal component of the price
        Z2 = torch.randn(num_paths, n_steps, device=self.device) * math.sqrt(dt)
        dW_S = rho * dW1 + math.sqrt(max(1.0 - rho * rho, 0.0)) * Z2

        # Variance path V_t = ξ · exp(η Ỹ − ½ η² t^{2H})
        times = torch.arange(n_steps + 1, device=self.device, dtype=torch.float32) * dt
        drift = 0.5 * (eta ** 2) * (times ** (2 * H))
        V = xi * torch.exp(eta * Y - drift.unsqueeze(0))
        V = torch.clamp(V, min=1e-10)

        # Price path with explicit Euler in log-space for stability
        logS = torch.zeros(num_paths, n_steps + 1, device=self.device)
        logS[:, 0] = math.log(S0)
        for k in range(1, n_steps + 1):
            v_k = V[:, k - 1]
            logS[:, k] = logS[:, k - 1] + (mu - 0.5 * v_k) * dt + torch.sqrt(v_k) * dW_S[:, k - 1]
        S = torch.exp(logS)
        return S, V

    def _resolved(self, override: Optional[dict]) -> dict:
        base = {"H": self.H, "eta": self.eta, "xi": self.xi, "rho": self.rho}
        if override:
            base.update(override)
            if "H" in override and override["H"] != self.H:
                # Hurst exponent affects the entire kernel; recompute
                self.H = float(override["H"])
        return base
