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
from collections.abc import Callable
from dataclasses import dataclass

import torch

# -----------------------------------------------------------------------------
# 1. Heston / Bates / SVJJ
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class TwoDriverHestonPaths:
    """Output of the independent-basis two-driver Heston simulator.

    Brownian histories are optional because retaining three
    ``(paths, steps, 2)`` tensors is useful for training and reconstruction but
    too expensive for large frozen-control evaluations.
    """

    spot: torch.Tensor
    variance: torch.Tensor
    log_likelihood: torch.Tensor
    control_energy: torch.Tensor
    running_spot_integral: torch.Tensor
    barrier_hit: torch.Tensor | None
    step_dt: float
    proposal_brownian_increments: torch.Tensor | None
    target_brownian_increments: torch.Tensor | None
    controls: torch.Tensor | None


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
        override_params: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Simulate (S, v) paths under P-measure with Full-Truncation Euler."""
        if T <= 0.0 or dt <= 0.0:
            raise ValueError(f"T and dt must be positive; got T={T}, dt={dt}")
        if num_paths <= 0:
            raise ValueError(f"num_paths must be positive; got {num_paths}")
        if v0 < 0.0:
            raise ValueError(f"v0 must be nonnegative; got {v0}")
        if model_type not in ("heston", "bates", "svjj"):
            raise ValueError(f"unknown model_type: {model_type}")

        params = self._resolved_params(override_params)
        n_steps = max(1, int(math.ceil(T / dt)))
        step_dt = T / n_steps

        S = torch.zeros((num_paths, n_steps + 1), device=self.device)
        # Keep the raw Euler state for the actual full-truncation recursion and
        # expose its nonnegative effective value to callers.
        v_state = torch.zeros((num_paths, n_steps + 1), device=self.device)
        v = torch.zeros((num_paths, n_steps + 1), device=self.device)
        S[:, 0] = float(S0)
        v_state[:, 0] = float(v0)
        v[:, 0] = float(v0)

        sqrt_dt = math.sqrt(step_dt)
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

            v_prev = v_state[:, t - 1]
            S_prev = S[:, t - 1]
            v_plus = torch.clamp(v_prev, min=0.0)
            sqrt_v = torch.sqrt(v_plus)

            dw_S = z1 * sqrt_dt
            dw_v = (rho * z1 + sqrt_one_minus_rho2 * z2) * sqrt_dt

            # Full-Truncation Euler
            dv = (
                params["kappa"] * (params["theta"] - v_plus) * step_dt
                + params["xi"] * sqrt_v * dw_v
            )
            v_new = v_prev + dv

            # Conditional log-Euler is strictly positive and avoids artificial
            # left-tail mass from additive Euler followed by a hard floor.
            next_S = S_prev * torch.exp(
                (params["mu"] - drift_comp - 0.5 * v_plus) * step_dt + sqrt_v * dw_S
            )

            # Jumps
            if model_type in ("bates", "svjj") and params["jump_lambda"] > 0:
                n_jumps = torch.poisson(
                    torch.full(
                        (num_paths,),
                        params["jump_lambda"] * step_dt,
                        device=self.device,
                    )
                )
                has_jumps = n_jumps > 0
                if has_jumps.any():
                    # Sum of n_jumps iid log-jumps: mean = n·m, var = n·s²
                    log_jump = n_jumps * params["jump_mean"] + torch.sqrt(n_jumps) * params[
                        "jump_std"
                    ] * torch.randn(num_paths, device=self.device)
                    next_S = next_S * torch.exp(log_jump * has_jumps.float())

                    if model_type == "svjj" and params["vol_jump_mean"] > 0:
                        rate = 1.0 / (params["vol_jump_mean"] + 1e-12)
                        # A sum of n iid exponential jumps is Gamma(n, rate),
                        # not n times one exponential draw.
                        counts = n_jumps[has_jumps]
                        variance_jump = torch.distributions.Gamma(
                            concentration=counts, rate=rate
                        ).sample()
                        v_new = v_new.clone()
                        v_new[has_jumps] = v_new[has_jumps] + variance_jump

            S[:, t] = next_S
            v_state[:, t] = v_new
            v[:, t] = torch.clamp(v_new, min=0.0)

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
        control_fn: Callable | None = None,
        model_type: str = "heston",
        barrier_level: float | None = None,
        barrier_type: str = "down-out",
        apply_v_drift_correction: bool = True,
        brownian_observer: Callable[[float, torch.Tensor], None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
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
        if T <= 0.0 or dt <= 0.0:
            raise ValueError(f"T and dt must be positive; got T={T}, dt={dt}")
        if num_paths <= 0:
            raise ValueError(f"num_paths must be positive; got {num_paths}")
        if v0 < 0.0:
            raise ValueError(f"v0 must be nonnegative; got {v0}")
        if model_type not in ("heston", "bates", "svjj"):
            raise ValueError(f"unknown model_type: {model_type}")
        n_steps = max(1, int(math.ceil(T / dt)))
        step_dt = T / n_steps
        sqrt_dt = math.sqrt(step_dt)
        rho = params["rho"]
        sqrt_one_minus_rho2 = math.sqrt(max(1.0 - rho * rho, 0.0))

        curr_S = torch.full((num_paths,), float(S0), device=self.device)
        curr_v_state = torch.full((num_paths,), float(v0), device=self.device)
        curr_v = torch.clamp(curr_v_state, min=0.0)

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

        if model_type in ("bates", "svjj"):
            kappa_J = math.exp(params["jump_mean"] + 0.5 * params["jump_std"] ** 2) - 1.0
            drift_comp = params["jump_lambda"] * kappa_J
        else:
            drift_comp = 0.0

        for k in range(1, n_steps + 1):
            t_curr = (k - 1) * step_dt
            z1 = torch.randn(num_paths, device=self.device)
            z2 = torch.randn(num_paths, device=self.device)

            # Q-Brownian increments
            dw_S_Q = z1 * sqrt_dt
            dw_perp_Q = z2 * sqrt_dt
            if brownian_observer is not None:
                brownian_observer(t_curr, dw_S_Q)

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

            # Price dynamics (Q), discretized in log space. The Girsanov drift
            # shift is √v·u and the Ito correction is −v/2.
            next_S = curr_S * torch.exp(
                (params["mu"] - drift_comp + sqrt_v * u_t - 0.5 * v_plus) * step_dt
                + sqrt_v * dw_S_Q
            )

            # Variance dynamics (Q): drift picks up ρ·ξ·√v · u
            v_drift_Q = params["kappa"] * (params["theta"] - v_plus)
            if apply_v_drift_correction and control_fn is not None:
                v_drift_Q = v_drift_Q + rho * params["xi"] * sqrt_v * u_t
            dW_v_Q = rho * dw_S_Q + sqrt_one_minus_rho2 * dw_perp_Q
            dv = v_drift_Q * step_dt + params["xi"] * sqrt_v * dW_v_Q
            next_v_state = curr_v_state + dv

            # Brownian control leaves the independent jump law unchanged, so
            # no jump term enters the likelihood ratio. The proposal must,
            # however, simulate the same compensated jump component as the
            # base Bates/SVJJ model.
            if model_type in ("bates", "svjj") and params["jump_lambda"] > 0:
                n_jumps = torch.poisson(
                    torch.full(
                        (num_paths,),
                        params["jump_lambda"] * step_dt,
                        device=self.device,
                    )
                )
                has_jumps = n_jumps > 0
                if has_jumps.any():
                    log_jump = n_jumps * params["jump_mean"] + torch.sqrt(n_jumps) * params[
                        "jump_std"
                    ] * torch.randn(num_paths, device=self.device)
                    next_S = next_S * torch.exp(log_jump * has_jumps.float())

                    if model_type == "svjj" and params["vol_jump_mean"] > 0:
                        rate = 1.0 / (params["vol_jump_mean"] + 1e-12)
                        counts = n_jumps[has_jumps]
                        variance_jump = torch.distributions.Gamma(
                            concentration=counts, rate=rate
                        ).sample()
                        next_v_state = next_v_state.clone()
                        next_v_state[has_jumps] = next_v_state[has_jumps] + variance_jump

            next_v = torch.clamp(next_v_state, min=0.0)

            running_int_S = running_int_S + curr_S * step_dt

            # Girsanov accumulators: log(dP/dQ) = −∫ u dW^Q − ½ ∫ u² dt
            int_u_dW = int_u_dW + u_t * z1 * sqrt_dt
            int_u_sq_dt = int_u_sq_dt + (u_t**2) * step_dt

            # Barrier
            if barrier_hit is not None:
                assert barrier_level is not None
                if barrier_type == "down-out":
                    barrier_hit = barrier_hit | (next_S <= barrier_level)
                elif barrier_type == "up-out":
                    barrier_hit = barrier_hit | (next_S >= barrier_level)
                else:
                    raise ValueError(f"unknown barrier_type: {barrier_type}")

            S_list.append(next_S)
            v_list.append(next_v)
            curr_S, curr_v_state, curr_v = next_S, next_v_state, next_v

        S = torch.stack(S_list, dim=1)
        v = torch.stack(v_list, dim=1)
        log_weights = -int_u_dW - 0.5 * int_u_sq_dt
        return S, v, log_weights, barrier_hit, running_int_S

    def simulate_controlled_two_driver(
        self,
        S0: float,
        v0: float,
        T: float,
        dt: float,
        num_paths: int,
        control_fn: Callable[
            [float, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
        ]
        | None = None,
        barrier_level: float | None = None,
        barrier_type: str = "down-out",
        record_brownian: bool = False,
        dtype: torch.dtype | None = None,
    ) -> TwoDriverHestonPaths:
        r"""Simulate controlled Heston paths in an independent Brownian basis.

        The target and proposal coordinates satisfy

        ``dB^M_i = dB^Q_i + u_i dt``, for ``i=1,2``.

        ``control_fn`` must return shape ``(num_paths, 2)``.  The first control
        shifts the spot Brownian coordinate; both coordinates shift variance
        through ``rho*u_1 + sqrt(1-rho^2)*u_2``.  The returned likelihood is
        always ``log(dM/dQ)`` and is accumulated in float64.

        Set ``record_brownian=True`` only for training or verification batches.
        Controls are evaluated from the current state before the matching
        Brownian increments are sampled, enforcing the simulator-side causal
        ordering required by path-integral feedback control.
        """
        if not math.isfinite(S0) or S0 <= 0.0:
            raise ValueError(f"S0 must be finite and positive; got {S0}")
        if not math.isfinite(T) or not math.isfinite(dt) or T <= 0.0 or dt <= 0.0:
            raise ValueError(f"T and dt must be finite and positive; got T={T}, dt={dt}")
        if num_paths <= 0:
            raise ValueError(f"num_paths must be positive; got {num_paths}")
        if not math.isfinite(v0) or v0 < 0.0:
            raise ValueError(f"v0 must be finite and nonnegative; got {v0}")
        if barrier_level is not None and not math.isfinite(barrier_level):
            raise ValueError("barrier_level must be finite when provided")
        if barrier_type not in ("down-out", "up-out"):
            raise ValueError(f"unknown barrier_type: {barrier_type}")

        params = self._resolved_params(None)
        rho = float(params["rho"])
        if not math.isfinite(rho) or not -1.0 <= rho <= 1.0:
            raise ValueError(f"rho must lie in [-1, 1]; got {rho}")
        sqrt_one_minus_rho2 = math.sqrt(max(1.0 - rho * rho, 0.0))
        simulation_dtype = dtype if dtype is not None else torch.get_default_dtype()
        if not torch.empty((), dtype=simulation_dtype).is_floating_point():
            raise TypeError("dtype must be a floating-point torch dtype")

        n_steps = max(1, int(math.ceil(T / dt)))
        step_dt = T / n_steps
        sqrt_dt = math.sqrt(step_dt)

        current_spot = torch.full(
            (num_paths,), float(S0), device=self.device, dtype=simulation_dtype
        )
        current_variance_state = torch.full(
            (num_paths,), float(v0), device=self.device, dtype=simulation_dtype
        )
        current_variance = torch.clamp(current_variance_state, min=0.0)
        spot_history = [current_spot]
        variance_history = [current_variance]

        stochastic_log_term = torch.zeros(num_paths, device=self.device, dtype=torch.float64)
        control_energy = torch.zeros(num_paths, device=self.device, dtype=torch.float64)
        running_spot_integral = torch.zeros(
            num_paths, device=self.device, dtype=simulation_dtype
        )
        barrier_hit = (
            torch.zeros(num_paths, dtype=torch.bool, device=self.device)
            if barrier_level is not None
            else None
        )

        proposal_history: list[torch.Tensor] | None = [] if record_brownian else None
        target_history: list[torch.Tensor] | None = [] if record_brownian else None
        control_history: list[torch.Tensor] | None = [] if record_brownian else None

        for step in range(n_steps):
            time = step * step_dt
            if time > 1e-9:
                running_average = running_spot_integral / time
            else:
                running_average = current_spot

            if control_fn is None:
                applied_control = torch.zeros(
                    (num_paths, 2), device=self.device, dtype=simulation_dtype
                )
            else:
                applied_control = control_fn(
                    time, current_spot, current_variance, running_average
                )
                if not isinstance(applied_control, torch.Tensor):
                    raise TypeError("two-driver control_fn must return a torch.Tensor")
                if applied_control.shape != (num_paths, 2):
                    raise ValueError(
                        "two-driver control_fn must return shape (num_paths, 2); "
                        f"got {tuple(applied_control.shape)}"
                    )
                if applied_control.device != self.device:
                    raise ValueError("two-driver control output must be on the simulator device")
                if applied_control.dtype != simulation_dtype:
                    raise ValueError("two-driver control output must match the simulation dtype")
                if not torch.isfinite(applied_control).all():
                    raise ValueError("two-driver control output must be finite")

            # Causal ordering: sample increments only after evaluating u(t, X_t).
            proposal_brownian_1 = (
                torch.randn(num_paths, device=self.device, dtype=simulation_dtype) * sqrt_dt
            )
            proposal_brownian_2 = (
                torch.randn(num_paths, device=self.device, dtype=simulation_dtype) * sqrt_dt
            )
            proposal_increment = torch.stack(
                (proposal_brownian_1, proposal_brownian_2), dim=-1
            )
            control_1 = applied_control[:, 0]
            control_2 = applied_control[:, 1]

            variance_plus = torch.clamp(current_variance, min=0.0)
            sqrt_variance = torch.sqrt(variance_plus)
            next_spot = current_spot * torch.exp(
                (
                    params["mu"]
                    + sqrt_variance * control_1
                    - 0.5 * variance_plus
                )
                * step_dt
                + sqrt_variance * proposal_brownian_1
            )

            variance_control = rho * control_1 + sqrt_one_minus_rho2 * control_2
            variance_brownian = (
                rho * proposal_brownian_1
                + sqrt_one_minus_rho2 * proposal_brownian_2
            )
            variance_drift = params["kappa"] * (params["theta"] - variance_plus)
            variance_drift = variance_drift + params["xi"] * sqrt_variance * variance_control
            next_variance_state = (
                current_variance_state
                + variance_drift * step_dt
                + params["xi"] * sqrt_variance * variance_brownian
            )
            next_variance = torch.clamp(next_variance_state, min=0.0)

            applied_control_64 = applied_control.to(torch.float64)
            proposal_increment_64 = proposal_increment.to(torch.float64)
            stochastic_log_term = stochastic_log_term + torch.sum(
                applied_control_64 * proposal_increment_64, dim=-1
            )
            control_energy = control_energy + step_dt * torch.sum(
                applied_control_64.square(), dim=-1
            )

            if proposal_history is not None:
                assert target_history is not None and control_history is not None
                proposal_history.append(proposal_increment)
                target_history.append(proposal_increment + applied_control * step_dt)
                control_history.append(applied_control)

            running_spot_integral = running_spot_integral + current_spot * step_dt
            if barrier_hit is not None:
                assert barrier_level is not None
                if barrier_type == "down-out":
                    barrier_hit = barrier_hit | (next_spot <= barrier_level)
                else:
                    barrier_hit = barrier_hit | (next_spot >= barrier_level)

            spot_history.append(next_spot)
            variance_history.append(next_variance)
            current_spot = next_spot
            current_variance_state = next_variance_state
            current_variance = next_variance

        proposal_increments = (
            torch.stack(proposal_history, dim=1) if proposal_history is not None else None
        )
        target_increments = (
            torch.stack(target_history, dim=1) if target_history is not None else None
        )
        recorded_controls = (
            torch.stack(control_history, dim=1) if control_history is not None else None
        )
        return TwoDriverHestonPaths(
            spot=torch.stack(spot_history, dim=1),
            variance=torch.stack(variance_history, dim=1),
            log_likelihood=-stochastic_log_term - 0.5 * control_energy,
            control_energy=control_energy,
            running_spot_integral=running_spot_integral,
            barrier_hit=barrier_hit,
            step_dt=step_dt,
            proposal_brownian_increments=proposal_increments,
            target_brownian_increments=target_increments,
            controls=recorded_controls,
        )

    # ------------------------------------------------------------------
    # 1.3 Helpers
    # ------------------------------------------------------------------

    def _resolved_params(self, override: dict | None) -> dict:
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
        cov = 0.5 * (
            torch.abs(t_i) ** (2 * H) + torch.abs(t_j) ** (2 * H) - torch.abs(t_i - t_j) ** (2 * H)
        )
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

        V_t = ξ · exp( η · W^H_t − ½ η² Var[W^H_t] )
        dS_t = √V_t · S_t · dW_t^{(S)}
        dW_t^{(S)} = ρ dW_t^{(1)} + √(1-ρ²) dW_t^{(2)}
        W^H_t = √(2H) ∫₀ᵗ (t-s)^{H-½} dW_s^{(1)}  (Volterra process)

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
        if kappa_hybrid != 1:
            raise ValueError(
                "only the BLP kappa=1 hybrid scheme is implemented; "
                f"got kappa_hybrid={kappa_hybrid}"
            )
        if xi <= 0.0:
            raise ValueError(f"xi must be positive; got {xi}")
        if not (-1.0 <= rho <= 1.0):
            raise ValueError(f"rho must be in [-1, 1]; got {rho}")
        self.H = float(H)
        self.eta = float(eta)
        self.xi = float(xi)
        self.rho = float(rho)
        self.kappa_hybrid = int(kappa_hybrid)
        self.device = torch.device(device) if not isinstance(device, torch.device) else device

    def _hybrid_scheme_volterra(
        self,
        num_paths: int,
        n_steps: int,
        dt: float,
        H: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return driving increments, normalized Volterra paths, and variance.

        Implementation follows Bennedsen–Lunde–Pakkanen (2017) Alg. BN, with
        kappa=1: the singular kernel is integrated exactly on the most recent
        interval and replaced by its cell-average on earlier intervals. The
        returned process includes the ``sqrt(2H)`` normalization used by the
        standard rBergomi convention, so ``Var(W^H_t)=t^(2H)`` in the
        continuous-time limit.
        """
        if num_paths <= 0 or n_steps <= 0:
            raise ValueError("num_paths and n_steps must be positive")
        if dt <= 0.0:
            raise ValueError(f"dt must be positive; got {dt}")

        H = self.H if H is None else float(H)
        if not (0.0 < H < 0.5):
            raise ValueError(f"rBergomi requires H in (0, 0.5); got {H}")
        alpha = H - 0.5  # Volterra kernel exponent
        dtype = torch.float64

        # Time grid t_i = i·dt for i=0..n_steps
        # We need W^H(t_i) = sqrt(2H) ∫₀^{t_i} (t_i - s)^α dW^{(1)}_s.
        # Discrete hybrid: split into [t_{i-1}, t_i] (exact 2D Gaussian) and earlier (Riemann sum).

        # Exact covariance of (ΔW, ∫ΔW) on first sub-interval where ΔW = W^{(1)}_{t_i}-W^{(1)}_{t_{i-1}}
        # and ∫_{t_{i-1}}^{t_i} (t_i - s)^α dW_s.
        c11 = dt
        c12 = dt ** (alpha + 1.0) / (alpha + 1.0)
        c22 = dt ** (2 * alpha + 1.0) / (2 * alpha + 1.0)
        cov_local = torch.tensor([[c11, c12], [c12, c22]], device=self.device, dtype=dtype)
        L_local = torch.linalg.cholesky(cov_local)

        # Sample n_steps independent 2-vectors per path
        Z = torch.randn(num_paths, n_steps, 2, device=self.device, dtype=dtype)
        incr = Z @ L_local.T  # shape (paths, steps, 2)
        dW1 = incr[:, :, 0]  # ΔW^(1)_i on interval (t_{i-1}, t_i]
        dI = incr[:, :, 1]  # ∫_{t_{i-1}}^{t_i} (t_i - s)^α dW_s

        # Earlier cells use the cell-average of x^alpha. Because dW1 already
        # has scale sqrt(dt), the kernel weight has scale dt^alpha (not
        # dt^(alpha+1)). Invalid/future lags are clamped before taking a
        # fractional power, then set to zero with torch.where; this avoids the
        # NaN * 0 bug in the previous implementation.
        i_idx = torch.arange(1, n_steps + 1, device=self.device, dtype=dtype).unsqueeze(1)
        j_idx = torch.arange(0, n_steps, device=self.device, dtype=dtype).unsqueeze(0)
        lag = i_idx - j_idx  # lag=1 is handled exactly by dI
        lag_safe = torch.clamp(lag, min=1.0)
        lag_prev = lag_safe - 1.0
        avg_kernel = (
            (dt**alpha) * (lag_safe ** (alpha + 1.0) - lag_prev ** (alpha + 1.0)) / (alpha + 1.0)
        )
        weights = torch.where(lag >= 2.0, avg_kernel, torch.zeros_like(avg_kernel))

        # Convolve: earlier[path, i] = sum_j b[i, j] * dW1[path, j]
        earlier = dW1 @ weights.T  # (paths, steps)

        scale = math.sqrt(2.0 * H)
        Y = torch.zeros(num_paths, n_steps + 1, device=self.device, dtype=dtype)
        Y[:, 1:] = scale * (earlier + dI)

        # Deterministic variance of the discretized Gaussian process. Using
        # this in the Wick exponential preserves E[V_t]=xi at finite dt and
        # converges to t^(2H) as the hybrid grid is refined.
        Y_var = torch.zeros(n_steps + 1, device=self.device, dtype=dtype)
        Y_var[1:] = (scale**2) * (c22 + dt * (weights**2).sum(dim=1))
        return dW1, Y, Y_var

    def simulate(
        self,
        S0: float,
        T: float,
        dt: float,
        num_paths: int,
        mu: float = 0.0,
        override_params: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if T <= 0.0 or dt <= 0.0:
            raise ValueError(f"T and dt must be positive; got T={T}, dt={dt}")
        if num_paths <= 0:
            raise ValueError(f"num_paths must be positive; got {num_paths}")

        params = self._resolved(override_params)
        H, eta, xi, rho = params["H"], params["eta"], params["xi"], params["rho"]
        n_steps = max(1, int(math.ceil(T / dt)))
        step_dt = T / n_steps

        dW1, Y, Y_var = self._hybrid_scheme_volterra(num_paths, n_steps, step_dt, H=H)

        # Independent Brownian driver for the orthogonal component of the price
        Z2 = torch.randn(num_paths, n_steps, device=self.device, dtype=dW1.dtype) * math.sqrt(
            step_dt
        )
        dW_S = rho * dW1 + math.sqrt(max(1.0 - rho * rho, 0.0)) * Z2

        # Discrete Wick exponential with the exact variance of the hybrid
        # Gaussian approximation.
        drift = 0.5 * (eta**2) * Y_var
        V = xi * torch.exp(eta * Y - drift.unsqueeze(0))
        V = torch.clamp(V, min=1e-10)

        # Price path with explicit Euler in log-space for stability
        logS = torch.zeros(num_paths, n_steps + 1, device=self.device, dtype=dW1.dtype)
        logS[:, 0] = math.log(S0)
        for k in range(1, n_steps + 1):
            v_k = V[:, k - 1]
            logS[:, k] = (
                logS[:, k - 1] + (mu - 0.5 * v_k) * step_dt + torch.sqrt(v_k) * dW_S[:, k - 1]
            )
        S = torch.exp(logS)
        return S, V

    def _resolved(self, override: dict | None) -> dict:
        base = {"H": self.H, "eta": self.eta, "xi": self.xi, "rho": self.rho}
        if override:
            base.update(override)
        base = {key: float(value) for key, value in base.items()}
        if not (0.0 < base["H"] < 0.5):
            raise ValueError(f"rBergomi requires H in (0, 0.5); got {base['H']}")
        if base["xi"] <= 0.0:
            raise ValueError(f"xi must be positive; got {base['xi']}")
        if not (-1.0 <= base["rho"] <= 1.0):
            raise ValueError(f"rho must be in [-1, 1]; got {base['rho']}")
        return base
