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


def strict_lognormal_variance(
    log_factor: torch.Tensor, *, xi: float
) -> torch.Tensor:
    """Evaluate ``xi * exp(log_factor)`` without silently flooring volatility."""

    if not math.isfinite(xi) or xi <= 0.0:
        raise ValueError("xi must be finite and positive")
    if not log_factor.is_floating_point() or not torch.isfinite(log_factor).all():
        raise FloatingPointError("lognormal variance exponent must be finite")
    log_variance = log_factor + math.log(xi)
    finfo = torch.finfo(log_variance.dtype)
    lower = math.log(finfo.tiny)
    upper = math.log(finfo.max)
    if bool((log_variance < lower).any()) or bool((log_variance > upper).any()):
        raise FloatingPointError(
            "rBergomi variance lies outside the normal floating-point range"
        )
    variance = torch.exp(log_variance)
    if (
        not torch.isfinite(variance).all()
        or bool((variance < finfo.tiny).any())
    ):
        raise FloatingPointError(
            "rBergomi lognormal variance became subnormal or nonfinite"
        )
    return variance


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
        running_spot_integral = torch.zeros(num_paths, device=self.device, dtype=simulation_dtype)
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
                applied_control = control_fn(time, current_spot, current_variance, running_average)
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
            proposal_increment = torch.stack((proposal_brownian_1, proposal_brownian_2), dim=-1)
            control_1 = applied_control[:, 0]
            control_2 = applied_control[:, 1]

            variance_plus = torch.clamp(current_variance, min=0.0)
            sqrt_variance = torch.sqrt(variance_plus)
            next_spot = current_spot * torch.exp(
                (params["mu"] + sqrt_variance * control_1 - 0.5 * variance_plus) * step_dt
                + sqrt_variance * proposal_brownian_1
            )

            variance_control = rho * control_1 + sqrt_one_minus_rho2 * control_2
            variance_brownian = (
                rho * proposal_brownian_1 + sqrt_one_minus_rho2 * proposal_brownian_2
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


@dataclass(frozen=True)
class TwoDriverRBergomiPaths:
    """Controlled BLP paths in the independent Brownian proposal basis.

    The local singular-cell integral is an auxiliary Gaussian correlated with
    the first Brownian increment.  It is recorded separately when requested so
    the augmented target/proposal path law can be reconstructed exactly.
    """

    spot: torch.Tensor
    variance: torch.Tensor
    volterra: torch.Tensor
    running_minimum: torch.Tensor
    log_likelihood: torch.Tensor
    control_energy: torch.Tensor
    step_dt: float
    proposal_brownian_increments: torch.Tensor | None
    target_brownian_increments: torch.Tensor | None
    proposal_local_integrals: torch.Tensor | None
    target_local_integrals: torch.Tensor | None
    controls: torch.Tensor | None


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

    def _hybrid_coefficients(
        self,
        n_steps: int,
        dt: float,
        *,
        H: float,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Return local Cholesky, historical weights, variance, and drift integral."""
        if n_steps <= 0 or not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("n_steps and dt must be positive")
        if not 0.0 < H < 0.5:
            raise ValueError(f"rBergomi requires H in (0, 0.5); got {H}")
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise TypeError("dtype must be floating point")
        alpha = H - 0.5
        c11 = dt
        c12 = dt ** (alpha + 1.0) / (alpha + 1.0)
        c22 = dt ** (2.0 * alpha + 1.0) / (2.0 * alpha + 1.0)
        covariance = torch.tensor(((c11, c12), (c12, c22)), device=self.device, dtype=dtype)
        local_cholesky = torch.linalg.cholesky(covariance)

        i_index = torch.arange(1, n_steps + 1, device=self.device, dtype=dtype).unsqueeze(1)
        j_index = torch.arange(0, n_steps, device=self.device, dtype=dtype).unsqueeze(0)
        lag = i_index - j_index
        safe_lag = torch.clamp(lag, min=1.0)
        previous_lag = safe_lag - 1.0
        average_kernel = (
            (dt**alpha)
            * (safe_lag ** (alpha + 1.0) - previous_lag ** (alpha + 1.0))
            / (alpha + 1.0)
        )
        historical_weights = torch.where(
            lag >= 2.0, average_kernel, torch.zeros_like(average_kernel)
        )
        scale = math.sqrt(2.0 * H)
        variance = torch.zeros(n_steps + 1, device=self.device, dtype=dtype)
        variance[1:] = (scale**2) * (c22 + dt * historical_weights.square().sum(dim=1))
        return local_cholesky, historical_weights, variance, c12

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
        dtype = torch.float64
        local_cholesky, weights, Y_var, _local_drift_coefficient = self._hybrid_coefficients(
            n_steps, dt, H=H, dtype=dtype
        )

        # Sample n_steps independent 2-vectors per path
        Z = torch.randn(num_paths, n_steps, 2, device=self.device, dtype=dtype)
        incr = Z @ local_cholesky.T  # shape (paths, steps, 2)
        dW1 = incr[:, :, 0]  # ΔW^(1)_i on interval (t_{i-1}, t_i]
        dI = incr[:, :, 1]  # ∫_{t_{i-1}}^{t_i} (t_i - s)^α dW_s

        # Convolve: earlier[path, i] = sum_j b[i, j] * dW1[path, j]
        earlier = dW1 @ weights.T  # (paths, steps)

        scale = math.sqrt(2.0 * H)
        Y = torch.zeros(num_paths, n_steps + 1, device=self.device, dtype=dtype)
        Y[:, 1:] = scale * (earlier + dI)

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
        result = self.simulate_controlled_two_driver(
            S0=S0,
            T=T,
            dt=dt,
            num_paths=num_paths,
            mu=mu,
            control_fn=None,
            override_params=override_params,
            record_augmented=False,
            dtype=torch.float64,
        )
        return result.spot, result.variance

    def simulate_controlled_two_driver(
        self,
        S0: float,
        T: float,
        dt: float,
        num_paths: int,
        *,
        mu: float = 0.0,
        control_fn: Callable[..., torch.Tensor] | None = None,
        override_params: dict | None = None,
        record_augmented: bool = False,
        dtype: torch.dtype = torch.float64,
    ) -> TwoDriverRBergomiPaths:
        r"""Simulate an exact mean-shift proposal for the declared BLP grid law.

        The proposal controls the two independent Brownian coordinates.  The
        recent-cell auxiliary integral receives the deterministic mean shift
        ``u1 * integral_0^dt r^(H-1/2) dr``.  Its Brownian-bridge residual is
        unchanged, so no additional likelihood term is present.
        """
        if not math.isfinite(S0) or S0 <= 0.0:
            raise ValueError("S0 must be finite and positive")
        if not math.isfinite(T) or not math.isfinite(dt) or T <= 0.0 or dt <= 0.0:
            raise ValueError(f"T and dt must be finite and positive; got T={T}, dt={dt}")
        if num_paths <= 0:
            raise ValueError(f"num_paths must be positive; got {num_paths}")
        if not math.isfinite(mu):
            raise ValueError("mu must be finite")
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise TypeError("dtype must be floating point")

        params = self._resolved(override_params)
        H, eta, xi, rho = params["H"], params["eta"], params["xi"], params["rho"]
        n_steps = max(1, int(math.ceil(T / dt)))
        step_dt = T / n_steps
        sqrt_dt = math.sqrt(step_dt)
        rho_perpendicular = math.sqrt(max(1.0 - rho * rho, 0.0))
        local_cholesky, historical_weights, volterra_variance, local_drift = (
            self._hybrid_coefficients(n_steps, step_dt, H=H, dtype=dtype)
        )
        volterra_scale = math.sqrt(2.0 * H)

        current_log_spot = torch.full((num_paths,), math.log(S0), device=self.device, dtype=dtype)
        current_volterra = torch.zeros(num_paths, device=self.device, dtype=dtype)
        current_variance = torch.full((num_paths,), xi, device=self.device, dtype=dtype)
        spot_history = [torch.exp(current_log_spot)]
        variance_history = [current_variance]
        volterra_history = [current_volterra]
        running_minimum = torch.exp(current_log_spot)
        running_minimum_history = [running_minimum]
        target_driver_one_history: list[torch.Tensor] = []

        reset_memory = getattr(control_fn, "reset_for_simulation", None)
        if callable(reset_memory):
            reset_memory(batch_size=num_paths, device=self.device, dtype=dtype)

        stochastic_log_term = torch.zeros(num_paths, device=self.device, dtype=torch.float64)
        control_energy = torch.zeros(num_paths, device=self.device, dtype=torch.float64)
        proposal_brownian_history: list[torch.Tensor] | None = [] if record_augmented else None
        target_brownian_history: list[torch.Tensor] | None = [] if record_augmented else None
        proposal_local_history: list[torch.Tensor] | None = [] if record_augmented else None
        target_local_history: list[torch.Tensor] | None = [] if record_augmented else None
        control_history: list[torch.Tensor] | None = [] if record_augmented else None

        for step in range(n_steps):
            time = step * step_dt
            current_spot = torch.exp(current_log_spot)
            if control_fn is None:
                control = torch.zeros((num_paths, 2), device=self.device, dtype=dtype)
            else:
                if bool(getattr(control_fn, "uses_running_minimum", False)):
                    control = control_fn(
                        time,
                        current_spot,
                        current_variance,
                        current_volterra,
                        running_minimum,
                    )
                else:
                    control = control_fn(time, current_spot, current_variance, current_volterra)
                if not isinstance(control, torch.Tensor):
                    raise TypeError("rBergomi control_fn must return a torch.Tensor")
                if control.shape != (num_paths, 2):
                    raise ValueError(
                        "rBergomi control_fn must return shape (num_paths, 2); "
                        f"got {tuple(control.shape)}"
                    )
                if control.device != self.device or control.dtype != dtype:
                    raise ValueError(
                        "rBergomi control output must match simulator device and dtype"
                    )
                if not torch.isfinite(control).all():
                    raise ValueError("rBergomi control output must be finite")

            # Causal order: evaluate the control before sampling this interval.
            local_standard_normal = torch.randn(num_paths, 2, device=self.device, dtype=dtype)
            local_pair = local_standard_normal @ local_cholesky.T
            proposal_driver_one = local_pair[:, 0]
            proposal_local_integral = local_pair[:, 1]
            proposal_driver_two = torch.randn(num_paths, device=self.device, dtype=dtype) * sqrt_dt
            proposal_brownian = torch.stack((proposal_driver_one, proposal_driver_two), dim=-1)
            target_brownian = proposal_brownian + control * step_dt
            target_driver_one = target_brownian[:, 0]
            target_driver_two = target_brownian[:, 1]
            target_local_integral = proposal_local_integral + control[:, 0] * local_drift
            observe_increment = getattr(control_fn, "observe_target_increment", None)
            if callable(observe_increment):
                observe_increment(target_driver_one, step_dt)

            spot_increment = rho * target_driver_one + rho_perpendicular * target_driver_two
            next_log_spot = (
                current_log_spot
                + (mu - 0.5 * current_variance) * step_dt
                + torch.sqrt(current_variance) * spot_increment
            )

            target_driver_one_history.append(target_driver_one)
            target_driver_one_matrix = torch.stack(target_driver_one_history, dim=1)
            historical = torch.sum(
                target_driver_one_matrix * historical_weights[step, : step + 1],
                dim=1,
            )
            next_volterra = volterra_scale * (historical + target_local_integral)
            next_variance = strict_lognormal_variance(
                eta * next_volterra - 0.5 * (eta**2) * volterra_variance[step + 1],
                xi=xi,
            )
            if not torch.isfinite(next_variance).all() or not torch.isfinite(next_log_spot).all():
                raise FloatingPointError("controlled rBergomi path became nonfinite")

            control_64 = control.to(torch.float64)
            proposal_64 = proposal_brownian.to(torch.float64)
            stochastic_log_term = stochastic_log_term + torch.sum(control_64 * proposal_64, dim=-1)
            control_energy = control_energy + step_dt * torch.sum(control_64.square(), dim=-1)

            if proposal_brownian_history is not None:
                assert target_brownian_history is not None
                assert proposal_local_history is not None
                assert target_local_history is not None
                assert control_history is not None
                proposal_brownian_history.append(proposal_brownian)
                target_brownian_history.append(target_brownian)
                proposal_local_history.append(proposal_local_integral)
                target_local_history.append(target_local_integral)
                control_history.append(control)

            current_log_spot = next_log_spot
            running_minimum = torch.minimum(running_minimum, torch.exp(next_log_spot))
            current_volterra = next_volterra
            current_variance = next_variance
            spot_history.append(torch.exp(current_log_spot))
            volterra_history.append(current_volterra)
            variance_history.append(current_variance)
            running_minimum_history.append(running_minimum)

        def stack_optional(values: list[torch.Tensor] | None) -> torch.Tensor | None:
            return torch.stack(values, dim=1) if values is not None else None

        return TwoDriverRBergomiPaths(
            spot=torch.stack(spot_history, dim=1),
            variance=torch.stack(variance_history, dim=1),
            volterra=torch.stack(volterra_history, dim=1),
            running_minimum=torch.stack(running_minimum_history, dim=1),
            log_likelihood=-stochastic_log_term - 0.5 * control_energy,
            control_energy=control_energy,
            step_dt=step_dt,
            proposal_brownian_increments=stack_optional(proposal_brownian_history),
            target_brownian_increments=stack_optional(target_brownian_history),
            proposal_local_integrals=stack_optional(proposal_local_history),
            target_local_integrals=stack_optional(target_local_history),
            controls=stack_optional(control_history),
        )

    def _resolved(self, override: dict | None) -> dict:
        base = {"H": self.H, "eta": self.eta, "xi": self.xi, "rho": self.rho}
        if override:
            base.update(override)
        base = {key: float(value) for key, value in base.items()}
        if not all(math.isfinite(value) for value in base.values()):
            raise ValueError("all rBergomi parameters must be finite")
        if not (0.0 < base["H"] < 0.5):
            raise ValueError(f"rBergomi requires H in (0, 0.5); got {base['H']}")
        if base["xi"] <= 0.0:
            raise ValueError(f"xi must be positive; got {base['xi']}")
        if base["eta"] < 0.0:
            raise ValueError(f"eta must be nonnegative; got {base['eta']}")
        if not (-1.0 <= base["rho"] <= 1.0):
            raise ValueError(f"rho must be in [-1, 1]; got {base['rho']}")
        return base
