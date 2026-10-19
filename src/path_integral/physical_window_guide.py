"""Fixed-energy, causal physical-time Gaussian guide dictionaries (G0/G1).

Geometry oracle core only: no bank fitting, learned weights, or performance gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

from .baselines.rbergomi_common import RBergomiBaselineProblem
from .cached_volterra_terminal import CachedVolterraTerminal
from .gaussian_span_marginalization import GaussianMixtureShiftSpec, sample_gaussian_mixture
from .mixture import log_mixture_q_over_p


@dataclass(frozen=True)
class PhysicalWindowGuide:
    problem: RBergomiBaselineProblem
    kind: str
    centres: tuple[int, ...]
    template_kinds: tuple[str, ...]
    vol_window_steps: tuple[int, ...]
    price_window_steps: tuple[int, ...]
    feature_weights: torch.Tensor
    feature_variances: torch.Tensor
    vol_directions: torch.Tensor
    price_directions: torch.Tensor
    spec: GaussianMixtureShiftSpec
    maximum_causal_roundoff: float

    @classmethod
    def create(cls, problem: RBergomiBaselineProblem, *, kind: Literal["point", "mixed-window"],
               amplitudes: list[float], price_amplitudes: list[float],
               defensive_mass: float = .1) -> PhysicalWindowGuide:
        n = problem.steps
        if (n < 32 or n > 4096 or n % 32 or kind not in ("point", "mixed-window")
                or not amplitudes or not price_amplitudes
                or any(not math.isfinite(x) or x <= 0 for x in amplitudes)
                or any(not math.isfinite(x) or x < 0 for x in price_amplitudes)
                or not math.isfinite(defensive_mass) or not 0 < defensive_mass < 1):
            raise ValueError("fixed 32-multiple grid and valid guide parameters required")
        price_steps: tuple[int, ...]
        if kind == "point":
            centres = tuple(k*n//32 for k in range(1, 32))
            kinds, vol_steps, price_steps = ("point",)*31, (1,)*31, (1,)*31
        else:
            centres = (tuple(k*n//16 for k in range(1, 16))
                       +tuple(round(n/8+i*(3*n/4)/15) for i in range(16)))
            kinds = ("point",)*15+("window",)*16
            vol_steps = (1,)*15+(n//8,)*16
            # Every G1 price direction has positive, grid-independent duration.
            # The final point template is truncated at maturity (T/16).
            price_steps = tuple(min(n//8, n-c) for c in centres)
        weights = torch.zeros((31, n), dtype=torch.float64)
        for row, (centre, width) in enumerate(zip(centres, vol_steps, strict=True)):
            weights[row, centre-width:centre] = 1/width
        cache = CachedVolterraTerminal.create(problem)
        kernel = cache.kernel
        # Adjoint of causal linear convolution: reverse, convolve, reverse.
        history = torch.fft.irfft(torch.fft.rfft(weights.flip(1), n=cache.fft_length, dim=1)
                                 *cache.kernel_transform, n=cache.fft_length, dim=1)[:, :n].flip(1)
        raw = torch.zeros((31, n, 2), dtype=torch.float64)
        raw[:, :, 0] = math.sqrt(2*problem.hurst)*(kernel.local_cholesky[0, 0]*history+kernel.local_cholesky[1, 0]*weights)
        raw[:, :, 1] = math.sqrt(2*problem.hurst)*kernel.local_cholesky[1, 1]*weights
        raw = raw.reshape(31, 2*n)
        past = torch.arange(2*n)[None] < 2*torch.tensor(centres)[:, None]
        leakage = float(torch.where(past, torch.zeros_like(raw), raw).abs().max())
        if not math.isfinite(leakage) or leakage > 1e-11:
            raise ValueError("adjoint FFT violates causal support beyond roundoff")
        raw = torch.where(past, raw, torch.zeros_like(raw))
        variances = raw.square().sum(1)
        if not torch.isfinite(variances).all() or not (variances > 0).all():
            raise ValueError("degenerate physical driver feature")
        u = raw/variances.sqrt()[:, None]
        v = torch.zeros_like(u)
        sign = -1. if problem.rho > 0 else 1.
        for row, (centre, width) in enumerate(zip(centres, price_steps, strict=True)):
            v[row, 2*torch.arange(centre, centre+width)] = sign/math.sqrt(width)
        v = v/torch.linalg.vector_norm(v, dim=1, keepdim=True)
        torch.testing.assert_close(u.square().sum(1), torch.ones(31, dtype=torch.float64), rtol=0, atol=2e-12)
        torch.testing.assert_close(v.square().sum(1), torch.ones(31, dtype=torch.float64), rtol=0, atol=2e-12)
        if not torch.equal((u*v).sum(1), torch.zeros(31, dtype=torch.float64)):
            raise ValueError("past driver/future price orthogonality lost")
        means = torch.stack([torch.zeros(2*n, dtype=torch.float64)]
                            +[a*u[i]+b*v[i] for i in range(31) for a in amplitudes for b in price_amplitudes])
        mixture_weights = torch.full((len(means),), (1-defensive_mass)/(len(means)-1), dtype=torch.float64)
        mixture_weights[0] = defensive_mass
        spec = GaussianMixtureShiftSpec(means, mixture_weights)
        return cls(problem, kind, centres, kinds, vol_steps, price_steps, weights, variances, u, v, spec, leakage)

    @property
    def dimension(self) -> int:
        return self.spec.dimension

    def log_q_over_p(self, z: torch.Tensor) -> torch.Tensor:
        self.problem._validate_latent(z, self.dimension)
        component_logs = z @ self.spec.means.T-.5*self.spec.means.square().sum(1)
        return log_mixture_q_over_p(component_logs, self.spec.weights)

    def sample(self, count: int, *, path_seed: int, label_seed: int) -> torch.Tensor:
        if (isinstance(count, bool) or not isinstance(count, int) or count < 1 or path_seed == label_seed
                or any(isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < 2**63 for s in (path_seed, label_seed))):
            raise ValueError("positive count and separate CPU PRNG seeds required")
        return sample_gaussian_mixture(self.spec, count,
                                       gaussian_generator=torch.Generator().manual_seed(path_seed),
                                       label_generator=torch.Generator().manual_seed(label_seed)).target_coordinates
