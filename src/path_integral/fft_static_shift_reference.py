"""Implicit exact uniform Volterra/next-price Gaussian shift mixture.

Same law as build_volterra_excursion_guide, without storing K mean vectors or
an N-by-2N monitoring matrix. Scoring factorizes the uniform amplitude grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .baselines.rbergomi_common import RBergomiBaselineProblem
from .cached_volterra_terminal import CachedVolterraTerminal


@dataclass(frozen=True)
class FFTStaticShiftReference:
    cache: CachedVolterraTerminal
    amplitudes: tuple[float, ...]
    price_amplitudes: tuple[float, ...]
    defensive_mass: float

    @classmethod
    def create(cls, problem: RBergomiBaselineProblem, *, amplitudes: list[float],
               price_amplitudes: list[float], defensive_mass: float = .1) -> FFTStaticShiftReference:
        if (problem.steps < 2 or not amplitudes or not price_amplitudes
                or any(not math.isfinite(x) or x <= 0 for x in amplitudes)
                or any(not math.isfinite(x) or x < 0 for x in price_amplitudes)
                or not math.isfinite(defensive_mass) or not 0 < defensive_mass < 1):
            raise ValueError("valid finite uniform shift grid and defensive mass required")
        return cls(CachedVolterraTerminal.create(problem), tuple(amplitudes), tuple(price_amplitudes), defensive_mass)

    @property
    def dimension(self) -> int:
        return self.cache.problem.local_dimension

    @property
    def component_count(self) -> int:
        return 1+(self.cache.problem.steps-1)*len(self.amplitudes)*len(self.price_amplitudes)

    def standardized_drivers(self, z: torch.Tensor) -> torch.Tensor:
        p, k = self.cache.problem, self.cache.kernel
        p._validate_latent(z, self.dimension)
        local = z.reshape(-1, p.steps, 2) @ k.local_cholesky.T
        history = torch.fft.irfft(torch.fft.rfft(local[:, :, 0], n=self.cache.fft_length, dim=1)
                                 *self.cache.kernel_transform, n=self.cache.fft_length, dim=1)[:, :p.steps]
        return math.sqrt(2*p.hurst)*(history+local[:, :, 1])[:, :-1]/k.volterra_variance[1:-1].sqrt()

    def log_q_over_p(self, z: torch.Tensor) -> torch.Tensor:
        y = self.standardized_drivers(z)
        sign = -1. if self.cache.problem.rho > 0 else 1.
        price = sign*z[:, 2::2]
        amplitudes = torch.tensor(self.amplitudes, dtype=torch.float64)
        prices = torch.tensor(self.price_amplitudes, dtype=torch.float64)
        vol_terms = torch.logsumexp(y[:, :, None]*amplitudes-.5*amplitudes.square(), 2)
        price_terms = torch.logsumexp(price[:, :, None]*prices-.5*prices.square(), 2)
        shift_terms = torch.logsumexp(vol_terms+price_terms, 1)+math.log1p(-self.defensive_mass)-math.log(self.component_count-1)
        return torch.logaddexp(torch.full_like(shift_terms, math.log(self.defensive_mass)), shift_terms)

    def apply_labels(self, base: torch.Tensor, *, monitors: torch.Tensor, amplitudes: torch.Tensor,
                     prices: torch.Tensor, natural: torch.Tensor) -> torch.Tensor:
        p, k = self.cache.problem, self.cache.kernel
        p._validate_latent(base, self.dimension)
        count, n = len(base), p.steps
        for values, lower, upper in ((monitors, 1, n), (amplitudes, 0, len(self.amplitudes)), (prices, 0, len(self.price_amplitudes))):
            if (values.shape != (count,) or values.dtype != torch.int64 or values.device.type != "cpu"
                    or (values < lower).any() or (values >= upper).any()):
                raise ValueError("valid component index vectors required")
        if natural.shape != (count,) or natural.dtype != torch.bool or natural.device.type != "cpu":
            raise ValueError("natural labels must be a CPU boolean vector")
        positions = torch.arange(n)
        lags = monitors[:, None]-positions
        historical = (lags >= 2)
        coefficients = k.historical_kernel[(lags-1).clamp(0, n-1)]
        local = positions[None] == monitors[:, None]-1
        scale = math.sqrt(2*p.hurst)/k.volterra_variance[monitors].sqrt()
        drift = torch.zeros((count, n, 2), dtype=torch.float64)
        drift[:, :, 0] = scale[:, None]*(historical*coefficients*k.local_cholesky[0, 0]+local*k.local_cholesky[1, 0])
        drift[:, :, 1] = scale[:, None]*local*k.local_cholesky[1, 1]
        a = torch.tensor(self.amplitudes, dtype=torch.float64)[amplitudes]
        b = torch.tensor(self.price_amplitudes, dtype=torch.float64)[prices]
        drift *= a[:, None, None]
        sign = -1. if p.rho > 0 else 1.
        flat = drift.reshape(count, -1)
        flat[torch.arange(count), 2*monitors] += sign*b
        return base+torch.where(natural[:, None], torch.zeros_like(flat), flat)

    def sample(self, count: int, *, path_seed: int, label_seed: int) -> torch.Tensor:
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("positive integer count required")
        if (path_seed == label_seed or
                any(isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < 2**63 for s in (path_seed, label_seed))):
            raise ValueError("valid independent CPU PRNG seeds required")
        paths, labels = torch.Generator().manual_seed(path_seed), torch.Generator().manual_seed(label_seed)
        base = torch.randn((count, self.dimension), dtype=torch.float64, generator=paths)
        # Each label is sampled from the declared normalized product mixture.
        natural = torch.rand(count, dtype=torch.float64, generator=labels) < self.defensive_mass
        monitors = torch.randint(1, self.cache.problem.steps, (count,), generator=labels)
        amplitudes = torch.randint(len(self.amplitudes), (count,), generator=labels)
        prices = torch.randint(len(self.price_amplitudes), (count,), generator=labels)
        return self.apply_labels(base, monitors=monitors, amplitudes=amplitudes, prices=prices, natural=natural)
