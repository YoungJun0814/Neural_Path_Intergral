"""Same exact excursion density, evaluated using the BLP linear convolution."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .baselines.rbergomi_common import RBergomiBaselineProblem
from .cached_volterra_terminal import CachedVolterraTerminal
from .collective_variance_reference import CollectiveVarianceReference
from .conditional_excursion_reference import ConditionalExcursionReference
from .coverage_envelope_reference import CoverageEnvelopeReference
from .rbergomi_coupling import adjacent_local_gaussian_coefficients
from .rbergomi_fft import blp_fft_kernel


@dataclass(frozen=True)
class FFTExcursionReference:
    reference: ConditionalExcursionReference
    cache: CachedVolterraTerminal

    @classmethod
    def create(cls, p: RBergomiBaselineProblem) -> FFTExcursionReference:
        cache = CachedVolterraTerminal.create(p)
        n, kernel = p.steps, cache.kernel
        rows = torch.zeros((n-1, 2*n), dtype=torch.float64)
        scale = math.sqrt(2*p.hurst)
        for i in range(1, n):
            if i > 1:
                rows[i-1, :2*(i-1):2] = scale*kernel.local_cholesky[0, 0]*kernel.historical_kernel[1:i].flip(0)
            rows[i-1, 2*(i-1):2*i] = scale*kernel.local_cholesky[1]
        norms = kernel.volterra_variance[1:n].sqrt()
        if not torch.allclose(rows.square().sum(1), norms.square(), rtol=1e-12, atol=1e-14):
            raise ValueError("analytic Volterra row variance mismatch")
        return cls(ConditionalExcursionReference(rows/norms[:, None], 2*torch.arange(1, n),
                                                 -1. if p.rho > 0 else 1.), cache)

    @property
    def dimension(self) -> int:
        return self.reference.dimension

    def sample(self, count: int, *, path_seed: int, label_seed: int) -> torch.Tensor:
        return self.reference.sample(count, path_seed=path_seed, label_seed=label_seed)

    def standardized_monitor_values(self, z: torch.Tensor) -> torch.Tensor:
        self.cache.problem._validate_latent(z, self.dimension)
        kernel, p = self.cache.kernel, self.cache.problem
        local = z.reshape(-1, p.steps, 2) @ kernel.local_cholesky.T
        history = torch.fft.irfft(torch.fft.rfft(local[:, :, 0], n=self.cache.fft_length, dim=1)
                                 * self.cache.kernel_transform, n=self.cache.fft_length, dim=1)[:, :p.steps]
        return math.sqrt(2*p.hurst)*(history+local[:, :, 1])[:, :-1]/kernel.volterra_variance[1:-1].sqrt()

    def excursion_log_density(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r = self.reference
        price = r.price_sign*z[:, r.price_indices]
        log_price_tail = float(torch.special.log_ndtr(torch.tensor(-r.price_threshold, dtype=torch.float64)))
        weight = math.log((1-r.defensive_mass)/(2*len(r.units)*len(r.thresholds)))
        terms = []
        for threshold in r.thresholds:
            active = y >= threshold
            tail = float(torch.special.log_ndtr(torch.tensor(-threshold, dtype=torch.float64)))
            terms += [active.sum(1).to(torch.float64).log()+weight-tail,
                      (active & (price >= r.price_threshold)).sum(1).to(torch.float64).log()+weight-tail-log_price_tail]
        terms.append(torch.full((len(z),), math.log(r.defensive_mass), dtype=torch.float64))
        return torch.logsumexp(torch.stack(terms, 1), 1)

    def log_q_over_p(self, z: torch.Tensor) -> torch.Tensor:
        return self.excursion_log_density(z, self.standardized_monitor_values(z))


@dataclass(frozen=True)
class FFTCoverageEnvelopeReference:
    """The same normalized envelope, with O(paths*N*log N) density scoring."""

    fast_excursion: FFTExcursionReference
    envelope: CoverageEnvelopeReference

    @classmethod
    def create(cls, p: RBergomiBaselineProblem) -> FFTCoverageEnvelopeReference:
        base = FFTExcursionReference.create(p)
        excursion = ConditionalExcursionReference(
            base.reference.units, base.reference.price_indices,
            base.reference.price_sign, (2., 3., 4., 5., 6.))
        collective = CollectiveVarianceReference(excursion.units, excursion.price_indices)
        return cls(FFTExcursionReference(excursion, base.cache),
                   CoverageEnvelopeReference(excursion, collective))

    @property
    def dimension(self) -> int:
        return self.envelope.dimension

    @property
    def cache(self) -> CachedVolterraTerminal:
        return self.fast_excursion.cache

    def sample(self, count: int, *, path_seed: int, label_seed: int) -> torch.Tensor:
        return self.envelope.sample(count, path_seed=path_seed, label_seed=label_seed)

    def log_q_over_p(self, z: torch.Tensor) -> torch.Tensor:
        y = self.fast_excursion.standardized_monitor_values(z)
        r = self.envelope.collective
        norms = y.square()+z[:, r.price_indices].square()
        components = .5*(1-1/r.variance)*norms-math.log(r.variance)
        inflated = torch.logsumexp(components, 1)+math.log((1-r.defensive_mass)/len(r.units))
        collective = torch.logaddexp(torch.full_like(inflated, math.log(r.defensive_mass)), inflated)
        result = torch.logsumexp(torch.stack((
            math.log(.4)+self.fast_excursion.excursion_log_density(z, y),
            math.log(.4)+collective,
            math.log(.2)+self.envelope.log_radial_over_p(z)), 1), 1)
        if not torch.isfinite(result).all():
            raise FloatingPointError("FFT coverage envelope density overflow; no clipping")
        return result


@dataclass(frozen=True)
class AdjacentBlockLatentAdapter:
    """Exact O(paths*N) adjacent adapter; no dense O(paths*N^2) multiply."""
    local_white: torch.Tensor
    fine_steps: int

    @classmethod
    def create(cls, p: RBergomiBaselineProblem) -> AdjacentBlockLatentAdapter:
        if p.steps < 2 or p.steps % 2:
            raise ValueError("even fine step count required")
        coefficients = adjacent_local_gaussian_coefficients(p.simulator(), fine_dt=p.step_dt)
        first, second = coefficients.first_cell_cholesky, coefficients.second_cell_cholesky
        raw = torch.stack((torch.cat((first[0, :2], second[0, :2], first[0, 2:])),
                           torch.cat((first[2, :2], second[1, :2], first[2, 2:]))))
        coarse = blp_fft_kernel(p.simulator(), n_steps=p.steps//2, step_dt=2*p.step_dt).local_cholesky
        return cls(torch.linalg.solve_triangular(coarse, raw, upper=False), p.steps)

    def __post_init__(self) -> None:
        a = self.local_white
        if (isinstance(self.fine_steps, bool) or not isinstance(self.fine_steps, int)
                or self.fine_steps < 2 or self.fine_steps % 2 or a.shape != (2, 5)
                or a.dtype != torch.float64 or a.device.type != "cpu" or not torch.isfinite(a).all()
                or not torch.allclose(a @ a.T, torch.eye(2, dtype=torch.float64), rtol=0, atol=2e-10)):
            raise ValueError("invalid local adjacent Gaussian whitening")

    def coarse(self, fine: torch.Tensor, augmentation: torch.Tensor) -> torch.Tensor:
        if (fine.ndim != 2 or fine.shape[1] != 2*self.fine_steps
                or augmentation.shape != (len(fine), self.fine_steps//2)
                or any(x.dtype != torch.float64 or x.device.type != "cpu" or not torch.isfinite(x).all()
                       for x in (fine, augmentation))):
            raise ValueError("invalid local coupling coordinates")
        return (fine.reshape(-1, self.fine_steps//2, 4) @ self.local_white[:, :4].T
                + augmentation[:, :, None]*self.local_white[:, 4]).reshape(-1, self.fine_steps)
