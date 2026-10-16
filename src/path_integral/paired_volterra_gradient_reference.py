"""Exact adjacent finite-grid probability family on augmented Gaussian space."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from .baselines.rbergomi_common import RBergomiBaselineProblem
from .cached_volterra_terminal import CachedVolterraTerminal
from .fft_excursion_reference import AdjacentBlockLatentAdapter


@dataclass(frozen=True)
class PairedVolterraProbability:
    fine: CachedVolterraTerminal
    coarse: CachedVolterraTerminal
    adapter: AdjacentBlockLatentAdapter

    @classmethod
    def create(cls, problem: RBergomiBaselineProblem) -> PairedVolterraProbability:
        if problem.steps < 2 or problem.steps % 2:
            raise ValueError("even fine step count required")
        return cls(CachedVolterraTerminal.create(problem),
                   CachedVolterraTerminal.create(replace(problem, steps=problem.steps//2)),
                   AdjacentBlockLatentAdapter.create(problem))

    @property
    def dimension(self) -> int:
        return 5*self.fine.problem.steps//2

    def log_components(self, joint: torch.Tensor) -> torch.Tensor:
        if (joint.ndim != 2 or joint.shape[1] != self.dimension or len(joint) < 1
                or joint.dtype != torch.float64 or joint.device.type != "cpu"
                or not torch.isfinite(joint).all()):
            raise ValueError("finite augmented Gaussian coordinates required")
        d = self.fine.problem.local_dimension
        fine = joint[:, :d]
        coarse = self.adapter.coarse(fine, joint[:, d:])
        return torch.stack((self.fine.log_probability(fine), self.coarse.log_probability(coarse)), 1)
