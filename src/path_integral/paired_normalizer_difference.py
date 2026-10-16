"""Whole-run paired delta-method diagnostics, not a continuum certificate."""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any

import torch


def paired_normalizer_difference(log_fine: torch.Tensor, log_coarse: torch.Tensor, *,
                                 comparisons: int, margin: float = .1, alpha: float = .05) -> dict[str, Any]:
    if (log_fine.ndim != 1 or len(log_fine) < 2 or log_coarse.shape != log_fine.shape
            or any(v.dtype != torch.float64 or v.device.type != "cpu" or not torch.isfinite(v).all()
                   for v in (log_fine, log_coarse))
            or isinstance(comparisons, bool) or not isinstance(comparisons, int) or comparisons < 1
            or not math.isfinite(margin) or not 0 < margin < 1 or not 0 < alpha < 1):
        raise ValueError("complete independent paired whole-run vectors required")
    scale = torch.maximum(log_fine.max(), log_coarse.max())
    fine, coarse = torch.exp(log_fine-scale), torch.exp(log_coarse-scale)
    denominator = float(fine.mean())
    if not math.isfinite(denominator) or denominator <= 0:
        raise FloatingPointError("unresolved paired denominator")
    difference = fine-coarse
    relative = float(difference.mean())/denominator
    influence = difference-relative*fine
    se = float(influence.std(unbiased=True))/math.sqrt(len(fine))/denominator
    z = NormalDist().inv_cdf(1-alpha/(2*comparisons))
    upper = abs(relative)+z*se
    return {"whole_count": len(fine), "relative_difference": relative, "paired_relative_se": se,
            "upper_absolute_difference": upper, "margin": margin, "confidence_z": z,
            "pass": upper <= margin, "se_unit": "independent_paired_whole_or_equal_size_block",
            "status": "development_delta_method_not_continuum_or_tail_certificate"}
