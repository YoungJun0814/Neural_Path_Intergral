"""Direct X-squared diagnostics from saved probability blocks; no raw X4 SE."""

from __future__ import annotations

import math
from typing import Any

import torch

from .structural_v2_reference import sensitivity


def direct_m2_blocks(parts: list[dict[str, Any]], *, bootstrap_seed: int,
                     expected_log_second: float) -> dict[str, Any]:
    if (len(parts) != 8 or len({p["count"] for p in parts}) != 1
            or any(type(p["count"]) is not int or p["count"] < 2 for p in parts)
            or not math.isfinite(expected_log_second)):
        raise ValueError("eight equal complete probability blocks required")
    squares = torch.tensor([p["log_sum_squares"] for p in parts], dtype=torch.float64)
    maxima = torch.tensor([p["log_maximum"] for p in parts], dtype=torch.float64)
    if not torch.isfinite(squares).all() or not torch.isfinite(maxima).all():
        raise ValueError("finite saved squared sums and maxima required")
    count = parts[0]["count"]
    total = float(torch.logsumexp(squares, 0))
    log_mean = total-math.log(8*count)
    if abs(log_mean-expected_log_second) > 1e-12:
        raise ValueError("merged direct second moment mismatch")
    maximum_share = math.exp(float(2*maxima.max())-total)
    if maximum_share > 1+1e-12:
        raise ValueError("maximum squared contribution exceeds observed sum")
    first, last = (float(torch.logsumexp(v, 0))-math.log(4*count) for v in squares.split(4))
    scale = max(first, last)
    a, b = math.exp(first-scale), math.exp(last-scale)
    return {"count": 8*count, "log_mean_m2": log_mean,
        "block_log_means_x2": (squares-math.log(count)).tolist(),
        "sensitivity": sensitivity((squares-math.log(count)).tolist(), bootstrap_seed=bootstrap_seed),
        "maximum_observed_squared_contribution_share": maximum_share,
        "first_last_half_symmetric_difference": 2*(a-b)/(a+b),
        "raw_iid_m2_se_available": False, "missing_moment": "sum_X4",
        "se_unit": "eight_equal_independent_probability_block_means_of_X2",
        "claim": "observed_sensitivity_not_tail_coverage_or_bias_proof"}
