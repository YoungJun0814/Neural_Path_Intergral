"""Conservative coupled L2 bound for original mean-cell BLP left drivers.

Not a weak-rate estimate, relative rare-probability certificate, or RMS-kernel bound.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BLPDriverMSEBound:
    hurst: float
    maturity: float
    steps: int
    step_dt: float
    increment_constant_upper: float
    projection_constant_upper: float
    integrated_mse_upper: float
    scope: str = "original_kappa1_cell_mean_BLP_left_driver_common_Brownian_coupling"


def original_blp_left_driver_mse_bound(*, hurst: float, maturity: float, steps: int) -> BLPDriverMSEBound:
    if (isinstance(hurst, bool) or not math.isfinite(hurst) or not 0 < hurst < .5
            or isinstance(maturity, bool) or not math.isfinite(maturity) or maturity <= 0
            or isinstance(steps, bool) or not isinstance(steps, int) or steps < 1):
        raise ValueError("H in (0,.5), finite T>0 and integer N>=1 required")
    alpha, h = hurst-.5, maturity/steps
    increment = 2+hurst*alpha*alpha/(1-hurst)
    projection = hurst*alpha*alpha/6*(1+1/(2-2*hurst))
    bound = maturity*h**(2*hurst)*(2*increment/(2*hurst+1)+2*projection)
    if not all(math.isfinite(v) and v >= 0 for v in (h, increment, projection, bound)) or h == 0 or bound == 0:
        raise FloatingPointError("unresolved numerical range for driver bound")
    return BLPDriverMSEBound(hurst, maturity, steps, h, increment, projection, bound)
