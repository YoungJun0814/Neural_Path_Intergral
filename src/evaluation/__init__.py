"""Evaluation utilities: backtesting, VaR tests, efficiency metrics."""

from .backtest import (
    christoffersen_independence,
    compute_var_series,
    efficiency_metrics,
    kupiec_pof_test,
    var_exceptions,
)

__all__ = [
    "compute_var_series",
    "var_exceptions",
    "kupiec_pof_test",
    "christoffersen_independence",
    "efficiency_metrics",
]
