import math

import pytest

from src.path_integral.complexity_certificates import (
    amortization_break_even_queries,
    local_newton_error_bound,
    relative_rmse_sample_count,
)


def test_local_newton_bound_obeys_quadratic_recurrence() -> None:
    assert local_newton_error_bound(
        0.2,
        strong_convexity=2.0,
        hessian_lipschitz=4.0,
        iterations=3,
    ) == pytest.approx(0.00000256)
    assert local_newton_error_bound(
        0.0,
        strong_convexity=2.0,
        hessian_lipschitz=4.0,
        iterations=10,
    ) == 0.0


def test_relative_rmse_sample_count_is_the_smallest_integer_certificate() -> None:
    relative_variance = 3.7
    tolerance = 0.1
    samples = relative_rmse_sample_count(relative_variance, tolerance)
    assert relative_variance / samples <= tolerance**2
    assert relative_variance / (samples - 1) > tolerance**2


def test_amortization_break_even_is_fail_closed_without_per_query_saving() -> None:
    assert amortization_break_even_queries(
        100.0,
        cold_work_per_query=12.0,
        amortized_work_per_query=10.0,
    ) == 50
    assert (
        amortization_break_even_queries(
            100.0,
            cold_work_per_query=10.0,
            amortized_work_per_query=10.0,
        )
        is None
    )
    with pytest.raises(ValueError):
        relative_rmse_sample_count(math.inf, 0.1)
