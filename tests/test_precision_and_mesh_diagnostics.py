import copy
import math
from dataclasses import replace

import pytest
import torch

from experiments.post_audit_high_eta_precision_validation import decision, grid
from experiments.post_audit_p1_recovery_p2_parallel import identity, seed
from experiments.post_audit_p2_driver_quadrature_diagnosis import (
    batch_values,
    summarize,
    thresholds,
)
from src.path_integral.cached_volterra_terminal import CachedVolterraTerminal
from src.path_integral.fft_excursion_reference import (
    AdjacentBlockLatentAdapter,
    FFTExcursionReference,
)
from src.path_integral.path_functionals import TerminalThresholdTask
from tests.test_p1_recovery_p2_parallel import problem


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_validation_grid_fixed_count_and_independent_streams():
    jobs = grid()
    assert len(jobs) == 1024
    names = [identity(j) for j in jobs]
    assert len(set(names)) == len(names)
    streams = [seed(name, stream) for name in names for stream in ("path", "label", "whole")]
    assert len(set(streams)) == len(streams)
    assert sum(4096 if j["target"] == "paired" else 254976 for j in jobs)+128 == 68419712


def test_scoped_precision_success_never_opens_global_gates():
    rows = [{"method": method, "target": target,
             "summary": {"log_mean": -10., "relative_se": .001, "maximum_fraction": .001},
             "sensitivity": {"maximum_leave_one_out_relative_shift": .001}}
            for method in ("guide-iid", "excursion-iid", "natural-smc", "proposal-smc") for target in ("mu", "risk")]
    result = decision(rows)
    assert result["high_eta_parent0_development_precision_pass"]
    assert not any(result[k] for k in ("p1_qualified", "p2_qualified", "large_training_authorized"))
    broken = copy.deepcopy(rows)
    broken[0]["summary"]["relative_se"] = .015001
    assert not decision(broken)["high_eta_parent0_development_precision_pass"]
    broken = copy.deepcopy(rows)
    broken[0]["summary"]["log_mean"] += math.log(1.2)
    assert not decision(broken)["high_eta_parent0_development_precision_pass"]


def test_price_coarsening_constant_volatility_limit():
    p = replace(problem(32), eta=1e-12, task=TerminalThresholdTask(level=100.))
    r, c, a = FFTExcursionReference.create(p), CachedVolterraTerminal.create(replace(p, steps=16)), AdjacentBlockLatentAdapter.create(p)
    generator = torch.Generator().manual_seed(471)
    z = torch.randn((1024, 64), dtype=torch.float64, generator=generator)
    eps = torch.randn((1024, 16), dtype=torch.float64, generator=generator)
    f, middle, coarse = thresholds(r, c, a, z, eps)
    torch.testing.assert_close(f, middle, rtol=0, atol=1e-10)
    torch.testing.assert_close(f, coarse, rtol=0, atol=1e-10)


def test_signed_mesh_decomposition_and_corruption_detection():
    p = problem(32)
    part = batch_values(FFTExcursionReference.create(p), CachedVolterraTerminal.create(replace(p, steps=16)),
                        AdjacentBlockLatentAdapter.create(p), "test-driver-quad")
    summary = summarize([part])
    assert summary["count"] == 2048
    assert summary["triangle_identity_max_scaled_error"] < 1e-10
    pieces = summary["components"]
    assert pieces[0]["mean"]+pieces[1]["mean"] == pytest.approx(pieces[2]["mean"], abs=1e-14)
    broken = copy.deepcopy(part)
    broken["corrections"][0]["sign"][0] *= -1
    with pytest.raises(AssertionError):
        summarize([broken])
