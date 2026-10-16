import math

import pytest
import torch

from experiments.post_audit_p2_fine_resource_sweep import (
    identity,
    jobs,
    joint_work,
    seed,
    summarize,
)
from experiments.post_audit_p2_paired_fine_extension import jobs as old_jobs
from experiments.post_audit_p2_paired_fine_extension import seed as old_seed
from src.path_integral.gradient_umbrella_smc import estimate_gradient_umbrella_normalizers
from src.path_integral.weighted_tempered_smc import WeightedSMCConfig


def test_fine_resources_fixed_grid_and_cost_and_disjoint_seeds():
    grid = jobs()
    assert len(grid) == len({identity(j) for j in grid}) == len({seed(j) for j in grid}) == 32
    assert {seed(j) for j in grid}.isdisjoint({old_seed(j) for j in old_jobs()})
    assert sum(joint_work(j) for j in grid) == 2054144
    assert 2*sum(joint_work(j) for j in grid) == 4108288
    assert 6144*sum(joint_work(j) for j in grid) == 12620660736


def test_resource_selection_remains_forecast_not_accuracy_authorization():
    records = [{**j, "log_component_normalizers": [-10.+.01*j["rep"], -10.+.01*j["rep"]],
                "stages": [{"surviving_initial_ancestors": 20}], "wall": 1.} for j in jobs()]
    out = summarize(records)
    assert len(out["rows"]) == 4
    assert all(r["paired_difference"]["relative_difference"] == 0 for r in out["rows"])
    assert all(r["fresh_precision_forecast"]["required_wholes"] >= 32 for r in out["rows"])
    assert not any(out[k] for k in ("p1_qualified", "p2_qualified", "large_training_authorized"))
    assert all(math.isfinite(r["components"][0]["summary"]["relative_se"]) for r in out["rows"])
    with pytest.raises(ValueError):
        summarize(records[:-1])


def test_operational_restart_changes_all_seeds_without_changing_numeric_work():
    old, fresh = jobs(), jobs("paired-fine-resources-v2")
    assert {seed(j) for j in old}.isdisjoint({seed(j) for j in fresh})
    assert [joint_work(j) for j in old] == [joint_work(j) for j in fresh]


def test_eight_thread_full_kernel_replay_and_gaussian_normalizer_oracle():
    old = torch.get_num_threads()
    try:
        torch.set_num_threads(8)
        means = torch.tensor([.2, -.1, .3, .1], dtype=torch.float64)
        def components(z):
            return -.05*(z-means).square().sum(1, keepdim=True)
        values = []
        for rep in range(4):
            cfg = WeightedSMCConfig(256, tuple((k/15)**2 for k in range(16)), 2, .3, 1, 7100+rep,
                                    resample_every=4, resampling_scheme="stratified")
            r = estimate_gradient_umbrella_normalizers(components, dimension=4,
                                                       log_mixture_weights=torch.zeros(1, dtype=torch.float64), config=cfg)
            replay = estimate_gradient_umbrella_normalizers(components, dimension=4,
                                                            log_mixture_weights=torch.zeros(1, dtype=torch.float64), config=cfg)
            assert r.log_normalizer == replay.log_normalizer and r.stages == replay.stages
            assert r.potential_evaluations == r.gradient_evaluations == 7424
            values.append(math.exp(r.log_normalizer))
        exact = 1.1**-2*math.exp(-.05*float(means.square().sum())/1.1)
        assert sum(values)/4 == pytest.approx(exact, rel=.03)
    finally:
        torch.set_num_threads(old)
