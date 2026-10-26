import json
from pathlib import Path

import pytest
import torch

from experiments.post_audit_analytic_residual_batch import Budget
from experiments.post_audit_conditional_risk_target_smc import (
    VolterraRiskRecorder,
    jobs,
    one_shot,
    seeds_for,
    summarize,
    tensor_digest,
)
from src.path_integral.cached_volterra_terminal import CachedVolterraTerminal
from src.path_integral.conditional_marginal_g import ACTIVE
from src.path_integral.gaussian_span_marginalization import GaussianMixtureShiftSpec
from src.path_integral.reference_protocol import canonical_json_bytes
from src.path_integral.residual_log_quadrature import log_gaussian_moments, slice_log_probability
from src.path_integral.residual_structural_oracle import LastVarianceResidualSlice
from src.path_integral.seed_ledger import SeedKey, SeedLedger
from tests.test_p1_recovery_p2_parallel import problem


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("phase", ["setup", "source_freeze", "protocol_io", "result_io"])
def test_startup_and_result_storage_failures_are_preserved_without_retry(tmp_path, phase):
    directory = tmp_path/phase

    def fail(path, progress):
        progress["phase"] = phase
        raise OSError("injected "+phase)

    with pytest.raises(OSError, match="injected"):
        one_shot(directory, fail)
    data = json.loads((directory/"failure.json").read_text())
    assert data["status"] == "STARTUP_OR_RUN_FAILED" and data["progress"]["phase"] == phase
    before = (directory/"failure.json").read_bytes()
    with pytest.raises(RuntimeError, match="one-shot"):
        one_shot(directory, fail)
    assert (directory/"failure.json").read_bytes() == before


def test_failed_receipt_persistence_never_turns_failure_into_success(tmp_path, monkeypatch, capsys):
    import experiments.post_audit_conditional_risk_target_smc as study

    def fail_writer(*args, **kwargs):
        raise OSError("receipt unavailable")

    def fail(path, progress):
        raise ValueError("original failure")

    monkeypatch.setattr(study, "write_json_atomic_nonoverwriting", fail_writer)
    with pytest.raises(ValueError, match="original"):
        one_shot(tmp_path/"failure", fail)
    assert "could not be persisted" in capsys.readouterr().err


def test_seed_manifest_covers_all_fixed_jobs_and_remains_disjoint():
    old = SeedLedger()
    old.allocate(SeedKey("previous", "test", "development", "a", 32, 0, "path"))
    parent = {"seeds": {"ledger": old.to_dict()}}
    mapping, receipt = seeds_for(["a", "b"], parent)
    assert len(jobs(["a", "b"])) == 16 and len(mapping) == 16
    assert len(receipt["ledger"]["records"]) == 128
    assert (mapping, receipt) == seeds_for(["a", "b"], parent)


def test_recorder_matches_scalar_law_and_independent_adaptive_moment():
    cache = CachedVolterraTerminal.create(problem(32))
    q = GaussianMixtureShiftSpec(torch.zeros((1, 62), dtype=torch.float64), torch.ones(1, dtype=torch.float64))
    x = torch.zeros((4, 62), dtype=torch.float64)
    x[1:, 0] = torch.tensor([-.3, .2, .4], dtype=torch.float64)
    budget = Budget(100000, 4, 60)
    recorder = VolterraRiskRecorder(cache, q, budget)
    value = recorder(x)
    assert budget.physical == 4 and len(recorder.trace) == 1
    assert recorder.trace[0]["status"] == "complete" and recorder.trace[0]["physical"] == 4
    z = torch.zeros((1, 64), dtype=torch.float64)
    z[:, ACTIVE] = x[:1]
    profile = LastVarianceResidualSlice.create(cache, z)
    import numpy as np
    reference = log_gaussian_moments(lambda u: slice_log_probability(profile, u))
    assert float(value.log_m2[0]) == pytest.approx(reference["moments"][1]["log_estimate"], abs=1e-9)
    np.testing.assert_allclose(value.features[:, 3].numpy(), x.square().sum(1).numpy())
    assert canonical_json_bytes(recorder.trace)


def test_budget_failure_is_recorded_before_any_physical_evaluation():
    cache = CachedVolterraTerminal.create(problem(32))
    q = GaussianMixtureShiftSpec(torch.zeros((1, 62), dtype=torch.float64), torch.ones(1, dtype=torch.float64))
    budget = Budget(100000, 0, 30)
    recorder = VolterraRiskRecorder(cache, q, budget)
    with pytest.raises(TimeoutError):
        recorder(torch.zeros((4, 62), dtype=torch.float64))
    assert recorder.trace[0]["status"] == "failed" and recorder.trace[0]["physical"] == 0
    assert budget.scalar == 0


def test_tensor_hash_binds_shape_dtype_and_content_without_numpy_scalar_payloads():
    x = torch.arange(6, dtype=torch.float64)
    assert tensor_digest(x) != tensor_digest(x.reshape(2, 3))
    assert tensor_digest(x) != tensor_digest(x.float())
    assert tensor_digest(x) == tensor_digest(x.clone())


def test_successful_one_shot_returns_value_but_never_overwrites_existing_artifact(tmp_path):
    def action(path: Path, progress):
        from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
        write_json_atomic_nonoverwriting(path/"done.json", {"status": "done"})
        return {"ok": True}
    assert one_shot(tmp_path/"one", action) == {"ok": True}
    with pytest.raises(RuntimeError):
        one_shot(tmp_path/"one", action)


@pytest.mark.parametrize("failure", ["none", "numerical", "lineage", "reference", "modes"])
def test_full_decision_hierarchy_never_adopts_G_or_grants_P2(failure):
    records = []
    generator = torch.Generator().manual_seed(19823)
    features = torch.randn((128, 5), dtype=torch.float64, generator=generator).tolist()
    for job in jobs(["a", "b"]):
        log_z = 10. if failure == "reference" and job["arm"] == "prior" and job["island"] == 3 else -2.
        f = [[float(job["island"])]*5]*128 if failure == "modes" else features
        result = {"log_normalizer": log_z, "final_weights": [1/128]*128, "final_features": f,
            "gate": {"pass": failure != "lineage"}}
        records.append({"job": job, "status": "failed" if failure == "numerical" else "complete", "result": result})
    summary = summarize(records, ["a", "b"])
    expected = {"none": "BANK_CANDIDATE_FOR_SEPARATE_G_PIPELINE_ONLY", "numerical": "STOP_NUMERICAL_TARGET_UNRESOLVED",
        "lineage": "STOP_GENEALOGY_OR_MUTATION_UNQUALIFIED", "reference": "NO_BANK_ADOPTION_REFERENCE_UNRESOLVED",
        "modes": "NO_BANK_ADOPTION_ISLAND_OR_MODE_UNRESOLVED"}[failure]
    assert summary["decision"] == expected
    assert summary["actual_G_fits"] == 0 and not summary["P2_qualified"]
    assert canonical_json_bytes(summary)


@pytest.mark.parametrize("category", ["RESOURCE", "TECHNICAL"])
def test_execution_limits_are_not_reported_as_scientific_target_failures(category):
    records = [{"job": job, "status": "failed", "failure_type": category} for job in jobs(["a", "b"])]
    result = summarize(records, ["a", "b"])
    assert result["decision"] == ("STOP_RESOURCE_LIMIT" if category == "RESOURCE" else "STOP_TECHNICAL_FAILURE")
    assert not result["P2_qualified"]
