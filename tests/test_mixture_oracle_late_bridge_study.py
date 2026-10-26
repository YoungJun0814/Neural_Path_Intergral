import json

import pytest
import torch

import experiments.post_audit_mixture_oracle_late_bridge as study
from experiments.post_audit_analytic_residual_batch import Budget
from src.path_integral.gaussian_span_marginalization import GaussianMixtureShiftSpec
from src.path_integral.reference_protocol import canonical_json_bytes
from src.path_integral.seed_ledger import SeedKey, SeedLedger


def test_temperature_allocation_is_the_only_SMC_configuration_change():
    a = study.configuration({"schedule": "quartic", "arm": "component"}, 128)
    b = study.configuration({"schedule": "linear", "arm": "component"}, 128)
    assert (a.particles, a.arm, a.scale, a.resample_every) == (b.particles, b.arm, b.scale, b.resample_every)
    assert len(a.temperatures) == len(b.temperatures) == 33
    assert sum(beta > .5 for beta in b.temperatures) == 16
    assert sum(beta > .5 for beta in a.temperatures) == 6
    with pytest.raises(ValueError):
        study.configuration({"schedule": "retuned", "arm": "component"}, 128)


@pytest.mark.parametrize("phase,count,old_count", [("oracle", 64, 1), ("volterra", 16, 1)])
def test_all_prespecified_jobs_have_unique_fresh_sealed_streams(phase, count, old_count):
    previous = SeedLedger()
    previous.allocate(SeedKey("old-study", "a", "development", "cell", 32, 0, "path"))
    planned = study.jobs(phase, ["a", "b"])
    mapping, receipt = study.seal(phase, planned, [previous.to_dict()])
    assert len(planned) == count and len(mapping) == count
    assert len(receipt["ledger"]["records"]) == count*8
    assert receipt["previous_seeds_compared"] == old_count
    assert receipt["collision_count"] == 0
    assert study.seal(phase, planned, [previous.to_dict()]) == (mapping, receipt)
    assert len(set(study.identity(j) for j in planned)) == count
    assert canonical_json_bytes(receipt)


def test_invalid_or_partial_job_design_is_not_silently_accepted():
    for phase, cells in (("unknown", []), ("volterra", ["a"]), ("volterra", ["a", "a"])):
        with pytest.raises(ValueError):
            study.jobs(phase, cells)
    with pytest.raises(ValueError):
        study.summarize_volterra([], ["a", "b"])


def test_gaussian_context_budget_is_separate_from_Volterra_physics_and_cannot_expand():
    budget = study.GaussianBudget()
    budget.charge(3200000)
    with pytest.raises(TimeoutError):
        budget.charge(1)
    assert budget.contexts == 3200000


@pytest.mark.parametrize("field", ["config", "source", "seeds", "mapping"])
def test_complete_protocol_binding_detects_each_replaced_field(field):
    mapping, ledger = {"job": 1}, {"ledger": 2}
    value = {"source": {"hash": 3}, "config": {"n": 4}, "seeds": ledger}
    protocol = {**value, "mapping": mapping}
    study.check_protocol(value, protocol, mapping, ledger)
    protocol[field] = {}
    with pytest.raises(ValueError, match="full protocol"):
        study.check_protocol(value, protocol, mapping, ledger)


@pytest.mark.parametrize("failure", ["none", "lineage", "normalizer", "mode", "reference", "resource", "technical", "numerical"])
def test_decision_hierarchy_preserves_failures_and_never_fits_G_or_grants_P2(failure):
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        features = torch.randn((128, 5), dtype=torch.float64, generator=torch.Generator().manual_seed(412354)).tolist()
        records = []
        for job in study.jobs("volterra", ["a", "b"]):
            log_z = -2.
            if failure == "normalizer" and job["schedule"] == "linear" and job["replicate"] == 3:
                log_z = 5.
            if failure == "reference" and job["schedule"] == "quartic":
                log_z = 0.
            data = {"log_normalizer": log_z, "gate": {"pass": failure != "lineage"},
                "final_features": [[float(job["replicate"])]*5]*128 if failure == "mode" else features,
                "final_weights": [1/128]*128}
            row = {"job": job, "status": "failed" if failure in ("resource", "technical", "numerical") else "complete", "result": data}
            if failure in ("resource", "technical"):
                row["failure_type"] = failure.upper()
            records.append(row)
        result = study.summarize_volterra(records, ["a", "b"])
        expected = {"none": "LATE_BRIDGE_BANK_CANDIDATE_ONLY", "lineage": "STOP_LATE_BRIDGE_BANK_UNQUALIFIED",
            "normalizer": "STOP_LATE_BRIDGE_ISLAND_OR_MODE_UNRESOLVED", "mode": "STOP_LATE_BRIDGE_ISLAND_OR_MODE_UNRESOLVED",
            "reference": "STOP_LATE_BRIDGE_REFERENCE_UNRESOLVED", "resource": "STOP_RESOURCE_LIMIT",
            "technical": "STOP_TECHNICAL_FAILURE", "numerical": "STOP_NUMERICAL_TARGET_UNRESOLVED"}
        assert result["decision"] == expected[failure]
        assert result["actual_G_fits"] == 0 and not result["event_q_changed"]
        assert not result["P2_qualified"] and not result["D3_authorized"] and not result["large_training_authorized"]
        assert canonical_json_bytes(result)
    finally:
        torch.set_num_threads(old)


@pytest.mark.parametrize("error,category", [(TimeoutError("cap"), "RESOURCE"),
    (FloatingPointError("potential"), "NUMERICAL"), (OSError("storage"), "TECHNICAL")])
def test_computational_failure_retains_category_and_consumed_work(monkeypatch, error, category):
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(study, "run_target_smc", fail)
    q = GaussianMixtureShiftSpec(torch.zeros((1, 62), dtype=torch.float64), torch.ones(1, dtype=torch.float64))
    job = study.jobs("volterra", ["a", "b"])[0]
    record = study.volterra_island(None, q, job, {}, Budget(0, 0, 30))
    assert record["status"] == "failed" and record["failure_type"] == category
    assert record["physical"] == record["scalar"] == 0
    assert record["completed_stages"] == []


def test_failed_oracle_cannot_authorize_Volterra_production(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "ROOT", tmp_path)
    path = tmp_path/study.ORACLE
    path.mkdir(parents=True)
    (path/"result.json").write_text(json.dumps({"summary": {"pass": False}}))
    (path/"audit.json").write_text("{}")
    with pytest.raises(ValueError, match="passed and fully replayed"):
        study.qualified_oracle()
