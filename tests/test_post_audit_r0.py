"""Adversarial R0 checks of the new evidence and conditional CE contracts."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from experiments.post_audit_r0_smoke import ROOT, build_smoke
from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.baselines.weighted_conditional_ce import (
    WeightedConditionalCEConfig,
    weighted_gaussian_moments,
)
from src.path_integral.comparator_qualification import (
    QualificationPolicy,
    qualify_estimate,
)
from src.path_integral.defensive_proposal_selection import (
    select_defensive_proposal_by_second_moment,
)
from src.path_integral.finite_rank_gaussian_transport import (
    DefensiveFiniteRankGaussianMixture,
    FiniteRankGaussianComponent,
)
from src.path_integral.legacy_v16_semantic_adapter import audit_legacy_v16
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.research_result_audit import (
    _load_bound_file,
    audit_measurement,
    verify_artifact_graph,
)
from src.path_integral.research_result_contract import Moments
from src.path_integral.volterra_conditional_payoffs import (
    evaluate_rbergomi_conditional_terminal,
)


@pytest.fixture(scope="module")
def smoke() -> dict[str, Any]:
    config = yaml.safe_load(
        (ROOT / "configs/post_audit/r0_smoke_v1.yaml").read_text(encoding="utf-8")
    )
    return build_smoke(config)


def test_smoke_audit_recomputes_both_methods_without_performance_claim(smoke: dict[str, Any]) -> None:
    audit = audit_measurement(smoke)
    assert audit["integrity"] == "pass"
    assert audit["semantic_validity"] == "pass"
    assert audit["performance"] == "unresolved"
    assert set(audit["methods"]) == {"conditional_natural", "weighted_conditional_ce"}
    assert len(smoke["manifest"]["seed_ledger"]["records"]) == 14
    assert smoke["methods"][1]["training_substream_ledger"]["records"]
    context = smoke["claim_context"]
    threshold = smoke["manifest"]["config"]["ce"]["minimum_target_ess"]
    assert context["ce_target_reached"] == (context["ce_target_ess_history"][-1] >= threshold)
    json.dumps(smoke, allow_nan=False)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("accuracy_z", 1e6),
        ("normalization_z", 1e6),
        ("relative_standard_error", 1e6),
        ("total_wall_seconds", -1.0),
        ("sample_variance", float("nan")),
        ("qualification", "qualified-because-passed"),
    ],
)
def test_stale_or_invalid_display_is_rejected(
    smoke: dict[str, Any], field: str, value: Any
) -> None:
    payload = copy.deepcopy(smoke)
    payload["passed"] = True
    payload["methods"][1]["reported"][field] = value
    assert audit_measurement(payload)["semantic_validity"] == "fail"


def test_stale_comparison_ratio_and_duplicate_keys_are_rejected(smoke: dict[str, Any]) -> None:
    payload = copy.deepcopy(smoke)
    payload["comparisons"][0]["reported_proxy_ratio"] = float("nan")
    assert audit_measurement(payload)["semantic_validity"] == "fail"
    payload = copy.deepcopy(smoke)
    payload["comparisons"][0]["reported_proxy_ratio"] *= 2
    assert audit_measurement(payload)["semantic_validity"] == "fail"
    payload = copy.deepcopy(smoke)
    payload["methods"].append(copy.deepcopy(payload["methods"][0]))
    assert audit_measurement(payload)["semantic_validity"] == "fail"
    payload = copy.deepcopy(smoke)
    payload["methods"][1]["clusters"].append(
        copy.deepcopy(payload["methods"][1]["clusters"][0])
    )
    assert audit_measurement(payload)["semantic_validity"] == "fail"


def test_task_proposal_seed_and_config_mismatch_are_rejected(smoke: dict[str, Any]) -> None:
    for mutation in (
        lambda item: item["manifest"].update(steps=32),
        lambda item: item["methods"][1].update(proposal_digest="0" * 64),
        lambda item: item["methods"][1]["clusters"][0].update(seed=123),
        lambda item: item["methods"][1]["clusters"][0].update(label_seed=123),
        lambda item: item["manifest"]["config"]["task"].update(strike=20.0),
    ):
        payload = copy.deepcopy(smoke)
        mutation(payload)
        assert audit_measurement(payload)["semantic_validity"] == "fail"


def test_bad_statistics_and_likelihood_bound_are_rejected(smoke: dict[str, Any]) -> None:
    for mutation in (
        lambda item: item["methods"][1]["clusters"][0]["moments"].update(count=True),
        lambda item: item["methods"][1]["clusters"][0]["moments"].update(m2=-1.0),
        lambda item: item["methods"][1]["clusters"][0].update(maximum_contribution=100.0),
        lambda item: item["methods"][1]["clusters"][0]["cost"].update(wall_seconds=-1.0),
    ):
        payload = copy.deepcopy(smoke)
        mutation(payload)
        assert audit_measurement(payload)["semantic_validity"] == "fail"


def test_zero_hit_record_remains_unresolved_even_with_zero_se(smoke: dict[str, Any]) -> None:
    payload = copy.deepcopy(smoke)
    natural = payload["methods"][0]
    for cluster in natural["clusters"]:
        cluster["moments"]["mean"] = 0.0
        cluster["moments"]["m2"] = 0.0
        cluster["maximum_contribution"] = 0.0
    reported = natural["reported"]
    reported.update(
        estimate=0.0, sample_variance=0.0, standard_error=0.0,
        relative_standard_error=None,
        accuracy_z=payload["reference"]["reported"]["estimate"]
        / payload["reference"]["reported"]["standard_error"],
        qualification="unresolved",
    )
    payload["comparisons"][0]["reported_proxy_ratio"] = 0.0
    audit = audit_measurement(payload)
    assert audit["semantic_validity"] == "pass"
    assert audit["statistical_evidence"] == "unresolved"


def test_running_moments_merge_matches_direct_tensor() -> None:
    values = torch.tensor((0.0, 1e-8, 3e-8, 1e-4, 0.0, 1e-5), dtype=torch.float64)
    left, right = Moments.from_values(values[:2]), Moments.from_values(values[2:])
    merged = left.merge(right)
    assert merged.mean == pytest.approx(float(torch.mean(values)), rel=1e-12)
    assert merged.sample_variance == pytest.approx(
        float(torch.var(values, unbiased=True)), rel=1e-12
    )


def test_weighted_gaussian_moments_match_truncated_normal_oracle() -> None:
    threshold = 1.0
    shifted_mean = 1.5
    grid = torch.linspace(-7.0, 9.0, 80_001, dtype=torch.float64)
    log_q = -0.5 * (grid - shifted_mean).square()
    log_p_over_q = -0.5 * grid.square() - log_q
    log_weight = torch.where(
        grid >= threshold, log_q + log_p_over_q, torch.full_like(grid, -torch.inf)
    )
    mean, covariance, ess = weighted_gaussian_moments(grid[:, None], log_weight)
    survival = 0.5 * math.erfc(threshold / math.sqrt(2.0))
    mills = math.exp(-0.5 * threshold**2) / math.sqrt(2.0 * math.pi) / survival
    assert float(mean[0]) == pytest.approx(mills, abs=2e-4)
    assert float(covariance[0, 0]) == pytest.approx(1 + threshold * mills - mills**2, abs=2e-4)
    assert ess > 100


def test_conditional_cdf_matches_raw_price_noise_average() -> None:
    problem = RBergomiBaselineProblem(
        task_id="r0-conditional-oracle", task=TerminalThresholdTask(level=80.0),
        spot=100.0, maturity=1.0, steps=4, hurst=0.12,
        eta=1.5, xi=0.04, rho=-0.7,
    )
    count = 30_000
    local = torch.zeros((1, problem.local_dimension), dtype=torch.float64)
    exact = float(evaluate_rbergomi_conditional_terminal(problem, local).payoffs.left_probability[0])
    price = torch.randn(
        (count, problem.steps), dtype=torch.float64,
        generator=torch.Generator().manual_seed(92103),
    )
    full = torch.cat((local.expand(count, -1), price), dim=1)
    observed = float(problem.hard_event(problem.simulate_latent(full)).to(torch.float64).mean())
    assert observed == pytest.approx(exact, abs=0.007)


def test_qualification_rule_applies_same_margin_to_both_sides() -> None:
    policy = QualificationPolicy()
    for name in ("candidate", "comparator"):
        decision = qualify_estimate(
            estimate=0.001, standard_error=0.0005,
            reference_estimate=0.001, reference_standard_error=0.00001,
            reference_independent=True, policy=policy,
        )
        assert decision.status == "unresolved", name
        assert decision.accuracy_z_diagnostic == 0.0


def test_two_sided_selection_radius_uses_four_j_over_gamma() -> None:
    proposal = DefensiveFiniteRankGaussianMixture(
        (FiniteRankGaussianComponent.natural(1),), torch.ones(1, dtype=torch.float64)
    )
    result = select_defensive_proposal_by_second_moment(
        (proposal,), ("natural",),
        lambda x: torch.ones(x.shape[0], dtype=torch.float64),
        sample_count=64, batch_size=32, path_seed=12011, label_seed=12012,
        confidence_delta=0.05,
    )
    expected = 7.0 * math.log(4.0 / 0.05) / (3.0 * 63)
    assert result.estimates[0].simultaneous_empirical_bernstein_radius == pytest.approx(expected)


def test_legacy_adapter_recalculates_ratios_but_does_not_authorize_performance() -> None:
    result = audit_legacy_v16(
        ROOT, ROOT / "configs/g11_v15/v16_final_policy_audit_v1.yaml"
    )
    assert result["integrity"] == "pass"
    assert result["semantic_validity"] == "pass"
    assert result["statistical_evidence"] == "unresolved"
    assert result["performance"] == "unresolved"
    assert len(result["proxy_ratios"]) == 7


def test_strict_loader_rejects_duplicate_and_nonfinite_json(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"value": 1, "value": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        _load_bound_file(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON constant"):
        _load_bound_file(nonfinite)


def test_recursive_binding_detects_mutation_and_escape(tmp_path: Path) -> None:
    child = tmp_path / "child.json"
    child.write_text('{"value": 1}', encoding="utf-8")
    # Bindings hash actual file bytes, not canonicalized JSON objects.
    import hashlib

    digest = hashlib.sha256(child.read_bytes()).hexdigest()
    graph = verify_artifact_graph(tmp_path, {"child": {"path": "child.json", "sha256": digest}})
    assert "child.json" in graph
    child.write_text('{"value": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_artifact_graph(tmp_path, {"child": {"path": "child.json", "sha256": digest}})
    with pytest.raises(ValueError, match="escapes"):
        verify_artifact_graph(
            tmp_path, {"outside": {"path": "../outside.json", "sha256": digest}}
        )


def test_ce_config_refuses_invalid_eigenvalues() -> None:
    with pytest.raises(ValueError, match="eigenvalue bounds"):
        WeightedConditionalCEConfig(minimum_eigenvalue=1.2)
