import math

import pytest

from experiments.post_audit_dictionary_learning_diagnosis import describe_weights, diagnose


def uniform():
    return [.1]+[.9/372]*372


def test_uniform_metadata_respects_actual_enumeration_and_mass():
    result = describe_weights(uniform())
    assert result["natural_mass"] == .1
    assert result["shifted_conditional_entropy"] == pytest.approx(math.log(372))
    assert result["shifted_effective_components"] == pytest.approx(372)
    assert result["mass_by_template_kind"] == pytest.approx({"point": .9*15/31, "window": .9*16/31})
    assert result["mass_by_vol_amplitude"] == pytest.approx(dict.fromkeys(("2", "4", "6", "8"), .225))
    assert result["mass_by_price_amplitude"] == pytest.approx(dict.fromkeys(("0", "2", "4"), .3))
    assert sum(result["mass_by_time_bin"].values()) == pytest.approx(.9)
    assert result["weighted_squared_mean_norm"] == pytest.approx(.9*(30+20/3))


def test_component_index_mapping_is_vol_outer_price_inner_template_outermost():
    weights = [.1]+[.1/372]*372
    index = 1+12*30+3*3+2
    weights[index] += .8
    result = describe_weights(weights)
    assert result["mass_by_template_kind"]["window"] == pytest.approx(.8+.1*16/31)
    assert result["mass_by_vol_amplitude"]["8"] == pytest.approx(.825)
    assert result["mass_by_price_amplitude"]["4"] == pytest.approx(.8+.1/3)
    assert result["template_masses"][30] == pytest.approx(.8+.1/31)
    assert result["mass_by_time_bin"]["late"] > .8


@pytest.mark.parametrize("weights,steps", [(uniform()[:-1], 32), (uniform(), 64), ([.2]+[.8/372]*372, 32),
    ([.1, float("nan")]+[.9/371]*371, 32)])
def test_wrong_geometry_or_invalid_mass_rejected(weights, steps):
    with pytest.raises(ValueError, match="complete N32"):
        describe_weights(weights, steps)


def test_partial_family_cannot_be_analyzed_as_complete():
    with pytest.raises(ValueError, match="complete audited"):
        diagnose({"fits": [], "rows": [], "heldout_risk": []})
