import pytest

from src.predict import HYBRID_DECISION_POLICY, resolve_hybrid_decision


def resolve(route, type_label, confidence=0.9, threshold=0.55):
    return resolve_hybrid_decision(route, confidence, type_label, threshold)


@pytest.mark.parametrize(
    ("route", "type_label"),
    [
        ("bin_hard", "hard_cheese"),
        ("bin_semi_hard", "raclette_cheese"),
        ("bin_soft", "soft_cheese"),
        ("bin_fresh", "fresh_cheese"),
        ("bin_blue", "blue_mould_cheese"),
    ],
)
def test_matching_type_and_route_are_actionable_for_every_bin(route, type_label):
    status, destination, implied, agreement, reason = resolve(route, type_label)
    assert (status, destination, implied) == ("ok", route, route)
    assert agreement
    assert reason == "models_agree"
    assert HYBRID_DECISION_POLICY == "route_authoritative_fail_closed_v1"


def test_cross_bin_disagreement_fails_closed_with_reason():
    status, destination, implied, agreement, reason = resolve("bin_hard", "blue_mould_cheese")
    assert (status, destination, implied) == ("uncertain", None, "bin_blue")
    assert not agreement
    assert reason == "cross_model_destination_conflict"


@pytest.mark.parametrize("route", ["empty", "not_cheese"])
def test_routing_reject_is_authoritative_but_disagreement_is_preserved(route):
    status, destination, implied, agreement, reason = resolve(route, "hard_cheese")
    assert status == route
    assert destination is None
    assert implied == "bin_hard"
    assert not agreement
    assert reason == "routing_reject_authoritative"


def test_low_confidence_route_is_never_actionable():
    status, destination, implied, agreement, reason = resolve(
        "bin_hard", "hard_cheese", confidence=0.54
    )
    assert (status, destination, implied, agreement) == ("uncertain", None, "bin_hard", True)
    assert reason == "route_below_confidence_threshold"


def test_unknown_route_is_rejected():
    with pytest.raises(ValueError, match="route inconnue"):
        resolve("bin_mystery", "hard_cheese")
