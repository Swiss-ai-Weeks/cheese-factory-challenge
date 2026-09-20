from dataclasses import replace

import numpy as np
import pytest

from sim.factory.perception import DevelopmentSortResult
from sim.factory.timing import (
    DecisionValidationError,
    ObservationContext,
    parse_request_headers,
    validate_decision,
)


def _observation():
    return ObservationContext.capture("object-007", 8, np.zeros((4, 4, 3), dtype=np.uint8))


def _result(observation):
    return DevelopmentSortResult(
        "ok",
        "bin_hard",
        0.9,
        "hard_cheese",
        0.9,
        [],
        5.0,
        item_id=observation.item_id,
        observation_sequence=observation.sequence,
        request_id=observation.request_id,
        frame_sha256=observation.frame_sha256,
        observed_at_epoch=observation.captured_at_epoch,
        server_received_at_epoch=observation.captured_at_epoch + 0.002,
        decision_at_epoch=observation.captured_at_epoch + 0.005,
    )


def test_normal_decision_is_traceable_to_exact_observation():
    observation = _observation()
    timing = validate_decision(
        observation,
        _result(observation),
        current_item_id="object-007",
        max_age_s=1.0,
        now_monotonic=observation.captured_at_monotonic + 0.020,
        now_epoch=observation.captured_at_epoch + 0.020,
    )

    assert timing.item_id == "object-007"
    assert timing.sequence == 8
    assert timing.frame_sha256 == observation.frame_sha256
    assert timing.age_ms == pytest.approx(20.0)


@pytest.mark.parametrize(
    "field,value,code",
    [
        ("item_id", "object-008", "item_mismatch"),
        ("observation_sequence", 9, "sequence_mismatch"),
        ("request_id", "replayed", "request_mismatch"),
        ("frame_sha256", "0" * 64, "frame_mismatch"),
        ("observed_at_epoch", 1.0, "timestamp_mismatch"),
    ],
)
def test_mismatched_or_replayed_decision_fails_closed(field, value, code):
    observation = _observation()
    with pytest.raises(DecisionValidationError) as caught:
        validate_decision(
            observation,
            replace(_result(observation), **{field: value}),
            current_item_id="object-007",
            max_age_s=1.0,
            now_monotonic=observation.captured_at_monotonic + 0.020,
            now_epoch=observation.captured_at_epoch + 0.020,
        )
    assert caught.value.code == code


def test_missing_delayed_and_future_decisions_fail_closed():
    observation = _observation()
    with pytest.raises(DecisionValidationError) as missing:
        validate_decision(
            observation,
            DevelopmentSortResult("uncertain", None, 0.0, "empty", 0.0, [], 0.0),
            current_item_id="object-007",
            max_age_s=1.0,
        )
    assert missing.value.code == "missing_metadata"

    with pytest.raises(DecisionValidationError) as delayed:
        validate_decision(
            observation,
            _result(observation),
            current_item_id="object-007",
            max_age_s=1.0,
            now_monotonic=observation.captured_at_monotonic + 1.001,
            now_epoch=observation.captured_at_epoch + 1.001,
        )
    assert delayed.value.code == "decision_timeout"

    future = replace(
        _result(observation),
        server_received_at_epoch=observation.captured_at_epoch + 2.0,
        decision_at_epoch=observation.captured_at_epoch + 3.0,
    )
    with pytest.raises(DecisionValidationError) as future_error:
        validate_decision(
            observation,
            future,
            current_item_id="object-007",
            max_age_s=5.0,
            now_monotonic=observation.captured_at_monotonic + 0.020,
            now_epoch=observation.captured_at_epoch + 0.020,
        )
    assert future_error.value.code == "future_timestamp"


def test_service_request_headers_round_trip_and_reject_malformed_values():
    observation = _observation()
    parsed = parse_request_headers(observation.request_headers())
    assert parsed["item_id"] == observation.item_id
    assert parsed["observation_sequence"] == observation.sequence
    assert parsed["frame_sha256"] == observation.frame_sha256

    malformed = observation.request_headers()
    malformed["X-Cheese-Frame-SHA256"] = "not-a-digest"
    with pytest.raises(ValueError, match="frame digest"):
        parse_request_headers(malformed)


def test_local_showcase_decision_carries_the_same_correlation_envelope():
    from sim.factory.perception import showcase_sort_result

    observation = _observation()
    result = showcase_sort_result("hard_cheese", observation)

    assert result.item_id == observation.item_id
    assert result.request_id == observation.request_id
    assert result.frame_sha256 == observation.frame_sha256
    validate_decision(
        observation,
        result,
        current_item_id=observation.item_id,
        max_age_s=1.0,
        now_monotonic=observation.captured_at_monotonic + 0.020,
        now_epoch=max(result.decision_at_epoch, observation.captured_at_epoch + 0.020),
    )
