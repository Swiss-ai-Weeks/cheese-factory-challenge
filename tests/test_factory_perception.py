import numpy as np
import json
from io import BytesIO
from PIL import Image

from sim.factory.perception import (
    BIN_OF_TYPE,
    DEVELOPMENT_PALETTE,
    DevelopmentColorSorter,
    ForegroundDetector,
    RemoteModelSorter,
    showcase_sort_result,
)


def test_detector_extracts_crop_and_empty_frame_is_safe():
    background = np.full((120, 160, 3), 50, dtype=np.uint8)
    frame = background.copy()
    frame[45:75, 65:95] = DEVELOPMENT_PALETTE["hard_cheese"]
    detector = ForegroundDetector(20, 100, 4, [0, 0, 160, 120])
    detector.set_background(background)
    assert detector.detect(background) is None
    detection = detector.detect(frame)
    assert detection is not None
    assert detection.area_px == 900
    assert detection.crop.shape[0] > 30


def test_development_classifier_uses_pixels_and_maps_all_bins():
    sorter = DevelopmentColorSorter(min_confidence=0.50)
    observed_bins = set()
    for cheese_type, target in BIN_OF_TYPE.items():
        authored = np.asarray(DEVELOPMENT_PALETTE[cheese_type], dtype=np.float64) / 255.0
        color = np.rint(255.0 * np.power(authored, 1.0 / 2.2)).astype(np.uint8)
        crop = np.full((48, 48, 3), color, dtype=np.uint8)
        result = sorter.predict(crop)
        assert result.status == "ok"
        assert result.cheese_type == cheese_type
        assert result.bin == target
        observed_bins.add(result.bin)
    assert len(observed_bins) == 5


def test_foreign_object_never_gets_cheese_bin():
    sorter = DevelopmentColorSorter(min_confidence=0.50)
    authored = np.asarray(DEVELOPMENT_PALETTE["not_cheese"], dtype=np.float64) / 255.0
    color = np.rint(255.0 * np.power(authored, 1.0 / 2.2)).astype(np.uint8)
    crop = np.full((48, 48, 3), color, dtype=np.uint8)
    result = sorter.predict(crop)
    assert result.status == "not_cheese"
    assert result.bin is None
    assert not result.actionable


def test_showcase_routes_every_bin_and_reject_without_claiming_latency():
    observed_bins = {showcase_sort_result(label).bin for label in BIN_OF_TYPE}
    assert observed_bins == set(BIN_OF_TYPE.values())
    rejected = showcase_sort_result("not_cheese")
    assert rejected.status == "not_cheese"
    assert rejected.bin is None
    assert rejected.latency_ms == 0.0


def test_showcase_rejects_unknown_scenario_label():
    try:
        showcase_sort_result("mystery_cheese")
    except ValueError as exc:
        assert "no route" in str(exc)
    else:
        raise AssertionError("unknown showcase label was accepted")


def test_remote_sorter_checks_health_and_posts_pixels(monkeypatch):
    calls = []

    class Response:
        def __init__(self, payload):
            self.payload = json.dumps(payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return self.payload

    def fake_urlopen(request, timeout):
        url = request if isinstance(request, str) else request.full_url
        calls.append((url, timeout))
        if url.endswith("/health"):
            return Response({
                "ok": True,
                "types": list(BIN_OF_TYPE) + ["empty", "not_cheese"],
                "contract_version": 2,
                "decision_policy": "route_authoritative_fail_closed_v1",
            })
        posted = Image.open(BytesIO(request.data))
        assert posted.size == (12, 10)
        return Response({
            "status": "ok",
            "bin": "bin_hard",
            "bin_confidence": 0.91,
            "cheese_type": "hard_cheese",
            "type_confidence": 0.82,
            "topk_types": [["hard_cheese", 0.82]],
            "latency_ms": 8.4,
            "route_label": "bin_hard",
            "type_implied_bin": "bin_hard",
            "decision_policy": "route_authoritative_fail_closed_v1",
            "agreement": True,
            "decision_reason": "models_agree",
        })

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    sorter = RemoteModelSorter("http://sorter:8765", timeout_s=12)
    result = sorter.predict(np.full((10, 12, 3), 127, dtype=np.uint8))
    assert result.actionable
    assert result.bin == "bin_hard"
    assert calls[0][0].endswith("/health")
    assert calls[1][0].endswith("/predict")


def test_remote_sorter_rejects_unsafe_contract(monkeypatch):
    class Response:
        def __init__(self, payload):
            self.payload = json.dumps(payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return self.payload

    responses = iter([
        {
            "ok": True,
            "types": list(BIN_OF_TYPE) + ["empty", "not_cheese"],
            "contract_version": 2,
            "decision_policy": "route_authoritative_fail_closed_v1",
        },
        {
            "status": "not_cheese",
            "bin": "bin_hard",
            "bin_confidence": 0.9,
            "cheese_type": "not_cheese",
            "type_confidence": 0.9,
            "latency_ms": 5.0,
            "route_label": "not_cheese",
            "type_implied_bin": None,
            "decision_policy": "route_authoritative_fail_closed_v1",
            "agreement": True,
            "decision_reason": "routing_reject_authoritative",
        },
    ])
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response(next(responses)))
    sorter = RemoteModelSorter("http://sorter:8765")
    try:
        sorter.predict(np.zeros((8, 8, 3), dtype=np.uint8))
    except RuntimeError as exc:
        assert "unsafe perception decision contract" in str(exc)
    else:
        raise AssertionError("unsafe service response was accepted")


def test_remote_sorter_rejects_actionable_cross_model_conflict(monkeypatch):
    class Response:
        def __init__(self, payload):
            self.payload = json.dumps(payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return self.payload

    responses = iter([
        {
            "ok": True,
            "types": list(BIN_OF_TYPE) + ["empty", "not_cheese"],
            "contract_version": 2,
            "decision_policy": "route_authoritative_fail_closed_v1",
        },
        {
            "status": "ok",
            "bin": "bin_hard",
            "bin_confidence": 0.91,
            "cheese_type": "blue_mould_cheese",
            "type_confidence": 0.82,
            "topk_types": [["blue_mould_cheese", 0.82]],
            "latency_ms": 5.0,
            "route_label": "bin_hard",
            "type_implied_bin": "bin_blue",
            "decision_policy": "route_authoritative_fail_closed_v1",
            "agreement": False,
            "decision_reason": "cross_model_destination_conflict",
        },
    ])
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response(next(responses)))
    sorter = RemoteModelSorter("http://sorter:8765")

    try:
        sorter.predict(np.zeros((8, 8, 3), dtype=np.uint8))
    except RuntimeError as exc:
        assert "actionable cross-model conflict" in str(exc)
    else:
        raise AssertionError("actionable route/type conflict was accepted")
