import numpy as np

from sim.factory.perception import (
    BIN_OF_TYPE,
    DEVELOPMENT_PALETTE,
    DevelopmentColorSorter,
    ForegroundDetector,
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
