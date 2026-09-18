from pathlib import Path
from types import SimpleNamespace
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from predict import HybridCheeseSorter


class _Predictor:
    def __init__(self, result):
        self.result = result

    def predict(self, frame, box=None):
        return self.result


def _sorter(route_label, route_confidence, threshold=0.55):
    sorter = object.__new__(HybridCheeseSorter)
    sorter.min_confidence = threshold
    sorter._types = _Predictor(SimpleNamespace(
        cheese_type="blue_mould_cheese",
        type_confidence=0.62,
        topk_types=[["blue_mould_cheese", 0.62]],
    ))
    sorter._routing = _Predictor(SimpleNamespace(
        label=route_label,
        confidence=route_confidence,
    ))
    return sorter


def test_hybrid_sorter_uses_direct_bin_for_action():
    result = _sorter("bin_blue", 0.91).predict(object())

    assert result.status == "ok"
    assert result.bin == "bin_blue"
    assert result.cheese_type == "blue_mould_cheese"


def test_hybrid_sorter_fails_closed_below_threshold():
    result = _sorter("bin_blue", 0.49).predict(object())

    assert result.status == "uncertain"
    assert result.bin is None


def test_hybrid_sorter_rejects_non_cheese_without_actuation():
    result = _sorter("not_cheese", 0.88).predict(object())

    assert result.status == "not_cheese"
    assert result.bin is None
