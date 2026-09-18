import pytest

from sim.factory.hud import HudSnapshot, MODE_BANNERS


def test_every_runtime_mode_has_an_unambiguous_banner():
    assert set(MODE_BANNERS) == {"model", "development", "showcase"}
    assert "TRAINED PERCEPTION" in MODE_BANNERS["model"]
    assert "NOT MODEL ACCURACY" in MODE_BANNERS["showcase"]


def test_hud_progress_is_compact_and_complete():
    snapshot = HudSnapshot(
        mode="showcase",
        scenario="judge-demo",
        completed=7,
        total=11,
        successful=6,
        rejected=1,
        failures=0,
    )
    assert snapshot.mode_banner == MODE_BANNERS["showcase"]
    assert snapshot.progress_text == "7/11 processed  ·  6 successful  ·  1 safe rejects  ·  0 faults"


def test_unknown_mode_cannot_be_presented_without_a_label():
    with pytest.raises(KeyError):
        _ = HudSnapshot(mode="mystery", scenario="x").mode_banner
