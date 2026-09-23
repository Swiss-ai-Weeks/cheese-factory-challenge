import pytest
from types import SimpleNamespace

from sim.factory.hud import HudSnapshot, MODE_BANNERS, configure_presentation_workspace, decision_details


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
    assert "ARM INHIBITED" in snapshot.safety


def test_unknown_mode_cannot_be_presented_without_a_label():
    with pytest.raises(KeyError):
        _ = HudSnapshot(mode="mystery", scenario="x").mode_banner


@pytest.mark.parametrize("mode", ["showcase", "development"])
def test_non_model_modes_never_show_fake_probability_bars_or_latency(mode):
    result = SimpleNamespace(
        topk_types=[("raclette_cheese", 1.0)], cheese_type="raclette_cheese",
        bin="bin_semi_hard", latency_ms=0.0, agreement=True,
    )
    details = decision_details(mode, result)
    assert details["topk"] == ()
    assert details["inference_ms"] is None
    assert "inference" in details["decision_note"]


def test_model_scores_are_real_and_disagreement_is_visible():
    result = SimpleNamespace(
        topk_types=[("raclette_cheese", 0.81), ("hard_cheese", 0.12), ("bad", float("nan"))],
        cheese_type="raclette_cheese", bin=None, latency_ms=32.4, agreement=False,
    )
    details = decision_details("model", result)
    assert details["topk"] == (("raclette cheese", 0.81), ("hard cheese", 0.12))
    assert details["inference_ms"] == 32.4
    assert details["route_key"] == "reject"
    assert "disagreement" in details["decision_note"]


def test_output_counts_only_completed_placements_and_diversions():
    initial = HudSnapshot(mode="showcase", scenario="x")
    failed = initial.with_outcome("bin_hard", placed=False, diverted=False)
    assert failed.output_counts == (0, 0, 0, 0, 0, 0)
    placed = failed.with_outcome("bin_fresh", placed=True, diverted=False)
    rejected = placed.with_outcome(None, placed=False, diverted=True)
    assert rejected.output_counts == (0, 0, 0, 1, 0, 1)
    assert initial.output_counts == (0, 0, 0, 0, 0, 0)


@pytest.mark.parametrize("completed,total,expected", [(0, 0, 0), (3, 10, 0.3), (12, 11, 1)])
def test_progress_stays_bounded(completed, total, expected):
    assert HudSnapshot(mode="model", scenario="x", completed=completed, total=total).progress == expected


class _Window:
    def __init__(self, visible=True):
        self.visible = visible


class _Workspace:
    windows = {
        "Stage": _Window(),
        "Property": _Window(),
        "Content": _Window(False),
        "Viewport": _Window(),
    }

    @classmethod
    def get_window(cls, title):
        return cls.windows.get(title)


def test_presentation_workspace_hides_only_visible_editor_panels():
    hidden = configure_presentation_workspace(_Workspace)

    assert hidden == ("Stage", "Property")
    assert not _Workspace.windows["Stage"].visible
    assert not _Workspace.windows["Property"].visible
    assert _Workspace.windows["Viewport"].visible
