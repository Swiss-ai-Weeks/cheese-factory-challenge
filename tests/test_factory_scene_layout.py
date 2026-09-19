import json

import pytest

from sim.factory.scene_layout import load_scene_layout


def test_checked_in_scene_layout_is_valid_and_human_named():
    layout = load_scene_layout()
    paths = [item["path"] for item in layout.boxes + layout.lights]

    assert layout.raw["schema_version"] == 1
    assert len(paths) == len(set(paths))
    assert "/World/Factory/Inspection/left_post" in paths
    assert layout.overview_camera["path"] == "/World/OverviewCamera"
    assert all(item["role"] for item in layout.boxes + layout.lights)


def test_scene_layout_rejects_unknown_material(tmp_path):
    document = json.loads(load_scene_layout().source.read_text(encoding="utf-8"))
    document["boxes"][0]["material"] = "does_not_exist"
    path = tmp_path / "bad-layout.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown material"):
        load_scene_layout(path)


def test_scene_layout_rejects_duplicate_prim_path(tmp_path):
    document = json.loads(load_scene_layout().source.read_text(encoding="utf-8"))
    document["boxes"].append(dict(document["boxes"][0]))
    path = tmp_path / "duplicate-layout.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate scene prim path"):
        load_scene_layout(path)
