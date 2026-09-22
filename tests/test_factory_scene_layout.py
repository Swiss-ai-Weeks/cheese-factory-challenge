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


def test_inspection_gantry_is_wide_collidable_and_registered_for_planning():
    layout = load_scene_layout()
    boxes = {item["path"]: item for item in layout.boxes}
    left = boxes["/World/Factory/Inspection/left_post"]
    right = boxes["/World/Factory/Inspection/right_post"]
    beam = boxes["/World/Factory/Inspection/crossbeam"]

    assert left["position"][0] <= -1.0
    assert right["position"][0] >= 1.75
    assert beam["position"][2] - beam["size"][2] / 2.0 >= 1.70
    assert beam["size"][0] >= right["position"][0] - left["position"][0]
    assert layout.planning_obstacle_paths == (
        "/World/Factory/PlanningObstacles/Inspection/left_post",
        "/World/Factory/PlanningObstacles/Inspection/right_post",
        "/World/Factory/PlanningObstacles/Inspection/crossbeam",
    )
    for declaration in (left, right, beam):
        assert declaration["collision"] == "static"
        assert declaration["collision_path"] in layout.planning_obstacle_paths


def test_planning_obstacle_requires_static_collision(tmp_path):
    document = json.loads(load_scene_layout().source.read_text(encoding="utf-8"))
    document["boxes"][0]["planning_obstacle"] = True
    path = tmp_path / "unsafe-layout.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="requires collision='static'"):
        load_scene_layout(path)


def test_collision_proxy_requires_static_collision(tmp_path):
    document = json.loads(load_scene_layout().source.read_text(encoding="utf-8"))
    document["boxes"][0]["collision_path"] = "/World/Factory/PlanningObstacles/bad"
    path = tmp_path / "unsafe-proxy-layout.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="collision_path requires collision='static'"):
        load_scene_layout(path)


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
