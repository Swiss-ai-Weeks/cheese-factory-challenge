"""Seeded Omniverse Replicator capture around the canonical factory camera."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import os
import re
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from sim.factory.config import PROJECT_ROOT, load_config
from sim.factory.perception import ForegroundDetector
from sim.factory.replicator_plan import sample_plan
from sim.factory.run_factory import CheeseFactorySample, _wait_frames


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_project_path(value: str) -> Path:
    if value.startswith("/workspace/"):
        return PROJECT_ROOT / value.removeprefix("/workspace/")
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _source_records(manifest_path: Path, limit: int) -> list[dict]:
    by_group = {}
    for row in csv.DictReader(manifest_path.open()):
        if row["split"] not in {"train", "val"}:
            raise ValueError(f"Replicator source has forbidden split: {row['split']}")
        by_group.setdefault(row["group"], row)

    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in by_group.values():
        extra = json.loads(row["extra"] or "{}")
        texture = _resolve_project_path(row["src_path"])
        by_label[row["label"]].append({
            "group": row["group"],
            "label": row["label"],
            "fr_label": extra["fr_label"],
            "split": row["split"],
            "texture": texture,
            "source_sha256": extra["source_sha256"],
        })
    for records in by_label.values():
        records.sort(key=lambda row: row["group"])

    selected = []
    depth = 0
    labels = sorted(by_label)
    while len(selected) < limit:
        added = False
        for label in labels:
            if depth < len(by_label[label]):
                selected.append(by_label[label][depth])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        depth += 1
    if not selected:
        raise ValueError("Replicator source manifest produced no eligible groups")
    for row in selected:
        if not row["texture"].is_file():
            raise FileNotFoundError(row["texture"])
    return selected


def _create_randomized_props(stage):
    import omni.replicator.core as rep
    from pxr import UsdGeom, UsdShade
    from sim.usd_kit import materiau_uni

    rep.functional.create.scope(name="Replicator", parent="/World/Factory")
    clutter = []
    clutter_shaders = []
    for index in range(3):
        prim = rep.functional.create.cube(
            parent="/World/Factory/Replicator", name=f"Clutter{index}"
        )
        material = materiau_uni(
            stage, f"/World/Factory/Replicator/ClutterMaterial{index}",
            (0.25, 0.32, 0.38), 0.55, 0.05,
        )
        UsdShade.MaterialBindingAPI(prim).Bind(material)
        clutter.append(prim)
        clutter_shaders.append(material.GetPrim().GetChild("Shader"))
    occluder = rep.functional.create.cube(
        parent="/World/Factory/Replicator", name="EdgeOccluder"
    )
    occluder_material = materiau_uni(
        stage, "/World/Factory/Replicator/OccluderMaterial", (0.08, 0.10, 0.12), 0.8, 0.1,
    )
    UsdShade.MaterialBindingAPI(occluder).Bind(occluder_material)
    UsdGeom.Imageable(occluder).MakeInvisible()
    return clutter, clutter_shaders, occluder


def _apply_plan(rep, stage, scene, plan, clutter, clutter_shaders, occluder) -> None:
    from isaacsim.core.rendering_manager import ViewportManager
    from pxr import UsdGeom

    rep.functional.modify.pose(
        scene._carrier_root.GetPrim(),
        position_value=plan.object_position,
        rotation_value=(0.0, 0.0, plan.object_rotation_deg),
    )
    camera = stage.GetPrimAtPath(scene.config.section("camera")["prim_path"])
    # Use the same camera-axis-aware API as canonical scene setup. Replicator's
    # generic pose helper resets this existing CameraSensor prim to a different
    # axis convention, even when only a position is supplied.
    ViewportManager.set_camera_view(
        scene.camera.camera.paths[0],
        eye=plan.camera_position,
        target=(
            plan.camera_position[0],
            plan.camera_position[1],
            float(scene.config.section("belt")["plane_z"]),
        ),
    )
    # RtxCamera's public focal-length API uses centimetre-scaled values and
    # converts them to the USD millimetre attribute internally.  Replicator
    # writes that USD attribute directly, so preserve the same x10 conversion.
    rep.functional.modify.attribute(camera, "focalLength", plan.focal_length_mm * 10.0)

    for path, intensity in (
        ("/World/Factory/KeyLight", plan.key_intensity),
        ("/World/Factory/Inspection/left_light", plan.work_light_intensity),
        ("/World/Factory/Inspection/right_light", plan.work_light_intensity),
        ("/World/Factory/DomeLight", plan.dome_intensity),
    ):
        rep.functional.modify.attribute(stage.GetPrimAtPath(path), "inputs:intensity", intensity)
    rep.functional.modify.attribute(
        stage.GetPrimAtPath("/World/Factory/KeyLight"), "inputs:color", plan.key_color,
    )
    rep.functional.modify.attribute(
        stage.GetPrimAtPath("/World/FactoryMaterials/Floor/Shader"),
        "inputs:diffuseColor", plan.floor_color,
    )
    plate_shader = stage.GetPrimAtPath("/World/InspectionCarrier/PlateMaterial/Shader")
    rep.functional.modify.attribute(plate_shader, "inputs:diffuseColor", plan.plate_color)
    rep.functional.modify.attribute(plate_shader, "inputs:roughness", plan.plate_roughness)

    for index, prim in enumerate(clutter):
        rep.functional.modify.pose(
            prim,
            position_value=plan.clutter_positions[index],
            scale_value=plan.clutter_scales[index],
            rotation_value=(0.0, 0.0, float((plan.object_rotation_deg + 47 * index) % 360)),
        )
        rep.functional.modify.attribute(
            clutter_shaders[index], "inputs:diffuseColor", plan.clutter_colors[index],
        )
    rep.functional.modify.pose(
        occluder,
        position_value=plan.occluder_position,
        scale_value=plan.occluder_scale,
        rotation_value=(0.0, 0.0, float((plan.object_rotation_deg + 23.0) % 360)),
    )
    imageable = UsdGeom.Imageable(occluder)
    (imageable.MakeVisible if plan.occluder_enabled else imageable.MakeInvisible)()


async def capture() -> None:
    import isaacsim.core.experimental.utils.app as app_utils
    import omni.replicator.core as rep
    import omni.usd
    from pxr import UsdGeom

    dataset_name = os.environ.get("CHEESE_REPLICATOR_DATASET", "factory_replicator_v1")
    if not re.fullmatch(r"[a-z0-9_]+", dataset_name):
        raise ValueError("CHEESE_REPLICATOR_DATASET must match [a-z0-9_]+")
    seed = int(os.environ.get("CHEESE_REPLICATOR_SEED", "2026"))
    views = int(os.environ.get("CHEESE_REPLICATOR_VIEWS", "3"))
    limit = int(os.environ.get("CHEESE_REPLICATOR_LIMIT", "6"))
    subframes = int(os.environ.get("CHEESE_REPLICATOR_SUBFRAMES", "8"))
    source_value = os.environ.get(
        "CHEESE_REPLICATOR_SOURCE_MANIFEST",
        "data/processed/manifest_factory_balanced_v2.csv",
    )
    source_manifest = _resolve_project_path(source_value)
    records = _source_records(source_manifest, limit)

    config = load_config()
    sample = CheeseFactorySample(config, "model")
    scene = sample.scene
    await sample.load_world_async()
    await sample.reset_async()
    app_utils.play(commit=True)
    stage = omni.usd.get_context().get_stage()

    camera = config.section("camera")
    clutter, clutter_shaders, occluder = _create_randomized_props(stage)
    rep.set_global_seed(seed)
    generator = rep.rng.ReplicatorRNG(seed=seed).generator

    focal_mm = 2.0955 / (
        2.0 * np.tan(np.deg2rad(float(camera["horizontal_fov_deg"])) / 2.0)
    )
    perception = config.section("perception")
    detector = ForegroundDetector(
        threshold=perception["background_threshold"],
        minimum_area_px=perception["minimum_area_px"],
        padding_px=perception["crop_padding_px"],
        roi=camera.get("roi"),
    )

    output_root = PROJECT_ROOT / "data" / "processed" / "images" / dataset_name
    full_root = PROJECT_ROOT / "outputs" / dataset_name / "full"
    manifest_path = PROJECT_ROOT / "data" / "processed" / f"manifest_{dataset_name}.csv"
    output_root.mkdir(parents=True, exist_ok=True)
    full_root.mkdir(parents=True, exist_ok=True)
    camera_config_hash = _sha256(config.source)
    rows = []
    frame_index = 0
    for source in records:
        for view in range(views):
            plan = sample_plan(generator, frame_index, focal_mm)
            scene.set_object_texture(
                f"replicator-{source['group']}-{view}",
                source["fr_label"],
                plan.object_position,
                source["texture"],
                size_fraction=plan.object_size_fraction,
                rotation_deg=0.0,
            )
            _apply_plan(rep, stage, scene, plan, clutter, clutter_shaders, occluder)

            carrier = UsdGeom.Imageable(scene._carrier_root)
            carrier.MakeInvisible()
            background, _ = await _wait_frames(scene, app_utils, subframes)
            detector.set_background(background)
            carrier.MakeVisible()
            frame, _ = await _wait_frames(scene, app_utils, subframes)
            detection = detector.detect(frame)
            if detection is None:
                debug_root = PROJECT_ROOT / "outputs" / dataset_name / "debug"
                debug_root.mkdir(parents=True, exist_ok=True)
                Image.fromarray(background).save(debug_root / f"frame-{frame_index:04d}-background.jpg")
                Image.fromarray(frame).save(debug_root / f"frame-{frame_index:04d}-foreground.jpg")
                raise RuntimeError(
                    f"no foreground for group={source['group']} view={view} frame={frame_index}"
                )

            uid = f"rep__{source['group']}__v{view}"
            destination = output_root / source["label"] / f"{uid}.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(detection.crop).save(destination, "JPEG", quality=95, subsampling=0)
            full_path = full_root / f"{uid}.jpg"
            Image.fromarray(frame).save(full_path, "JPEG", quality=92, subsampling=0)
            plan_data = plan.as_dict()
            plan_json = json.dumps(plan_data, sort_keys=True, separators=(",", ":"))
            plan_hash = hashlib.sha256(plan_json.encode()).hexdigest()
            extra = {
                "camera_config_sha256": camera_config_hash,
                "capture_domain": "canonical_factory_replicator",
                "fr_label": source["fr_label"],
                "full_frame_path": str(full_path.relative_to(PROJECT_ROOT)),
                "full_frame_sha256": _sha256(full_path),
                "replicator_extension": "omni.replicator.core-1.13.36",
                "replicator_plan": plan_data,
                "replicator_plan_sha256": plan_hash,
                "seed": seed,
                "source_sha256": source["source_sha256"],
                "view": view,
            }
            rows.append({
                "uid": uid,
                "source": "sim_belt",
                "task": "sim_bin",
                "label": source["label"],
                "split": source["split"],
                "path": str(destination.relative_to(PROJECT_ROOT / "data" / "processed")),
                "width": detection.crop.shape[1],
                "height": detection.crop.shape[0],
                "orig_width": frame.shape[1],
                "orig_height": frame.shape[0],
                "content_hash": _sha256(destination),
                "src_path": str(source["texture"]),
                "crop_box": json.dumps(detection.box),
                "group": source["group"],
                "split_label": source["label"],
                "extra": json.dumps(extra, sort_keys=True),
            })
            print(
                f"REPLICATOR_CAPTURE frame={frame_index} group={source['group']} "
                f"view={view} occluder={plan.occluder_enabled}",
                flush=True,
            )
            frame_index += 1

    fields = list(rows[0])
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"REPLICATOR_CAPTURE_DONE seed={seed} sources={len(records)} "
        f"views={views} rows={len(rows)} manifest={manifest_path}",
        flush=True,
    )


async def _start() -> None:
    try:
        await capture()
    except Exception as exc:
        print(f"REPLICATOR_CAPTURE_FATAL {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        os._exit(1)
    os._exit(0)


asyncio.ensure_future(_start())
