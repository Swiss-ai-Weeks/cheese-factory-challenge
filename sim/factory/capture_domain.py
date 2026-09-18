"""Capture leak-free factory-camera crops for sim-to-sim adaptation.

The source cutout group retains the split assigned by ``manifest_sim.csv``.
Training and validation captures therefore never reuse a test object.
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import sys
import traceback
from pathlib import Path

from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from sim.factory.capture_filters import VALID_BINS, VALID_SPLITS, filter_records, selection
from sim.factory.config import PROJECT_ROOT, load_config
from sim.factory.perception import ForegroundDetector
from sim.factory.run_factory import CheeseFactorySample, _wait_frames


def _records() -> list[dict]:
    processed = PROJECT_ROOT / "data" / "processed"
    group_split = {}
    for row in csv.DictReader((processed / "manifest_sim.csv").open()):
        group_split[row["group"]] = row["split"]
    records = []
    for row in csv.DictReader((processed / "cutouts" / "manifest.csv").open()):
        split = group_split.get(row["uid"])
        if split is None:
            continue
        label = "not_cheese" if row["bin"] == "not_cheese" else row["label"]
        records.append({
            "uid": row["uid"],
            "label": label,
            "bin": row["bin"],
            "split": split,
            "texture": processed / row["path"],
        })
    return sorted(records, key=lambda row: (row["split"], row["bin"], row["uid"]))


async def capture(
    views: int = 2,
    offset: int = 0,
    limit: int = 100,
    view_start: int = 0,
) -> None:
    import isaacsim.core.experimental.utils.app as app_utils

    config = load_config()
    sample = CheeseFactorySample(config, "model")
    scene = sample.scene
    output = PROJECT_ROOT / "data" / "processed" / "images" / "factory_adapt"
    manifest_path = PROJECT_ROOT / "data" / "processed" / "manifest_factory_adapt.csv"
    factory_manifest_path = PROJECT_ROOT / "data" / "processed" / "manifest_factory_only.csv"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        await sample.load_world_async()
        await sample.reset_async()
        app_utils.play(commit=True)
        spawn = tuple(float(v) for v in config.section("belt")["spawn_position"])
        scene.set_object("capture-background", "not_cheese", spawn)
        background, _ = await _wait_frames(scene, app_utils, int(config.section("camera")["warmup_frames"]))
        perception = config.section("perception")
        detector = ForegroundDetector(
            threshold=perception["background_threshold"],
            minimum_area_px=perception["minimum_area_px"],
            padding_px=perception["crop_padding_px"],
            roi=config.section("camera").get("roi"),
        )
        detector.set_background(background)
        pick = list(spawn)
        pick[1] = float(config.section("belt")["pick_line_y"])
        all_source_records = _records()
        selected_splits = selection("CHEESE_CAPTURE_SPLITS", VALID_SPLITS)
        selected_bins = selection("CHEESE_CAPTURE_BINS", VALID_BINS)
        eligible_records = filter_records(
            all_source_records, selected_splits, selected_bins,
        )
        source_records = eligible_records[offset:offset + limit]
        print(
            f"FACTORY_CAPTURE total_sources={len(all_source_records)} "
            f"eligible_sources={len(eligible_records)} views={views} "
            f"view_start={view_start} "
            f"offset={offset} batch={len(source_records)}",
            flush=True,
        )
        for index, source in enumerate(source_records, start=offset):
            for view in range(view_start, view_start + views):
                size_fraction = (0.52, 0.72, 0.65)[view % 3]
                rotation = 0.0 if view % 3 == 2 else float((37 * view + 19 * index) % 360)
                scene.set_object_texture(
                    f"capture-{source['uid']}-{view}",
                    source["label"],
                    tuple(pick),
                    source["texture"],
                    size_fraction=size_fraction,
                    rotation_deg=rotation,
                )
                frame, _ = await _wait_frames(scene, app_utils, 3)
                detection = detector.detect(frame)
                if detection is None:
                    print(f"FACTORY_CAPTURE_MISS uid={source['uid']} view={view}", flush=True)
                    continue
                destination = output / source["bin"] / f"{source['uid']}__v{view}.jpg"
                destination.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(detection.crop).save(destination, "JPEG", quality=95, subsampling=0)
                rows.append({
                    "uid": f"factory__{source['uid']}__v{view}",
                    "source": "sim_belt",
                    "task": "sim_bin",
                    "label": source["bin"],
                    "split": source["split"],
                    "path": str(destination.relative_to(PROJECT_ROOT / "data" / "processed")),
                    "width": detection.crop.shape[1],
                    "height": detection.crop.shape[0],
                    "orig_width": detection.crop.shape[1],
                    "orig_height": detection.crop.shape[0],
                    "content_hash": "",
                    "src_path": str(source["texture"]),
                    "crop_box": json.dumps(detection.box),
                    "group": source["uid"],
                    "split_label": source["bin"],
                    "extra": json.dumps({"fr_label": source["label"], "domain": "factory_camera"}),
                })
            if (index - offset + 1) % 50 == 0:
                print(
                    f"FACTORY_CAPTURE_PROGRESS {index - offset + 1}/{len(source_records)} "
                    f"offset={offset} images={len(rows)}",
                    flush=True,
                )
        fields = [
            "uid", "source", "task", "label", "split", "path", "width", "height",
            "orig_width", "orig_height", "content_hash", "src_path", "crop_box",
            "group", "split_label", "extra",
        ]
        base_rows = list(csv.DictReader(
            (PROJECT_ROOT / "data" / "processed" / "manifest_sim.csv").open()
        ))
        prior_rows = []
        if manifest_path.stat().st_size:
            prior_rows = [
                row for row in csv.DictReader(manifest_path.open())
                if row["uid"].startswith("factory__")
            ]
        merged = {row["uid"]: row for row in prior_rows}
        merged.update({row["uid"]: row for row in rows})
        with manifest_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(base_rows + [merged[uid] for uid in sorted(merged)])
        with factory_manifest_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(merged[uid] for uid in sorted(merged))
        print(
            f"FACTORY_CAPTURE_DONE batch_captures={len(rows)} captures={len(merged)} "
            f"total={len(base_rows) + len(merged)} "
            f"manifest={manifest_path} factory_manifest={factory_manifest_path}",
            flush=True,
        )
    finally:
        # This is a one-shot Kit process. Isaac Sim 6.1 can segfault inside the
        # experimental debug-draw teardown used by ``clear_async``; the caller
        # deliberately exits the process immediately after the manifest is
        # durable, so explicit scene teardown is both unnecessary and unsafe.
        pass


async def _start() -> None:
    try:
        await capture(
            int(os.environ.get("CHEESE_CAPTURE_VIEWS", "2")),
            int(os.environ.get("CHEESE_CAPTURE_OFFSET", "0")),
            int(os.environ.get("CHEESE_CAPTURE_LIMIT", "100")),
            int(os.environ.get("CHEESE_CAPTURE_VIEW_START", "0")),
        )
    except Exception as exc:
        print(f"FACTORY_CAPTURE_FATAL {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        os._exit(1)
    os._exit(0)


asyncio.ensure_future(_start())
