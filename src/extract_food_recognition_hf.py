"""Convert the public Food Recognition parquet mirror to Supervisely layout.

The original Dataset Tools Dropbox link is currently disabled. The public
mirror retains each image, ordered object boxes, and the matching semicolon-
separated labels. This converter reconstructs the folder/annotation contract
consumed by ``normalize.py`` and ``cutouts.py``. Bounding-box annotations are
marked so ``cutouts.py`` can refine them with GrabCut instead of treating the
entire rectangle as foreground.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "raw" / "food-recognition-2022-hf" / "data"
DESTINATION = ROOT / "data" / "raw" / "food-recognition-2022"
SPLIT_FOLDER = {"train": "training", "validation": "validation", "test": "test"}


def labels_for_row(value: str) -> list[str]:
    return [label.strip() for label in value.split(";") if label.strip()]


def convert_shard(path: Path, single_object_only: bool) -> tuple[int, int, int]:
    split = path.name.split("-", 1)[0]
    folder = DESTINATION / SPLIT_FOLDER[split]
    image_dir, annotation_dir = folder / "img", folder / "ann"
    image_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    written = mismatched = excluded = 0

    parquet = pq.ParquetFile(path)
    columns = ["image", "width", "height", "objects", "name_label"]
    for batch in parquet.iter_batches(columns=columns, batch_size=128):
        for row in batch.to_pylist():
            labels = labels_for_row(row["name_label"])
            boxes = row["objects"]["bbox"]
            if len(labels) != len(boxes):
                mismatched += 1
                continue
            if single_object_only and len(labels) != 1:
                excluded += 1
                continue
            image_name = Path(row["image"]["path"]).name
            image_path = image_dir / image_name
            annotation_path = annotation_dir / f"{image_name}.json"
            if not image_path.exists():
                image_path.write_bytes(row["image"]["bytes"])

            objects = []
            for label, (x, y, width, height) in zip(labels, boxes):
                x0, y0 = max(0.0, x), max(0.0, y)
                x1 = min(float(row["width"]), x + width)
                y1 = min(float(row["height"]), y + height)
                objects.append({
                    "classTitle": label,
                    "geometryType": "rectangle",
                    "source": "hf_bbox",
                    "points": {
                        "exterior": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                        "interior": [],
                    },
                })
            annotation = {
                "description": "Converted from zhiyingzou0202/food_recognition_2022_processed",
                "size": {"height": row["height"], "width": row["width"]},
                "objects": objects,
            }
            annotation_path.write_text(json.dumps(annotation, separators=(",", ":")))
            written += 1
    return written, mismatched, excluded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--single-object-only", action="store_true",
                        help="retain only rows whose sole label maps unambiguously to one box")
    parser.add_argument("--clean", action="store_true",
                        help="remove the previously converted destination before writing")
    args = parser.parse_args()
    shards = sorted(args.source.glob("*.parquet"))
    if not shards:
        raise SystemExit(f"no parquet shards found in {args.source}")
    if args.clean and DESTINATION.exists():
        expected = (ROOT / "data" / "raw" / "food-recognition-2022").resolve()
        if DESTINATION.resolve() != expected:
            raise SystemExit(f"refusing to clean unexpected path: {DESTINATION.resolve()}")
        shutil.rmtree(DESTINATION)

    total_written = total_mismatched = total_excluded = 0
    for shard in shards:
        written, mismatched, excluded = convert_shard(shard, args.single_object_only)
        total_written += written
        total_mismatched += mismatched
        total_excluded += excluded
        print(f"{shard.name}: {written} images, {mismatched} mismatches, "
              f"{excluded} ambiguous multi-object rows excluded", flush=True)
    print(f"converted {total_written} images; {total_mismatched} mismatches; "
          f"{total_excluded} ambiguous rows excluded")
    return 0 if total_written and not total_mismatched else 1


if __name__ == "__main__":
    raise SystemExit(main())
