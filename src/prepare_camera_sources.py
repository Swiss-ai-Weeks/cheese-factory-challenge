"""Build reviewed RGBA blue-cheese sources for factory-camera capture.

Only named blue-cheese varieties from the normalized train/validation splits
are eligible.  The committed review file is an explicit human QA decision;
generic ``blue`` images are excluded because that source label contains
unrelated blue objects.  Test rows are never read into the candidate set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
DEFAULT_REVIEW = ROOT / "config" / "camera_source_review_v2.csv"
DEFAULT_OUTPUT = PROCESSED / "camera_sources_v2"
MODEL_NAME = "u2net"
MODEL_SHA256 = "8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491"
VARIETY_TOKENS = (
    "blue", "bleu", "stilton", "gorgonzola", "roquefort", "cabrales", "castello",
)
MIN_ACCEPTED = {"train": 19, "val": 4}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def eligible_rows(rows: list[dict]) -> list[dict]:
    eligible = []
    for row in rows:
        if row["source"] != "cheese_images" or row["split"] not in {"train", "val"}:
            continue
        if row["label"] == "blue":
            continue
        extra = json.loads(row["extra"] or "{}")
        text = " ".join((row["label"], extra.get("variety_name", ""))).lower()
        if any(token in text for token in VARIETY_TOKENS):
            eligible.append(row)
    return sorted(eligible, key=lambda row: (row["split"], row["label"], row["uid"]))


def load_review(path: Path, candidate_uids: set[str]) -> dict[str, dict]:
    rows = list(csv.DictReader(path.open()))
    required = {"uid", "decision", "reason"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"review must contain {sorted(required)}")
    review = {}
    for row in rows:
        uid = row["uid"].strip()
        decision = row["decision"].strip()
        if uid in review:
            raise ValueError(f"duplicate review UID: {uid}")
        if uid not in candidate_uids:
            raise ValueError(f"review UID is not an eligible train/val source: {uid}")
        if decision not in {"accept", "reject"}:
            raise ValueError(f"invalid review decision for {uid}: {decision!r}")
        if not row["reason"].strip():
            raise ValueError(f"review reason is empty for {uid}")
        review[uid] = row
    return review


def mask_metrics(alpha: np.ndarray) -> dict:
    from scipy import ndimage

    opaque = alpha >= 64
    labels, component_count = ndimage.label(opaque)
    sizes = np.bincount(labels.ravel())[1:]
    foreground = int(opaque.sum())
    ys, xs = np.where(opaque)
    if foreground:
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        center_offset = float(
            ((xs.mean() / alpha.shape[1] - 0.5) ** 2
             + (ys.mean() / alpha.shape[0] - 0.5) ** 2) ** 0.5
        )
    else:
        bbox = (0, 0, 0, 0)
        center_offset = 1.0
    return {
        "fill_ratio": foreground / alpha.size,
        "largest_component_ratio": (
            float(sizes.max() / foreground) if foreground and sizes.size else 0.0
        ),
        "center_offset": center_offset,
        "component_count": int(component_count),
        "bbox": bbox,
    }


def mask_passes(metrics: dict) -> bool:
    x0, y0, x1, y1 = metrics["bbox"]
    return (
        0.08 <= metrics["fill_ratio"] <= 0.85
        and metrics["largest_component_ratio"] >= 0.72
        and metrics["center_offset"] <= 0.32
        and x1 - x0 >= 64
        and y1 - y0 >= 64
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROCESSED / "manifest.csv")
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--model-cache", type=Path, default=ROOT / "outputs" / "segmentation-models",
    )
    args = parser.parse_args()

    candidates = eligible_rows(list(csv.DictReader(args.manifest.open())))
    by_uid = {row["uid"]: row for row in candidates}
    review = load_review(args.review, set(by_uid))
    accepted = [by_uid[uid] for uid, item in review.items() if item["decision"] == "accept"]
    counts = {split: sum(row["split"] == split for row in accepted) for split in MIN_ACCEPTED}
    for split, minimum in MIN_ACCEPTED.items():
        if counts[split] < minimum:
            raise ValueError(f"accepted {split} sources {counts[split]} < required {minimum}")

    os.environ["U2NET_HOME"] = str(args.model_cache)
    from rembg import new_session, remove

    session = new_session(MODEL_NAME)
    model_path = args.model_cache / "models" / MODEL_NAME / f"{MODEL_NAME}.onnx"
    actual_model_hash = sha256(model_path)
    if actual_model_hash != MODEL_SHA256:
        raise RuntimeError(
            f"segmentation model hash mismatch: {actual_model_hash} != {MODEL_SHA256}"
        )

    cutout_root = args.output / "cutouts"
    cutout_root.mkdir(parents=True, exist_ok=True)
    output_rows = []
    for index, row in enumerate(accepted, 1):
        source_path = PROCESSED / row["path"]
        source = Image.open(source_path).convert("RGB")
        rgba = remove(
            source,
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10,
        )
        metrics = mask_metrics(np.asarray(rgba.getchannel("A")))
        if not mask_passes(metrics):
            raise RuntimeError(f"accepted source {row['uid']} failed mask QA: {metrics}")
        x0, y0, x1, y1 = metrics["bbox"]
        padding = 4
        rgba = rgba.crop((
            max(0, x0 - padding), max(0, y0 - padding),
            min(rgba.width, x1 + padding), min(rgba.height, y1 + padding),
        ))
        destination = cutout_root / f"webblue__{row['uid']}.png"
        rgba.save(destination, "PNG", optimize=True)
        extra = {
            "candidate_label": row["label"],
            "review_reason": review[row["uid"]]["reason"],
            "segmentation_model": MODEL_NAME,
            "segmentation_model_sha256": actual_model_hash,
            "segmentation_tool": "rembg==2.0.84",
            **{key: metrics[key] for key in (
                "fill_ratio", "largest_component_ratio", "center_offset",
                "component_count", "bbox",
            )},
        }
        output_rows.append({
            "uid": f"webblue__{row['uid']}",
            "label": "blue_mould_cheese",
            "fr_label": "blue_mould_cheese",
            "bin": "bin_blue",
            "split": row["split"],
            "path": str(destination.relative_to(PROCESSED)),
            "content_hash": sha256(destination),
            "source_uid": row["uid"],
            "source_path": row["path"],
            "source_hash": row["content_hash"],
            "group": f"webblue__{row['uid']}",
            "extra": json.dumps(extra, sort_keys=True),
        })
        print(f"SEGMENTED {index}/{len(accepted)} uid={row['uid']}", flush=True)

    fields = list(output_rows[0])
    manifest_path = args.output / "manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(output_rows, key=lambda row: (row["split"], row["uid"])))
    print(
        f"CAMERA_SOURCES_DONE candidates={len(candidates)} accepted={len(output_rows)} "
        f"train={counts['train']} val={counts['val']} manifest={manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
