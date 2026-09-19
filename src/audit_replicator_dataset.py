"""Audit Replicator captures, provenance, variation and seeded replay."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(row: dict) -> dict:
    extra = json.loads(row["extra"])
    plan_json = json.dumps(
        extra["replicator_plan"], sort_keys=True, separators=(",", ":"),
    )
    plan_hash = hashlib.sha256(plan_json.encode()).hexdigest()
    if plan_hash != extra["replicator_plan_sha256"]:
        raise ValueError(f"randomization plan hash mismatch: {row['uid']}")
    return extra


def audit_manifest(path: Path, sources: int, views: int) -> tuple[dict, list[dict]]:
    rows = list(csv.DictReader(path.open()))
    if len(rows) != sources * views:
        raise ValueError(f"{path} has {len(rows)} rows; expected {sources * views}")
    if len({row["uid"] for row in rows}) != len(rows):
        raise ValueError(f"duplicate UIDs in {path}")
    if any(row["split"] not in {"train", "val"} for row in rows):
        raise ValueError(f"forbidden split in {path}")

    by_group = defaultdict(list)
    plans = []
    seeds = set()
    for row in rows:
        image = PROCESSED / row["path"]
        if not image.is_file() or sha256(image) != row["content_hash"]:
            raise ValueError(f"crop file/hash mismatch: {row['uid']}")
        extra = _metadata(row)
        full = ROOT / extra["full_frame_path"]
        if not full.is_file() or sha256(full) != extra["full_frame_sha256"]:
            raise ValueError(f"full-frame file/hash mismatch: {row['uid']}")
        if not re.fullmatch(r"[0-9a-f]{64}", extra.get("source_sha256", "")):
            raise ValueError(f"missing source hash: {row['uid']}")
        if not re.fullmatch(r"[0-9a-f]{64}", extra.get("camera_config_sha256", "")):
            raise ValueError(f"missing camera config hash: {row['uid']}")
        by_group[row["group"]].append(extra["view"])
        plans.append(extra["replicator_plan"])
        seeds.add(extra["seed"])
    if len(by_group) != sources:
        raise ValueError(f"{path} has {len(by_group)} groups; expected {sources}")
    expected_views = set(range(views))
    for group, actual in by_group.items():
        if set(actual) != expected_views or len(actual) != views:
            raise ValueError(f"incomplete view set for {group}: {sorted(actual)}")
    if len(seeds) != 1:
        raise ValueError(f"multiple seeds in {path}: {sorted(seeds)}")

    def span(key: str, component: int | None = None) -> float:
        values = [
            float(plan[key] if component is None else plan[key][component])
            for plan in plans
        ]
        return max(values) - min(values)

    variation = {
        "object_rotation_span_deg": span("object_rotation_deg"),
        "object_size_span": span("object_size_fraction"),
        "camera_height_span_m": span("camera_position", 2),
        "focal_length_span_mm": span("focal_length_mm"),
        "key_intensity_span": span("key_intensity"),
        "dome_intensity_span": span("dome_intensity"),
        "floor_red_span": span("floor_color", 0),
        "occluder_enabled_frames": sum(bool(plan["occluder_enabled"]) for plan in plans),
        "occluder_disabled_frames": sum(not bool(plan["occluder_enabled"]) for plan in plans),
    }
    minimums = {
        "object_rotation_span_deg": 120.0,
        "object_size_span": 0.12,
        "camera_height_span_m": 0.06,
        "focal_length_span_mm": 0.15,
        "key_intensity_span": 1800.0,
        "dome_intensity_span": 120.0,
        "floor_red_span": 0.04,
    }
    for key, minimum in minimums.items():
        if variation[key] < minimum:
            raise ValueError(f"insufficient {key}: {variation[key]:.6f} < {minimum}")
    if not variation["occluder_enabled_frames"] or not variation["occluder_disabled_frames"]:
        raise ValueError("smoke set must contain both occluded and unoccluded frames")

    return ({
        "rows": len(rows),
        "groups": len(by_group),
        "views_per_group": views,
        "seed": next(iter(seeds)),
        "split_rows": dict(sorted(Counter(row["split"] for row in rows).items())),
        "label_rows": dict(sorted(Counter(row["label"] for row in rows).items())),
        "variation": variation,
        "test_rows": 0,
        "duplicate_uids": 0,
        "missing_or_mismatched_files": 0,
    }, rows)


def compare_seeded_replay(first: list[dict], second: list[dict]) -> dict:
    def indexed(rows):
        result = {}
        for row in rows:
            extra = _metadata(row)
            key = (row["group"], int(extra["view"]))
            result[key] = (row, extra)
        return result

    left, right = indexed(first), indexed(second)
    if set(left) != set(right):
        raise ValueError("seeded replay uses different source/view keys")
    parameter_matches = 0
    crop_matches = 0
    full_frame_matches = 0
    for key in sorted(left):
        left_row, left_extra = left[key]
        right_row, right_extra = right[key]
        if left_extra["replicator_plan_sha256"] != right_extra["replicator_plan_sha256"]:
            raise ValueError(f"seeded parameter replay mismatch for {key}")
        parameter_matches += 1
        crop_matches += left_row["content_hash"] == right_row["content_hash"]
        full_frame_matches += left_extra["full_frame_sha256"] == right_extra["full_frame_sha256"]
    return {
        "source_view_keys": len(left),
        "parameter_matches": parameter_matches,
        "crop_exact_matches": crop_matches,
        "full_frame_exact_matches": full_frame_matches,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--sources", type=int, default=6)
    parser.add_argument("--views", type=int, default=3)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    audit, rows = audit_manifest(args.manifest, args.sources, args.views)
    report = {"manifest": audit}
    if args.reference:
        reference_audit, reference_rows = audit_manifest(args.reference, args.sources, args.views)
        report["reference_manifest"] = reference_audit
        report["seeded_replay"] = compare_seeded_replay(rows, reference_rows)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
