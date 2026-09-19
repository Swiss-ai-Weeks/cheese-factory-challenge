"""Fail-closed audit for the P1.2 camera sources and balanced dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from build_camera_dataset import CANONICAL_VIEWS, GROUP_TARGETS, LABELS, PROCESSED


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_sources(path: Path) -> dict:
    rows = list(csv.DictReader(path.open()))
    if not rows:
        raise ValueError("camera source manifest is empty")
    if any(row["split"] not in {"train", "val"} for row in rows):
        raise ValueError("camera source manifest contains a forbidden split")
    if len({row["uid"] for row in rows}) != len(rows):
        raise ValueError("camera source manifest contains duplicate UIDs")
    split_counts = Counter(row["split"] for row in rows)
    if split_counts["train"] < 19 or split_counts["val"] < 4:
        raise ValueError(f"insufficient independent sources: {dict(split_counts)}")
    for row in rows:
        image = PROCESSED / row["path"]
        if not image.is_file() or sha256(image) != row["content_hash"]:
            raise ValueError(f"source file/hash mismatch: {row['uid']}")
        if row["label"] != "blue_mould_cheese" or row["bin"] != "bin_blue":
            raise ValueError(f"unexpected source label: {row['uid']}")
        if not re.fullmatch(r"[0-9a-f]{32}", row.get("source_hash", "")):
            raise ValueError(f"missing normalized-source hash: {row['uid']}")
        extra = json.loads(row["extra"])
        if not re.fullmatch(r"[0-9a-f]{64}", extra.get("segmentation_model_sha256", "")):
            raise ValueError(f"missing segmentation-model hash: {row['uid']}")
        if not extra.get("review_reason"):
            raise ValueError(f"missing source-review reason: {row['uid']}")
    return {"rows": len(rows), "split_counts": dict(sorted(split_counts.items()))}


def audit_dataset(path: Path) -> dict:
    rows = list(csv.DictReader(path.open()))
    expected_rows = sum(GROUP_TARGETS.values()) * len(LABELS) * len(CANONICAL_VIEWS)
    if len(rows) != expected_rows:
        raise ValueError(f"dataset has {len(rows)} rows; expected {expected_rows}")
    uids = [row["uid"] for row in rows]
    if len(set(uids)) != len(uids):
        raise ValueError("dataset contains duplicate UIDs")
    if any(row["split"] not in GROUP_TARGETS for row in rows):
        raise ValueError("dataset contains a forbidden split")

    split_of_group = defaultdict(set)
    rows_by_group = defaultdict(list)
    group_counts = Counter()
    class_rows = Counter()
    for row in rows:
        key = (row["split"], row["label"], row["group"])
        split_of_group[row["group"]].add(row["split"])
        rows_by_group[key].append(row)
        class_rows[(row["split"], row["label"])] += 1
    leaked = sorted(group for group, splits in split_of_group.items() if len(splits) != 1)
    if leaked:
        raise ValueError(f"groups cross splits: {leaked[:5]}")

    for (split, label, group), group_rows in rows_by_group.items():
        views = {
            int(re.search(r"__v(\d+)$", row["uid"]).group(1))
            for row in group_rows
            if re.search(r"__v(\d+)$", row["uid"])
        }
        if views != set(CANONICAL_VIEWS) or len(group_rows) != len(CANONICAL_VIEWS):
            raise ValueError(f"non-canonical views for {split}/{label}/{group}: {sorted(views)}")
        group_counts[(split, label)] += 1

    for split, target in GROUP_TARGETS.items():
        for label in LABELS:
            if group_counts[(split, label)] != target:
                raise ValueError(
                    f"group imbalance {split}/{label}: {group_counts[(split, label)]} != {target}"
                )
            if class_rows[(split, label)] != target * len(CANONICAL_VIEWS):
                raise ValueError(f"row imbalance {split}/{label}")

    config_hashes = set()
    for row in rows:
        image = PROCESSED / row["path"]
        if not image.is_file() or sha256(image) != row["content_hash"]:
            raise ValueError(f"capture file/hash mismatch: {row['uid']}")
        extra = json.loads(row["extra"])
        if not re.fullmatch(r"[0-9a-f]{64}", extra.get("source_sha256", "")):
            raise ValueError(f"missing source hash: {row['uid']}")
        if not re.fullmatch(r"[0-9a-f]{64}", extra.get("camera_config_sha256", "")):
            raise ValueError(f"missing camera config hash: {row['uid']}")
        config_hashes.add(extra["camera_config_sha256"])
    if len(config_hashes) != 1:
        raise ValueError(f"multiple camera configurations found: {sorted(config_hashes)}")

    return {
        "rows": len(rows),
        "splits": sorted(GROUP_TARGETS),
        "views_per_group": len(CANONICAL_VIEWS),
        "group_counts": {
            split: {label: group_counts[(split, label)] for label in LABELS}
            for split in GROUP_TARGETS
        },
        "row_counts": {
            split: {label: class_rows[(split, label)] for label in LABELS}
            for split in GROUP_TARGETS
        },
        "camera_config_sha256": next(iter(config_hashes)),
        "test_rows": 0,
        "group_leaks": 0,
        "duplicate_uids": 0,
        "missing_or_mismatched_files": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources", type=Path, default=PROCESSED / "camera_sources_v2" / "manifest.csv",
    )
    parser.add_argument(
        "--dataset", type=Path, default=PROCESSED / "manifest_factory_balanced_v2.csv",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = {
        "source_manifest": audit_sources(args.sources),
        "dataset_manifest": audit_dataset(args.dataset),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
