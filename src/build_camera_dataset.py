"""Build a balanced, group-safe train/validation factory-camera manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
LABELS = (
    "bin_blue", "bin_fresh", "bin_hard", "bin_semi_hard", "bin_soft", "not_cheese",
)
GROUP_TARGETS = {"train": 26, "val": 5}
CANONICAL_VIEWS = (0, 1, 2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def view_number(uid: str) -> int | None:
    match = re.search(r"__v(\d+)$", uid)
    return int(match.group(1)) if match else None


def resolve_project_path(value: str) -> Path | None:
    if not value:
        return None
    if value.startswith("/workspace/"):
        return ROOT / value.removeprefix("/workspace/")
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def select_balanced(rows: list[dict]) -> list[dict]:
    by_key: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["split"] not in GROUP_TARGETS or row["label"] not in LABELS:
            continue
        by_key[(row["split"], row["label"])][row["group"]].append(row)

    selected = []
    for split, target in GROUP_TARGETS.items():
        for label in LABELS:
            groups = by_key[(split, label)]
            complete = []
            for group, group_rows in groups.items():
                by_view = {view_number(row["uid"]): row for row in group_rows}
                if all(view in by_view for view in CANONICAL_VIEWS):
                    complete.append((group, by_view))
            complete.sort(key=lambda item: item[0])
            if len(complete) < target:
                raise ValueError(
                    f"{split}/{label} has {len(complete)} complete independent groups; need {target}"
                )
            for _, by_view in complete[:target]:
                selected.extend(by_view[view] for view in CANONICAL_VIEWS)
    return selected


def enrich(rows: list[dict], camera_config: Path) -> list[dict]:
    config_hash = sha256(camera_config)
    enriched = []
    for row in rows:
        image_path = PROCESSED / row["path"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        extra = json.loads(row["extra"] or "{}")
        source_hash = extra.get("source_sha256", "")
        source_path = resolve_project_path(row.get("src_path", ""))
        if not source_hash and source_path and source_path.is_file():
            source_hash = sha256(source_path)
        if not source_hash:
            raise ValueError(f"source provenance hash unavailable for {row['uid']}")
        extra.update({
            "camera_config": str(camera_config.relative_to(ROOT)),
            "camera_config_sha256": config_hash,
            "dataset": "factory_camera_balanced_v2",
            "source_sha256": source_hash,
        })
        item = dict(row)
        item["content_hash"] = sha256(image_path)
        item["extra"] = json.dumps(extra, sort_keys=True)
        enriched.append(item)
    return sorted(enriched, key=lambda row: (row["split"], row["label"], row["group"], row["uid"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", type=Path, default=PROCESSED / "manifest_factory_only.csv",
    )
    parser.add_argument(
        "--additional", type=Path,
        default=PROCESSED / "manifest_factory_blue_v2_only.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROCESSED / "manifest_factory_balanced_v2.csv",
    )
    parser.add_argument(
        "--camera-config", type=Path, default=ROOT / "sim" / "factory" / "config.yaml",
    )
    args = parser.parse_args()

    rows = list(csv.DictReader(args.baseline.open())) + list(csv.DictReader(args.additional.open()))
    selected = enrich(select_balanced(rows), args.camera_config)
    fields = list(selected[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    print(
        f"BALANCED_CAMERA_DATASET_DONE rows={len(selected)} "
        f"train_groups={len(LABELS) * GROUP_TARGETS['train']} "
        f"val_groups={len(LABELS) * GROUP_TARGETS['val']} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
