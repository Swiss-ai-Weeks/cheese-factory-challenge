"""Build the frozen P1.4 train/validation manifest without consuming test rows."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def build(balanced_path: Path, operational_path: Path, output_path: Path, empty_rows: int) -> dict:
    with balanced_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        balanced = list(reader)
    if not fields or any(row["split"] not in {"train", "val"} for row in balanced):
        raise ValueError("balanced manifest must contain train/val rows only")
    if any(row["label"] == "empty" for row in balanced):
        raise ValueError("balanced manifest unexpectedly already contains empty rows")

    by_group = defaultdict(list)
    with operational_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "train" and row["label"] == "empty":
                by_group[row["group"]].append(row)
    candidates = [rows[0] for _, rows in sorted(by_group.items()) if len(rows) == 1]
    if len(candidates) < empty_rows:
        raise ValueError(f"only {len(candidates)} independent empty rows; need {empty_rows}")
    selected_empty = candidates[:empty_rows]
    combined = balanced + selected_empty
    uids = [row["uid"] for row in combined]
    if len(uids) != len(set(uids)):
        raise ValueError("candidate manifest contains duplicate UIDs")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(combined)
    return {
        "balanced_rows": len(balanced),
        "empty_train_groups": len(selected_empty),
        "empty_train_rows": len(selected_empty),
        "rows": len(combined),
        "test_rows": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--balanced", type=Path,
        default=ROOT / "data" / "processed" / "manifest_factory_balanced_v2.csv",
    )
    parser.add_argument(
        "--operational", type=Path,
        default=ROOT / "data" / "processed" / "manifest_factory_adapt.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data" / "processed" / "manifest_p14_candidate.csv",
    )
    parser.add_argument("--empty-rows", type=int, default=78)
    args = parser.parse_args()
    if args.empty_rows < 1:
        raise ValueError("--empty-rows must be positive")
    print(build(args.balanced, args.operational, args.output, args.empty_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
