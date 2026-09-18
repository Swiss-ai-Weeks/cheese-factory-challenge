"""Validated environment filters for factory-domain capture jobs."""

from __future__ import annotations

import os

VALID_SPLITS = {"train", "val", "test"}
VALID_BINS = {
    "bin_blue", "bin_fresh", "bin_hard", "bin_semi_hard", "bin_soft",
    "not_cheese", "empty",
}


def selection(name: str, allowed: set[str]) -> set[str] | None:
    """Parse a comma-separated capture filter and reject silent typos."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    selected = {value.strip() for value in raw.split(",") if value.strip()}
    unknown = selected - allowed
    if unknown:
        raise ValueError(
            f"{name} contains unsupported values: {', '.join(sorted(unknown))}"
        )
    if not selected:
        raise ValueError(f"{name} did not contain any values")
    return selected


def filter_records(
    records: list[dict],
    splits: set[str] | None,
    bins: set[str] | None,
) -> list[dict]:
    """Return only records explicitly eligible for a capture job."""
    return [
        row for row in records
        if (splits is None or row["split"] in splits)
        and (bins is None or row["bin"] in bins)
    ]
