"""Idempotently download the three public cheese-training datasets.

The output directory names intentionally match ``src/normalize.py``. Existing
non-empty datasets are preserved unless ``--force`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def image_count(path: Path) -> int:
    return sum(1 for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def download_hidb(force: bool, workers: int) -> None:
    destination = RAW / "cheese_hidb"
    if image_count(destination) and not force:
        print(f"CHEESE-HIDB already present: {image_count(destination)} images")
        return
    if force:
        shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir(parents=True, exist_ok=True)

    api = "https://api.github.com/repos/andrealoddo/CHEESE-HIDB/git/trees/main?recursive=1"
    with urllib.request.urlopen(api, timeout=60) as response:
        tree = json.load(response)["tree"]
    paths = [entry["path"] for entry in tree if entry["type"] == "blob"
             and Path(entry["path"]).suffix.lower() in IMAGE_SUFFIXES]

    def fetch(relative: str) -> None:
        target = destination / relative
        if target.exists() and target.stat().st_size:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        encoded = urllib.parse.quote(relative)
        url = f"https://raw.githubusercontent.com/andrealoddo/CHEESE-HIDB/main/{encoded}"
        temporary = target.with_suffix(target.suffix + ".part")
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(target)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(fetch, paths))
    print(f"CHEESE-HIDB downloaded: {image_count(destination)} images")


def download_cheese_images(force: bool) -> None:
    destination = RAW / "cheese_images"
    if image_count(destination) and not force:
        print(f"cheese-images already present: {image_count(destination)} images")
        return
    if force:
        shutil.rmtree(destination, ignore_errors=True)
    from huggingface_hub import snapshot_download

    snapshot_download(
        "NoeFlandre/cheese-images",
        repo_type="dataset",
        local_dir=destination,
    )
    print(f"cheese-images downloaded: {image_count(destination)} images")


def download_food_recognition(force: bool) -> None:
    destination = RAW / "food-recognition-2022"
    if image_count(destination) and not force:
        print(f"Food Recognition 2022 already present: {image_count(destination)} images")
        return
    if force:
        shutil.rmtree(destination, ignore_errors=True)
    import dataset_tools as datasets

    RAW.mkdir(parents=True, exist_ok=True)
    datasets.download(dataset="Food Recognition 2022", dst_dir=str(RAW))
    count = image_count(destination)
    if count:
        print(f"Food Recognition 2022 downloaded: {count} images")
        return

    # Dataset Tools currently resolves to a temporarily disabled Dropbox link.
    # Use the public Hugging Face parquet mirror and convert it into the same
    # Supervisely folder structure in ``extract_food_recognition_hf.py``.
    archive = RAW / "food-recognition-2022.tar"
    if archive.exists() and archive.read_bytes()[:15].lower().startswith(b"<!doctype html"):
        archive.unlink()
    mirror = RAW / "food-recognition-2022-hf"
    from huggingface_hub import snapshot_download

    snapshot_download(
        "zhiyingzou0202/food_recognition_2022_processed",
        repo_type="dataset",
        local_dir=mirror,
    )
    parquet = list((mirror / "data").glob("*.parquet"))
    if not parquet:
        raise RuntimeError("Food Recognition mirror downloaded no parquet shards")
    print(f"Food Recognition mirror downloaded: {len(parquet)} parquet shards")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", default=["all"],
                        choices=["all", "hidb", "cheese-images", "food-recognition"])
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    selected = {"hidb", "cheese-images", "food-recognition"} if "all" in args.datasets else set(args.datasets)
    RAW.mkdir(parents=True, exist_ok=True)
    if "hidb" in selected:
        download_hidb(args.force, args.workers)
    if "cheese-images" in selected:
        download_cheese_images(args.force)
    if "food-recognition" in selected:
        download_food_recognition(args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
