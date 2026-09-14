"""Normalisation des 3 jeux de donnees fromage vers un format unique.

Sources
-------
cheese_hidb        CHEESE-HIDB (github.com/andrealoddo/CHEESE-HIDB)
                   381 photos 6016x4016 de meules, hierarchie produit/maturite.
cheese_images      NoeFlandre/cheese-images (HuggingFace)
                   ~3200 photos web, 290 varietes de fromage.
food_recognition   Food Recognition 2022 (datasetninja.com/food-recognition)
                   scenes de repas annotees en polygones ; on decoupe les
                   instances dont la classe est un fromage.

Sortie
------
data/processed/images/<source>/<label_slug>/<uid>.jpg   JPEG RGB, grand cote <= 512
data/processed/manifest.csv                             une ligne par image
data/processed/label_maps.json                          label -> index, par tache
data/processed/stats.json                               moyenne/ecart-type par canal (split train)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageFile, ImageOps

ImageFile.LOAD_TRUNCATED_IMAGES = False
Image.MAX_IMAGE_PIXELS = 300_000_000  # les images HIDB font 24 Mpx

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

LONG_SIDE = 512      # grand cote de l'image normalisee
JPEG_QUALITY = 92
MIN_SIDE = 48        # on jette les vignettes/crops trop petits
SEED = 42
SPLIT_RATIO = (0.70, 0.15, 0.15)

# Classes de Food Recognition 2022 qui sont reellement du fromage (et pas un
# plat qui en contient : cheesecake, quiche, sandwich... sont ecartes).
FR_CHEESE_CLASSES = {
    "cheese": "cheese_generic",
    "hard-cheese": "hard_cheese",
    "semi-hard-cheese": "semi_hard_cheese",
    "soft-cheese": "soft_cheese",
    "fresh-cheese": "fresh_cheese",
    "cottage-cheese": "cottage_cheese",
    "cream-cheese": "cream_cheese",
    "goat-cheese-soft": "goat_cheese_soft",
    "blue-mould-cheese": "blue_mould_cheese",
    "emmental-cheese": "emmental_cheese",
    "cheese-for-raclette": "raclette_cheese",
    "processed-cheese": "processed_cheese",
}

# Axe "texture" commun a HIDB et Food Recognition (None = non derivable).
TEXTURE_OF_HIDB = {"semi_hard": "semi_hard", "hard": "hard", "extra_hard": "extra_hard"}
TEXTURE_OF_FR = {
    "hard_cheese": "hard",
    "semi_hard_cheese": "semi_hard",
    "soft_cheese": "soft",
    "fresh_cheese": "fresh",
    "cottage_cheese": "fresh",
    "cream_cheese": "fresh",
    "goat_cheese_soft": "soft",
    "blue_mould_cheese": "soft",
    "emmental_cheese": "hard",
    "raclette_cheese": "semi_hard",
    "processed_cheese": "soft",
}


def slug(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return re.sub(r"_+", "_", text) or "unknown"


# --------------------------------------------------------------------------
# Collecte des enregistrements bruts (aucune image ouverte a ce stade)
# --------------------------------------------------------------------------

def collect_cheese_hidb() -> list[dict]:
    """CHEESE-HIDB, groupe par meule physique.

    L'acquisition est un tour de plateau : chaque meule est photographiee
    exactement 42 fois (DSC_1857..DSC_1898 = une seule meule vue sous 42 angles).
    Decouper ces 42 vues entre train/val/test reviendrait a evaluer le modele sur
    une meule qu'il a deja vue -- d'ou un 100 % de precision sans aucune valeur.
    On reconstitue donc la meule a partir des blocs de numeros consecutifs et on
    l'utilise comme cle de groupe : un split honnete tient des meules entieres
    a l'ecart.
    """
    base = RAW / "cheese_hidb"
    by_class = defaultdict(list)
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        rel = path.relative_to(base)
        if len(rel.parts) != 3:
            continue
        match = re.search(r"(\d+)", path.stem)
        seq = int(match.group(1)) if match else 0
        by_class[(slug(rel.parts[0]), slug(rel.parts[1]))].append((seq, path))

    records = []
    for (product, ripeness), items in sorted(by_class.items()):
        items.sort()
        wheel, previous = 0, None
        for seq, path in items:
            if previous is not None and seq - previous > 1:
                wheel += 1
            previous = seq
            label = f"{product}__{ripeness}"
            records.append({
                "source": "cheese_hidb",
                "task": "hidb_product_ripeness",
                "label": label,
                "src_path": str(path),
                "split": None,
                "group": f"{label}#wheel{wheel}",
                "split_label": product,
                "crop_box": None,
                "extra": {
                    "product": product,
                    "ripeness": ripeness,
                    "texture": TEXTURE_OF_HIDB.get(product),
                    "wheel_id": f"{label}#wheel{wheel}",
                    "sequence": seq,
                },
            })
    return records


def collect_cheese_images() -> list[dict]:
    base = RAW / "cheese_images" / "cheese_dataset"
    records = []
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        rel = path.relative_to(base)
        if len(rel.parts) != 2:
            continue
        records.append({
            "source": "cheese_images",
            "task": "variety",
            "label": slug(rel.parts[0]),
            "src_path": str(path),
            "split": None,
            "group": str(path),
            "split_label": slug(rel.parts[0]),
            "crop_box": None,
            "extra": {"variety_name": rel.parts[0], "texture": None},
        })
    return records


def _parse_fr_annotation(args) -> list[dict]:
    ann_path, img_path, split = args
    try:
        ann = json.loads(Path(ann_path).read_text())
    except (OSError, json.JSONDecodeError):
        return []
    width, height = ann["size"]["width"], ann["size"]["height"]
    out = []
    for idx, obj in enumerate(ann.get("objects", [])):
        label = FR_CHEESE_CLASSES.get(obj.get("classTitle"))
        if label is None:
            continue
        points = obj.get("points", {}).get("exterior") or []
        if len(points) < 3:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        # marge de 8 % pour garder un peu de contexte autour de l'instance
        mx, my = 0.08 * (x1 - x0), 0.08 * (y1 - y0)
        box = (
            max(0, int(x0 - mx)), max(0, int(y0 - my)),
            min(width, int(x1 + mx) + 1), min(height, int(y1 + my) + 1),
        )
        if box[2] - box[0] < MIN_SIDE or box[3] - box[1] < MIN_SIDE:
            continue
        out.append({
            "source": "food_recognition",
            "task": "fr_cheese_type",
            "label": label,
            "src_path": img_path,
            "split": None,
            "group": img_path,
            "split_label": label,
            "crop_box": box,
            "extra": {
                "class_title": obj["classTitle"],
                "object_index": idx,
                "texture": TEXTURE_OF_FR.get(label),
                "official_split": split,
            },
        })
    return out


def collect_food_recognition(workers: int) -> list[dict]:
    base = RAW / "food-recognition-2022"
    split_of = {"training": "train", "validation": "val", "test": "test"}
    jobs = []
    for folder, split in split_of.items():
        ann_dir, img_dir = base / folder / "ann", base / folder / "img"
        if not ann_dir.is_dir():
            continue
        for ann_path in sorted(ann_dir.iterdir()):
            if ann_path.suffix != ".json":
                continue
            img_path = img_dir / ann_path.name[: -len(".json")]
            if img_path.exists():
                jobs.append((str(ann_path), str(img_path), split))

    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for chunk in pool.map(_parse_fr_annotation, jobs, chunksize=256):
            records.extend(chunk)
    return records


# --------------------------------------------------------------------------
# Normalisation image
# --------------------------------------------------------------------------

def _process_one(record: dict) -> dict | None:
    """Ouvre, redresse, recadre, convertit en RGB, redimensionne, enregistre."""
    try:
        with Image.open(record["src_path"]) as img:
            img = ImageOps.exif_transpose(img)
            orig_w, orig_h = img.size
            if record["crop_box"]:
                img = img.crop(tuple(record["crop_box"]))
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGBA")
                flat = Image.new("RGB", img.size, (255, 255, 255))
                flat.paste(img, mask=img.split()[-1])
                img = flat
            elif img.mode != "RGB":
                img = img.convert("RGB")

            w, h = img.size
            if min(w, h) < MIN_SIDE:
                return {"error": "too_small", "src_path": record["src_path"]}
            scale = LONG_SIDE / max(w, h)
            if scale < 1.0:
                img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                                 Image.Resampling.LANCZOS)

            uid = hashlib.sha1(
                f"{record['src_path']}|{record['crop_box']}"
                f"|{record['extra'].get('object_index', '')}".encode()
            ).hexdigest()[:16]
            dest = OUT / "images" / record["source"] / record["label"] / f"{uid}.jpg"
            dest.parent.mkdir(parents=True, exist_ok=True)
            img.save(dest, "JPEG", quality=JPEG_QUALITY, optimize=True,
                     subsampling=0, progressive=True)
            content_hash = hashlib.md5(img.tobytes()).hexdigest()
    except Exception as exc:  # image corrompue, tronquee, format exotique
        return {"error": f"{type(exc).__name__}: {exc}", "src_path": record["src_path"]}

    return {
        "uid": uid,
        "source": record["source"],
        "task": record["task"],
        "label": record["label"],
        "split": record["split"] or "",
        "path": str(dest.relative_to(OUT)),
        "width": img.size[0],
        "height": img.size[1],
        "orig_width": orig_w,
        "orig_height": orig_h,
        "content_hash": content_hash,
        "src_path": str(Path(record["src_path"]).relative_to(ROOT)),
        "crop_box": json.dumps(record["crop_box"]) if record["crop_box"] else "",
        "group": record["group"],
        "split_label": record.get("split_label") or record["label"],
        "extra": json.dumps(record["extra"], ensure_ascii=False),
    }


def normalize(records: list[dict], workers: int) -> tuple[list[dict], list[dict]]:
    rows, errors = [], []
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_process_one, r) for r in records]
        for fut in as_completed(futures):
            res = fut.result()
            done += 1
            if done % 2000 == 0:
                print(f"  {done}/{len(records)} traitees", flush=True)
            if res is None:
                continue
            (errors if "error" in res else rows).append(res)
    return rows, errors


# --------------------------------------------------------------------------
# Deduplication et decoupage train/val/test
# --------------------------------------------------------------------------

def deduplicate(rows: list[dict]) -> tuple[list[dict], int]:
    seen, kept, dropped = {}, [], 0
    for row in sorted(rows, key=lambda r: (r["source"], r["label"], r["uid"])):
        key = (row["source"], row["content_hash"])
        if key in seen:
            dropped += 1
            if row["path"] != seen[key]:
                Path(OUT / row["path"]).unlink(missing_ok=True)
            continue
        seen[key] = row["path"]
        kept.append(row)
    return kept, dropped


def assign_splits(rows: list[dict]) -> None:
    """Split train/val/test stratifie par classe, groupe par image source.

    Deux recadrages issus de la meme photo (Food Recognition) tombent toujours
    dans le meme split : sans cela le modele reverrait en validation un bout
    d'une scene deja vue a l'entrainement.
    Les classes les plus rares sont servies en premier pour qu'elles obtiennent
    leur quota ; en dessous de 3 groupes, la classe part entierement en train.
    """
    by_source = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)

    for source in sorted(by_source):
        rng = random.Random(f"{SEED}:{source}")
        src_rows = by_source[source]
        groups = defaultdict(list)
        for row in src_rows:
            groups[row["group"]].append(row)

        groups_of_label = defaultdict(set)
        for group, items in groups.items():
            for row in items:
                groups_of_label[row.get("split_label") or row["label"]].add(group)

        assigned: dict[str, str] = {}
        # classes rares d'abord : elles ont le moins de marge pour atteindre leur quota
        for label in sorted(groups_of_label, key=lambda l: (len(groups_of_label[l]), l)):
            label_groups = sorted(groups_of_label[label])
            rng.shuffle(label_groups)
            free = [g for g in label_groups if g not in assigned]
            if len(label_groups) < 3:
                for g in free:
                    assigned[g] = "train"
                continue
            counts = Counter(assigned[g] for g in label_groups if g in assigned)
            n = len(label_groups)
            need_val = max(1, round(n * SPLIT_RATIO[1])) - counts["val"]
            need_test = max(1, round(n * SPLIT_RATIO[2])) - counts["test"]
            # on ne vide jamais le train : au moins un groupe lui reste
            budget = max(0, len(free) - 1)
            need_val = max(0, min(need_val, budget))
            need_test = max(0, min(need_test, budget - need_val))
            for i, g in enumerate(free):
                if i < need_val:
                    assigned[g] = "val"
                elif i < need_val + need_test:
                    assigned[g] = "test"
                else:
                    assigned[g] = "train"

        for row in src_rows:
            row["split"] = assigned.get(row["group"], "train")


# --------------------------------------------------------------------------
# Statistiques de normalisation (moyenne / ecart-type sur le split train)
# --------------------------------------------------------------------------

def _channel_moments(path: str) -> tuple:
    import numpy as np
    with Image.open(OUT / path) as img:
        arr = np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0
    arr = arr.reshape(-1, 3)
    return arr.shape[0], arr.sum(0), (arr ** 2).sum(0)


def channel_stats(rows: list[dict], workers: int, sample: int = 6000) -> dict:
    import numpy as np
    train = [r["path"] for r in rows if r["split"] == "train"]
    rng = random.Random(SEED)
    if len(train) > sample:
        train = rng.sample(train, sample)
    total = np.zeros(1)
    s = np.zeros(3)
    sq = np.zeros(3)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for n, si, sqi in pool.map(_channel_moments, train, chunksize=32):
            total += n
            s += si
            sq += sqi
    mean = s / total
    std = np.sqrt(np.maximum(sq / total - mean ** 2, 1e-12))
    return {
        "n_images_sampled": len(train),
        "mean": [round(float(x), 5) for x in mean],
        "std": [round(float(x), 5) for x in std],
    }


# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=min(48, os.cpu_count() or 8))
    parser.add_argument("--sources", nargs="*",
                        default=["cheese_hidb", "cheese_images", "food_recognition"])
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    records = []
    if "cheese_hidb" in args.sources:
        r = collect_cheese_hidb()
        print(f"cheese_hidb       : {len(r)} images")
        records += r
    if "cheese_images" in args.sources:
        r = collect_cheese_images()
        print(f"cheese_images     : {len(r)} images")
        records += r
    if "food_recognition" in args.sources:
        r = collect_food_recognition(args.workers)
        print(f"food_recognition  : {len(r)} instances de fromage a decouper")
        records += r

    if not records:
        print("Aucun enregistrement trouve, telechargement incomplet ?", file=sys.stderr)
        return 1

    print(f"\nNormalisation de {len(records)} images sur {args.workers} processus...")
    rows, errors = normalize(records, args.workers)
    print(f"  {len(rows)} ecrites, {len(errors)} en echec")

    rows, dup = deduplicate(rows)
    print(f"  {dup} doublons supprimes -> {len(rows)} images finales")

    assign_splits(rows)
    print("  splits attribues")

    stats = channel_stats(rows, args.workers)
    print(f"  mean={stats['mean']} std={stats['std']}")

    import csv
    fields = ["uid", "source", "task", "label", "split", "path", "width", "height",
              "orig_width", "orig_height", "content_hash", "src_path", "crop_box",
              "group", "split_label", "extra"]
    rows.sort(key=lambda r: (r["source"], r["label"], r["uid"]))
    with open(OUT / "manifest.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    label_maps = {}
    for task in sorted({r["task"] for r in rows}):
        labels = sorted({r["label"] for r in rows if r["task"] == task})
        label_maps[task] = {label: i for i, label in enumerate(labels)}
    textures = sorted({json.loads(r["extra"]).get("texture") for r in rows} - {None})
    label_maps["texture"] = {t: i for i, t in enumerate(textures)}
    (OUT / "label_maps.json").write_text(json.dumps(label_maps, indent=2))

    summary = {
        "long_side": LONG_SIDE,
        "jpeg_quality": JPEG_QUALITY,
        "min_side": MIN_SIDE,
        "seed": SEED,
        "n_images": len(rows),
        "n_errors": len(errors),
        "n_duplicates_removed": dup,
        "channel_stats": stats,
        "per_source": {
            src: {
                "images": sum(1 for r in rows if r["source"] == src),
                "classes": len({r["label"] for r in rows if r["source"] == src}),
                "splits": dict(Counter(r["split"] for r in rows if r["source"] == src)),
            }
            for src in sorted({r["source"] for r in rows})
        },
        "tasks": {t: len(m) for t, m in label_maps.items()},
    }
    (OUT / "stats.json").write_text(json.dumps(summary, indent=2))
    if errors:
        (OUT / "errors.json").write_text(json.dumps(errors, indent=2))

    print("\n" + json.dumps(summary["per_source"], indent=2))
    print(f"\nEcrit : {OUT/'manifest.csv'}, {OUT/'label_maps.json'}, {OUT/'stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
