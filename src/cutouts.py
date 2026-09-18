"""Decoupes RGBA du fromage, silhouette issue des polygones de Food Recognition.

Les images normalisees sont des rectangles : le fromage plus le pain, l'assiette
ou la main qui l'entourent. Pour poser une piece dans une scene 3D il faut sa
vraie silhouette, sinon on colle un rectangle photographique dans une assiette.
Food Recognition annote chaque instance par un polygone : on s'en sert comme
canal alpha.

Sortie : data/processed/cutouts/<bac>/<uid>.png  (RGBA, grand cote <= 512)
         data/processed/cutouts/manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "food-recognition-2022"
OUT = ROOT / "data" / "processed" / "cutouts"

LONG_SIDE = 512
MIN_SIDE = 64          # une piece plus petite ne donne rien d'exploitable une fois posee
MIN_FILL = 0.12        # polygone couvrant moins de 12 % de sa boite : silhouette douteuse
FEATHER = 1.2          # adoucit le bord, evite l'aspect decoupe aux ciseaux
SEED_NEG = 7           # tirage des negatifs, deterministe


def _one(job: tuple) -> dict | None:
    uid, img_path, ann_path, obj_index, label, bin_label, split = job
    try:
        ann = json.loads(Path(ann_path).read_text())
        obj = ann["objects"][obj_index]
        points = [tuple(p) for p in obj["points"]["exterior"]]
        if len(points) < 3:
            return None

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        if x1 - x0 < MIN_SIDE or y1 - y0 < MIN_SIDE:
            return None

        with Image.open(img_path) as src:
            img = src.convert("RGB")

        # Le jeu original fournit des polygones. Le miroir de secours conserve
        # seulement les boites : GrabCut en affine alors la silhouette au lieu
        # de coller un rectangle photographique dans la scene Isaac.
        masque = Image.new("L", img.size, 0)
        if obj.get("source") == "hf_bbox":
            import cv2
            import numpy as np

            rgb = np.asarray(img)
            gc = np.zeros(rgb.shape[:2], np.uint8)
            ix0, iy0 = max(0, int(x0)), max(0, int(y0))
            ix1 = min(img.width, int(x1 + 0.999))
            iy1 = min(img.height, int(y1 + 0.999))
            largeur, hauteur = ix1 - ix0, iy1 - iy0
            marge_x, marge_y = max(1, int(0.02 * largeur)), max(1, int(0.02 * hauteur))
            rx0, ry0 = max(1, ix0 - marge_x), max(1, iy0 - marge_y)
            rx1 = min(img.width - 1, ix1 + marge_x)
            ry1 = min(img.height - 1, iy1 + marge_y)
            rectangle = (rx0, ry0, max(1, rx1 - rx0), max(1, ry1 - ry0))
            bg_model = np.zeros((1, 65), np.float64)
            fg_model = np.zeros((1, 65), np.float64)
            cv2.grabCut(rgb, gc, rectangle, bg_model, fg_model, 4, cv2.GC_INIT_WITH_RECT)
            foreground = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype("uint8")
            masque = Image.fromarray(foreground, mode="L")
            # Les images difficiles ne doivent pas disparaitre silencieusement.
            crop_mask = foreground[iy0:iy1, ix0:ix1]
            if not crop_mask.size or crop_mask.mean() < 12.0:
                masque = Image.new("L", img.size, 0)
                ImageDraw.Draw(masque).polygon(points, fill=255)
        else:
            ImageDraw.Draw(masque).polygon(points, fill=255)
            # les polygones interieurs (trous) sont retires
            for trou in obj["points"].get("interior", []):
                if len(trou) >= 3:
                    ImageDraw.Draw(masque).polygon([tuple(p) for p in trou], fill=0)

        img = img.crop((x0, y0, x1, y1))
        masque = masque.crop((x0, y0, x1, y1))

        remplissage = sum(masque.getdata()) / (255 * masque.size[0] * masque.size[1])
        if remplissage < MIN_FILL:
            return None

        masque = masque.filter(ImageFilter.GaussianBlur(FEATHER))
        img.putalpha(masque)

        w, h = img.size
        echelle = LONG_SIDE / max(w, h)
        if echelle < 1.0:
            img = img.resize((max(1, round(w * echelle)), max(1, round(h * echelle))),
                             Image.Resampling.LANCZOS)

        dest = OUT / bin_label / f"{uid}.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        img.save(dest, "PNG", optimize=True)
    except Exception:
        return None

    return {
        "uid": uid, "label": label, "bin": bin_label, "split": split,
        "path": str(dest.relative_to(OUT.parent)),
        "width": img.size[0], "height": img.size[1],
        "fill_ratio": round(remplissage, 4),
        "source_image": Path(img_path).name,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=min(48, os.cpu_count() or 8))
    ap.add_argument("--negatives", type=int, default=0,
                    help="extrait aussi N objets qui ne sont PAS du fromage, repartis "
                         "sur les 482 autres classes de Food Recognition. Ils servent "
                         "de classe `not_cheese` : le modele doit savoir dire que ce "
                         "qui passe sur le tapis n'est pas du fromage.")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from dataset import BIN_OF_FR_LABEL

    manifeste = list(csv.DictReader(open(ROOT / "data/processed/manifest.csv")))
    dossier = {"train": "training", "val": "validation", "test": "test"}

    jobs = []
    for r in manifeste:
        if r["source"] != "food_recognition":
            continue
        bac = BIN_OF_FR_LABEL.get(r["label"], "")
        if not bac:                       # cheese_generic : pas de bac attribue
            continue
        extra = json.loads(r["extra"])
        officiel = dossier[{"train": "train", "val": "val", "test": "test"}[
            extra.get("official_split", "train")]]
        nom = Path(r["src_path"]).name
        ann = RAW / officiel / "ann" / f"{nom}.json"
        img = RAW / officiel / "img" / nom
        if not ann.exists() or not img.exists():
            continue
        jobs.append((r["uid"], str(img), str(ann), extra["object_index"],
                     r["label"], bac, r["split"]))

    if args.negatives:
        # On pioche dans les autres classes en tournant sur la liste triee par
        # rarete croissante : on veut de la diversite, pas 300 tranches de pain.
        import random as _random
        rng = _random.Random(SEED_NEG)
        par_classe = defaultdict(list)
        for folder in ("training", "validation"):
            ann_dir = RAW / folder / "ann"
            if not ann_dir.is_dir():
                continue
            for ann_path in sorted(ann_dir.iterdir()):
                if ann_path.suffix != ".json":
                    continue
                img = RAW / folder / "img" / ann_path.name[: -len(".json")]
                if not img.exists():
                    continue
                try:
                    ann = json.loads(ann_path.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                for idx, obj in enumerate(ann.get("objects", [])):
                    titre = obj.get("classTitle", "")
                    if "cheese" in titre.lower():
                        continue
                    pts = obj.get("points", {}).get("exterior") or []
                    if len(pts) < 3:
                        continue
                    xs = [q[0] for q in pts]; ys = [q[1] for q in pts]
                    if max(xs) - min(xs) < 96 or max(ys) - min(ys) < 96:
                        continue
                    par_classe[titre].append((str(img), str(ann_path), idx, titre))

        classes = sorted(par_classe, key=lambda c: (-len(par_classe[c]), c))
        for c in classes:
            rng.shuffle(par_classe[c])
        pioche, i = [], 0
        while len(pioche) < args.negatives and any(par_classe[c] for c in classes):
            c = classes[i % len(classes)]
            if par_classe[c]:
                pioche.append(par_classe[c].pop())
            i += 1
        print(f"{len(pioche)} negatifs pioches sur {len({p[3] for p in pioche})} classes distinctes")
        for img, ann, idx, titre in pioche:
            uid = hashlib.sha1(f"neg|{img}|{idx}".encode()).hexdigest()[:16]
            jobs.append((uid, img, ann, idx, titre, "not_cheese", ""))

    print(f"{len(jobs)} instances a decouper sur {args.workers} processus...")
    OUT.mkdir(parents=True, exist_ok=True)
    lignes = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for res in pool.map(_one, jobs, chunksize=32):
            if res:
                lignes.append(res)

    champs = ["uid", "label", "bin", "split", "path", "width", "height",
              "fill_ratio", "source_image"]
    with open(OUT / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=champs)
        w.writeheader()
        w.writerows(sorted(lignes, key=lambda r: (r["bin"], r["uid"])))

    import collections
    print(f"{len(lignes)} decoupes ecrites ({len(jobs) - len(lignes)} rejetees)")
    for bac, n in sorted(collections.Counter(r["bin"] for r in lignes).items()):
        print(f"  {bac:16s} {n:4d}")
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
