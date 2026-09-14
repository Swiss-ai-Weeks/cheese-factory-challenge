"""Construit le manifeste des rendus Isaac Sim, avec des splits sans fuite.

Les N vues d'une meme piece ne sont pas N echantillons : c'est le meme objet vu
sous N angles. Si elles se repartissent entre train et test, on remesure la fuite
corrigee sur les meules CHEESE-HIDB. Le decoupage se fait donc par piece source
(`group` = uid de la decoupe), jamais par fichier.

Les images sont rangees dans la meme arborescence que le reste
(data/processed/images/sim_belt/<bac>/) par lien dur, donc sans cout disque.

    python src/render_manifest.py
    -> data/processed/manifest_sim.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RENDUS = ROOT / "sim" / "out"
PROCESSED = ROOT / "data" / "processed"
IMAGES = PROCESSED / "images" / "sim_belt"

SEED = 42
SPLIT_RATIO = (0.70, 0.15, 0.15)

# Les rendus sortent en PNG 768 px. On les ramene au format du reste du pipeline
# (JPEG 512 px, qualite 92) pour deux raisons : le decodage PNG est le goulot du
# dataloader, et surtout comparer `bin` a `sim_bin` n'a de sens que si les deux
# modeles voient des images produites de la meme facon. Sinon l'ecart melange le
# changement de domaine avec un changement de resolution et de compression.
LONG_SIDE = 512
JPEG_QUALITY = 92


def _convertir(job: tuple) -> str | None:
    source, dest = job
    try:
        from PIL import Image
        with Image.open(source) as im:
            im = im.convert("RGB")
            w, h = im.size
            k = LONG_SIDE / max(w, h)
            if k < 1.0:
                im = im.resize((max(1, round(w * k)), max(1, round(h * k))),
                               Image.Resampling.LANCZOS)
            im.save(dest, "JPEG", quality=JPEG_QUALITY, optimize=True, subsampling=0)
    except Exception as exc:
        return f"{source}: {exc}"
    return None


def decoupe_par_groupe(lignes: list[dict]) -> None:
    """Split stratifie par bac, mais assigne par piece source entiere."""
    rng = random.Random(f"{SEED}:sim_belt")
    groupes = defaultdict(list)
    for r in lignes:
        groupes[r["group"]].append(r)

    groupes_du_bac = defaultdict(set)
    for g, items in groupes.items():
        groupes_du_bac[items[0]["label"]].add(g)

    assigne: dict[str, str] = {}
    for bac in sorted(groupes_du_bac, key=lambda b: (len(groupes_du_bac[b]), b)):
        gs = sorted(groupes_du_bac[bac])
        rng.shuffle(gs)
        libres = [g for g in gs if g not in assigne]
        if len(gs) < 3:
            for g in libres:
                assigne[g] = "train"
            continue
        deja = Counter(assigne[g] for g in gs if g in assigne)
        n = len(gs)
        n_val = max(1, round(n * SPLIT_RATIO[1])) - deja["val"]
        n_test = max(1, round(n * SPLIT_RATIO[2])) - deja["test"]
        budget = max(0, len(libres) - 1)
        n_val = max(0, min(n_val, budget))
        n_test = max(0, min(n_test, budget - n_val))
        for i, g in enumerate(libres):
            assigne[g] = "val" if i < n_val else "test" if i < n_val + n_test else "train"

    for r in lignes:
        r["split"] = assigne.get(r["group"], "train")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--renders", default=str(RENDUS))
    ap.add_argument("--out", default=str(PROCESSED / "manifest_sim.csv"))
    args = ap.parse_args()

    src = Path(args.renders)

    # Les JSON des shards peuvent annoncer des rendus qu'un autre shard a ecrases.
    # La verite, c'est ce qui est sur le disque : on enumere les fichiers et on ne
    # prend dans les JSON que les metadonnees d'angle, quand elles existent.
    infos = {}
    for m in sorted(src.glob("manifest_*.json")):
        for e in json.loads(m.read_text()):
            infos[e["file"]] = e

    # Les JSON de shard se sont deja ecrases entre eux : une passe de rendu
    # ulterieure reecrit `manifest_<i>.json` et efface les metadonnees de la
    # precedente. Le manifeste des decoupes, lui, est autoritaire -- `group` est
    # l'uid de la decoupe -- donc on en tire le type fin, et on ne garde les JSON
    # que pour l'angle de prise de vue.
    types_par_uid = {}
    chemin_cut = PROCESSED / "cutouts" / "manifest.csv"
    if chemin_cut.exists():
        for c in csv.DictReader(open(chemin_cut)):
            types_par_uid[c["uid"]] = c["label"] if c["bin"] != "not_cheese" else "not_cheese"

    meta = []
    for f in sorted(src.glob("*.png")):
        bac, uid, vue = f.stem.split("__")
        e = infos.get(f.name, {})
        meta.append({"file": f.name, "bin": bac, "group": uid,
                     "view": int(vue.lstrip("v")),
                     "label": types_par_uid.get(uid, "empty" if bac == "empty"
                                                else e.get("label", "")),
                     "azimut": e.get("azimut", 0.0),
                     "elevation": e.get("elevation", 0.0)})
    if not meta:
        print("aucun rendu trouve sur le disque")
        return 1
    sans_type = sum(1 for m in meta if not m["label"])
    print(f"{len(meta)} fichiers sur disque, "
          f"{sum(1 for m in meta if m['file'] not in infos)} sans angle de vue, "
          f"{sans_type} sans type")
    if sans_type:
        print(f"  ATTENTION : {sans_type} rendus sans type seraient exclus de "
              f"l'entrainement")

    lignes, conversions = [], []
    for r in meta:
        fichier = src / r["file"]
        if not fichier.exists():
            continue
        dest = IMAGES / r["bin"] / (Path(r["file"]).stem + ".jpg")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            conversions.append((fichier, dest))
        lignes.append({
            "uid": Path(r["file"]).stem,
            "source": "sim_belt",
            "task": "sim_bin",
            "label": r["bin"],
            "split": "",
            "path": str(dest.relative_to(PROCESSED)),
            "width": 0, "height": 0, "orig_width": 0, "orig_height": 0,
            "content_hash": "", "src_path": "", "crop_box": "",
            "group": r["group"],
            "split_label": r["bin"],
            "extra": json.dumps({
                "fr_label": r["label"], "view": r["view"],
                "azimut": r["azimut"], "elevation": r["elevation"],
                "texture": None,
            }),
        })

    if conversions:
        from concurrent.futures import ProcessPoolExecutor
        n = min(48, os.cpu_count() or 8)
        print(f"conversion de {len(conversions)} rendus en JPEG {LONG_SIDE} px "
              f"sur {n} processus...")
        echecs = []
        with ProcessPoolExecutor(max_workers=n) as pool:
            for e in pool.map(_convertir, conversions, chunksize=32):
                if e:
                    echecs.append(e)
        if echecs:
            print(f"  {len(echecs)} echecs, ex. {echecs[0]}")
        lignes = [r for r in lignes if (PROCESSED / r["path"]).exists()]

    decoupe_par_groupe(lignes)

    champs = ["uid", "source", "task", "label", "split", "path", "width", "height",
              "orig_width", "orig_height", "content_hash", "src_path", "crop_box",
              "group", "split_label", "extra"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=champs)
        w.writeheader()
        w.writerows(sorted(lignes, key=lambda r: (r["label"], r["uid"])))

    sp = Counter(r["split"] for r in lignes)
    print(f"{len(lignes)} rendus, {len({r['group'] for r in lignes})} pieces distinctes")
    print(f"splits : train={sp['train']} val={sp['val']} test={sp['test']}")
    pieces = defaultdict(set)
    for r in lignes:
        pieces[r["split"]].add(r["group"])
    print(f"pieces : train={len(pieces['train'])} val={len(pieces['val'])} test={len(pieces['test'])}")
    chevauchement = sum(1 for g, s in
                        ((g, {r['split'] for r in lignes if r['group'] == g}) for g in list(pieces['val'])[:50])
                        if len(s) > 1)
    print(f"pieces a cheval (echantillon de 50) : {chevauchement}")
    for bac, n in sorted(Counter(r["label"] for r in lignes).items()):
        print(f"  {bac:16s} {n:5d}")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
