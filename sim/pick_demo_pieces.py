"""Choisit les pieces qui passeront sur le tapis de la demo.

La scene de tri (`sorting_line.py`) doit montrer le modele au travail, pas ses
cas limites : on selectionne donc des decoupes que le modele reconnait deja
correctement sur les rendus d'entrainement (meme domaine, meme assiette). C'est
une curation de demo, pas une mesure — les chiffres du modele sont dans
`runs/sim_type13/results.json`.

    .venv/bin/python sim/pick_demo_pieces.py --per-bin 2 --out sim/demo_pieces.json
"""

from __future__ import annotations

import argparse, csv, json, random, sys
from collections import defaultdict
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "src"))

from PIL import Image                                   # noqa: E402
from predict import CheeseSorter                        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default=str(RACINE / "runs/sim_type13/best.pt"))
ap.add_argument("--manifest", default=str(RACINE / "data/processed/cutouts/manifest.csv"))
ap.add_argument("--renders", default=str(RACINE / "sim/out"))
ap.add_argument("--per-bin", type=int, default=2, help="pieces retenues par bac")
ap.add_argument("--negatives", type=int, default=1, help="objets `not_cheese` a intercaler")
ap.add_argument("--candidates", type=int, default=70, help="pieces testees par bac")
ap.add_argument("--views", type=int, default=3, help="vues evaluees par piece")
ap.add_argument("--min-fill", type=float, default=0.45)
ap.add_argument("--max-ar", type=float, default=2.2)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--out", default=str(RACINE / "sim/demo_pieces.json"))
args = ap.parse_args()

rng = random.Random(args.seed)
rendus = Path(args.renders)

lignes = [r for r in csv.DictReader(open(args.manifest))]
def _ar(r):
    w, h = float(r["width"]), float(r["height"])
    return max(w / h, h / w)
lignes = [r for r in lignes
          if float(r["fill_ratio"]) >= args.min_fill and _ar(r) <= args.max_ar]

par_bac: dict[str, list] = defaultdict(list)
for r in lignes:
    par_bac[r["bin"]].append(r)
for v in par_bac.values():
    rng.shuffle(v)

sorter = CheeseSorter(args.checkpoint, min_confidence=0.55)
sorter.warmup()
print(f"modele charge : {len(sorter.types)} types -> {sorter.bins}", flush=True)


def evalue(r):
    """Note une piece : fraction de vues bien triees, confiance moyenne."""
    vues = sorted(rendus.glob(f"{r['bin']}__{r['uid']}__v*.png"))[: args.views]
    if not vues:
        return None
    attendu = r["bin"] if r["bin"] != "not_cheese" else None
    bons, conf, types = 0, 0.0, []
    for chemin in vues:
        with Image.open(chemin) as img:
            res = sorter.predict(img)
        juste = (res.bin == attendu) if attendu else (res.status == "not_cheese")
        bons += int(juste)
        conf += res.bin_confidence
        types.append(res.cheese_type)
    n = len(vues)
    return {"uid": r["uid"], "label": r["label"], "bin": r["bin"], "path": r["path"],
            "width": int(r["width"]), "height": int(r["height"]),
            "accord": bons / n, "confiance": conf / n, "types_vus": types}


choisies = []
cibles = [b for b in sorted(par_bac) if b != "not_cheese"]
for bac in cibles + (["not_cheese"] if args.negatives else []):
    n_voulu = args.negatives if bac == "not_cheese" else args.per_bin
    notes = []
    for r in par_bac[bac][: args.candidates]:
        note = evalue(r)
        if note:
            notes.append(note)
        if len([x for x in notes if x["accord"] == 1.0]) >= n_voulu * 4:
            break
    notes.sort(key=lambda d: (-d["accord"], -d["confiance"]))
    retenues = notes[:n_voulu]
    choisies += retenues
    for d in retenues:
        print(f"  {bac:14s} {d['uid']}  {d['label']:20s} "
              f"accord={d['accord']:.2f} conf={d['confiance']:.3f}", flush=True)

# On alterne les bacs pour que la video montre les aiguillages partir dans tous
# les sens plutot que trois pieces d'affilee dans le meme bac.
ordre, restant = [], list(choisies)
dernier = None
while restant:
    suivant = next((d for d in restant if d["bin"] != dernier), restant[0])
    restant.remove(suivant)
    ordre.append(suivant)
    dernier = suivant["bin"]

Path(args.out).write_text(json.dumps(ordre, indent=1))
print(f"\n{len(ordre)} pieces ecrites dans {args.out}")
