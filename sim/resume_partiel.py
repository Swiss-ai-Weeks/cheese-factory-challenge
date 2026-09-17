"""Reconstruit `resume.json` a partir de `timeline.jsonl`.

`pick_line.py` n'ecrit son resume qu'a la toute fin. Un rendu arrete en cours
de route — parce qu'on manque de temps, pas parce qu'il a echoue — laisse donc
des images et un journal sans resume, et `make_video.py` ne sait plus quoi en
faire. Ce script relit le journal, ne garde que les images des pieces menees a
leur terme, et fabrique le resume correspondant.
"""
import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--render", required=True)
ap.add_argument("--pilote", default=None)
args = ap.parse_args()

R = Path(args.render)
journal = [json.loads(l) for l in (R / "timeline.jsonl").read_text().splitlines() if l]
if not journal:
    raise SystemExit("journal vide")

# on coupe a la derniere piece terminee : une piece a moitie traitee ferait un
# plan qui s'arrete au milieu d'un geste
dernier = max((i for i, e in enumerate(journal) if e["traitees"] > 0
               and (i + 1 == len(journal) or journal[i + 1]["traitees"] > e["traitees"])),
              default=len(journal) - 1)
fin = journal[dernier]
garde = journal[: dernier + 1]

decisions, vus = [], set()
for e in garde:
    d = e.get("derniere")
    if d and d["piece"] not in vus:
        vus.add(d["piece"])
        decisions.append(d)
decisions = decisions[: fin["traitees"]]

voies = list(fin["compteurs"].keys())
resume = {
    "images": len(garde), "fps": 30, "duree_s": round(len(garde) / 30, 2),
    "pieces": fin["traitees"], "traitees": fin["traitees"],
    "posees": fin["posees"], "justes": fin["justes"],
    "compteurs": fin["compteurs"], "voies": voies,
    "pilote": args.pilote or "automate",
    "resultats": decisions,
}
(R / "resume.json").write_text(json.dumps(resume, indent=1))

# les images au-dela de la coupe ne doivent pas partir au montage
trop = sorted((R / "frames").glob("f*.png"))[len(garde):]
for f in trop:
    f.unlink(missing_ok=True)

print(f"{len(garde)} images gardees, {fin['traitees']} pieces, "
      f"{fin['posees']} posees, {fin['justes']} justes, {len(trop)} images ecartees")
