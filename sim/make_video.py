"""Monte la video de la ligne de tri : rendu Isaac Sim + HUD du modele.

Les images brutes viennent de `sorting_line.py` ; ici on incruste ce que le
modele a decide sur chaque piece — l'image qu'a vue la camera, le type, le bac,
les probabilites et la latence — puis on encode en H.264.

    .venv/bin/python sim/make_video.py --render sim/render --out sim/cheese_sorting.mp4
"""

from __future__ import annotations

import argparse, json, subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

RACINE = Path(__file__).resolve().parent.parent
GRAS = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
NORMAL = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

ap = argparse.ArgumentParser()
ap.add_argument("--render", default=str(RACINE / "sim/render"))
ap.add_argument("--out", default=str(RACINE / "sim/cheese_sorting.mp4"))
ap.add_argument("--fps", type=int, default=24)
ap.add_argument("--titre", type=float, default=2.0, help="secondes de carton d'ouverture")
ap.add_argument("--fin", type=float, default=3.0, help="secondes de carton de fin")
ap.add_argument("--crf", type=int, default=18)
args = ap.parse_args()

RENDU = Path(args.render)
COULEURS = {k: tuple(v) for k, v in
            json.loads((RACINE / "sim/labels/colors.json").read_text()).items()}
NOMS = {"bin_hard": "BIN HARD", "bin_semi_hard": "BIN SEMI-HARD", "bin_soft": "BIN SOFT",
        "bin_fresh": "BIN FRESH", "bin_blue": "BIN BLUE", "reject": "REJECT"}
ORDRE = ["bin_hard", "bin_semi_hard", "bin_soft", "bin_fresh", "bin_blue", "reject"]

F = {n: ImageFont.truetype(GRAS, n) for n in (15, 17, 19, 22, 26, 30, 34, 44, 54)}
FN = {n: ImageFont.truetype(NORMAL, n) for n in (14, 16, 18, 20, 24)}
FM = {n: ImageFont.truetype(MONO, n) for n in (14, 16, 18)}

ENCRE = (238, 240, 244)
ESTOMPE = (150, 156, 166)
FOND = (14, 16, 20)


def carte(d, xy, taille, alpha=205, rayon=14, bord=None):
    """Panneau sombre translucide : le HUD doit rester lisible sur le rendu."""
    x, y = xy; w, h = taille
    d.rounded_rectangle([x, y, x + w, y + h], radius=rayon, fill=FOND + (alpha,),
                        outline=(bord or (70, 76, 88)) + (235,), width=2)


def barre(d, xy, largeur, hauteur, part, couleur):
    x, y = xy
    d.rounded_rectangle([x, y, x + largeur, y + hauteur], radius=hauteur // 2,
                        fill=(46, 50, 58, 230))
    if part > 0.01:
        d.rounded_rectangle([x, y, x + max(hauteur, largeur * part), y + hauteur],
                            radius=hauteur // 2, fill=couleur + (245,))


def puce(d, xy, texte, couleur, police, marge=(14, 7)):
    """Pastille coloree portant le nom d'une destination."""
    x, y = xy
    w = d.textlength(texte, font=police)
    d.rounded_rectangle([x, y, x + w + 2 * marge[0], y + police.size + 2 * marge[1]],
                        radius=9, fill=couleur + (255,))
    d.text((x + marge[0], y + marge[1] - 1), texte, font=police, fill=(20, 20, 22))
    return w + 2 * marge[0]


def hud(fond: Image.Image, enr: dict, insp_cache: dict) -> Image.Image:
    img = fond.convert("RGBA")
    calque = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(calque)
    W, H = img.size
    derniere = enr.get("derniere")
    couleur = COULEURS[derniere["voie"]] if derniere else (96, 104, 118)

    # -- bandeau de titre ---------------------------------------------------
    d.rectangle([0, 0, W, 52], fill=FOND + (205,))
    d.text((20, 13), "CHEESE SORTING LINE", font=F[26], fill=ENCRE)
    d.text((366, 20), "Isaac Sim  ·  ConvNeXt-Base sim_type13  ·  13 classes → 6 sorties",
           font=FN[16], fill=ESTOMPE)
    d.text((W - 20, 17), f"piece {min(enr['piece'] + 1, enr['n_pieces'])}/{enr['n_pieces']}",
           font=F[19], fill=ENCRE, anchor="ra")

    # -- vignette de la camera d'inspection ---------------------------------
    vx, vy, vw = 14, H - 286, 228
    carte(d, (vx, vy), (vw, 270), bord=couleur)
    d.text((vx + 16, vy + 12), "CAMERA D'INSPECTION", font=F[15], fill=ESTOMPE)
    ix, iy, iw = vx + 16, vy + 34, 196
    if enr.get("insp"):
        vignette = insp_cache.get(enr["insp"])
        if vignette is None:
            vignette = Image.open(RENDU / "insp" / enr["insp"]).convert("RGBA").resize(
                (iw, iw), Image.LANCZOS)
            insp_cache[enr["insp"]] = vignette
        calque.paste(vignette, (ix, iy))
        d.rectangle([ix, iy, ix + iw - 1, iy + iw - 1], outline=couleur + (255,), width=3)
    else:
        d.rounded_rectangle([ix, iy, ix + iw, iy + iw], radius=8, fill=(30, 33, 40, 235))
        d.text((ix + iw / 2, iy + iw / 2), "en attente", font=FN[16], fill=ESTOMPE,
               anchor="mm")
    d.text((ix, vy + 240), "768 px  ·  entree modele 384 px", font=FN[14], fill=ESTOMPE)

    # -- decision du modele --------------------------------------------------
    cx, cy, cw, ch = vx + vw + 14, vy, 480, 270
    carte(d, (cx, cy), (cw, ch))
    d.text((cx + 18, cy + 14), "DECISION DU MODELE", font=F[15], fill=ESTOMPE)
    if enr.get("inspection") and derniere and derniere["piece"] == enr["piece"]:
        d.text((cx + cw - 18, cy + 12), "● ANALYSE", font=F[15], fill=(118, 208, 128),
               anchor="ra")

    if derniere:
        dec = derniere
        d.text((cx + 18, cy + 40), dec["cheese_type"].replace("_", " "), font=F[30],
               fill=ENCRE)
        largeur = puce(d, (cx + 18, cy + 86), NOMS[dec["voie"]], couleur, F[22])
        d.text((cx + 32 + largeur, cy + 94), f"{dec['bin_confidence'] * 100:.1f} %",
               font=F[22], fill=ENCRE)
        d.text((cx + 18, cy + 132), "probabilites par type", font=FN[14], fill=ESTOMPE)
        # la verite terrain vient du manifeste de la piece posee sur le tapis :
        # le HUD montre donc aussi les erreurs, quand il y en a. Cette ligne est
        # la seule assez large pour les noms de bac longs.
        d.text((cx + cw - 18, cy + 130),
               "✓ bac attendu" if dec["juste"] else f"✗ attendu : {NOMS[dec['verite_voie']]}",
               font=F[17], fill=(118, 208, 128) if dec["juste"] else (232, 170, 80),
               anchor="ra")
        y = cy + 154
        for nom, prob in dec["topk_types"][:3]:
            d.text((cx + 18, y), nom.replace("_", " ")[:26], font=FM[14], fill=ENCRE)
            barre(d, (cx + 252, y + 3), 140, 11, prob, couleur)
            d.text((cx + cw - 18, y), f"{prob * 100:5.1f}%", font=FM[14], fill=ESTOMPE,
                   anchor="ra")
            y += 24
        etat = {"ok": "status ok — le bras peut ranger",
                "not_cheese": "status not_cheese — ne pas actionner le bras",
                "empty": "status empty — rien a ramasser",
                "uncertain": "status uncertain — laisser passer",
                "erreur": "serveur injoignable"}.get(dec["status"], dec["status"])
        d.text((cx + 18, cy + ch - 32), etat, font=FN[16], fill=ESTOMPE)
        d.text((cx + cw - 18, cy + ch - 32), f"{dec['latency_ms']:.0f} ms", font=FM[16],
               fill=ESTOMPE, anchor="ra")
    else:
        d.text((cx + 18, cy + 112), "premiere piece en approche…", font=FN[20],
               fill=ESTOMPE)

    # -- compteurs par sortie -------------------------------------------------
    bx, by, bw = W - 258, 74, 240
    carte(d, (bx, by), (bw, 44 + 34 * len(ORDRE)))
    d.text((bx + 16, by + 14), "PIECES PAR SORTIE", font=F[15], fill=ESTOMPE)
    y = by + 40
    for cle in ORDRE:
        actif = bool(derniere) and derniere["voie"] == cle
        d.rounded_rectangle([bx + 16, y + 4, bx + 30, y + 18], radius=4,
                            fill=COULEURS[cle] + (255,))
        d.text((bx + 40, y + 2), NOMS[cle], font=F[15] if actif else FN[14],
               fill=ENCRE if actif else ESTOMPE)
        d.text((bx + bw - 16, y), str(enr["compteurs"][cle]), font=F[19],
               fill=ENCRE if actif else ESTOMPE, anchor="ra")
        y += 34

    return Image.alpha_composite(img, calque).convert("RGB")


def carton(taille, titre, lignes, sous=None) -> Image.Image:
    """Carton plein ecran : ouverture et bilan."""
    img = Image.new("RGB", taille, FOND)
    d = ImageDraw.Draw(img)
    W, H = taille
    d.text((W / 2, H / 2 - 150), titre, font=F[54], fill=ENCRE, anchor="mm")
    if sous:
        d.text((W / 2, H / 2 - 96), sous, font=FN[24], fill=ESTOMPE, anchor="mm")
    y = H / 2 - 40
    for texte, couleur in lignes:
        d.text((W / 2, y), texte, font=F[26] if couleur else FN[20],
               fill=couleur or ESTOMPE, anchor="mm")
        y += 42
    return img


# --------------------------------------------------------------------------

journal = [json.loads(l) for l in (RENDU / "timeline.jsonl").read_text().splitlines() if l]
resume = json.loads((RENDU / "resume.json").read_text())
for enr in journal:
    enr["n_pieces"] = resume["pieces"]
print(f"{len(journal)} images, {resume['justes']}/{resume['traitees']} pieces bien triees")

premiere = Image.open(RENDU / "frames" / f"f{journal[0]['f']:05d}.png")
taille = premiere.size

ouverture = carton(taille, "CHEESE SORTING LINE",
                   [("Isaac Sim  ·  ConvNeXt-Base (sim_type13)", None),
                    ("", None),
                    ("le modele lit la camera, l'aiguillage suit", ENCRE)],
                   sous="une voie de sortie par classe")

lignes_fin = [(f"{resume['traitees']} pieces inspectees", ENCRE),
              (f"{resume['justes']}/{resume['traitees']} rangees dans le bac attendu",
               (118, 208, 128) if resume["justes"] == resume["traitees"] else (226, 180, 90)),
              ("", None)]
lignes_fin += [(f"{NOMS[c]} : {resume['compteurs'][c]}", COULEURS[c])
               for c in ORDRE if resume["compteurs"][c]]
final = carton(taille, "BILAN", lignes_fin, sous="decisions prises par le modele, pas par un script")

cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
       "-s", f"{taille[0]}x{taille[1]}", "-r", str(args.fps), "-i", "-",
       "-c:v", "libx264", "-preset", "slow", "-crf", str(args.crf),
       "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)

cache: dict = {}
for _ in range(int(args.titre * args.fps)):
    proc.stdin.write(ouverture.tobytes())
for i, enr in enumerate(journal):
    chemin = RENDU / "frames" / f"f{enr['f']:05d}.png"
    if not chemin.exists():
        continue
    with Image.open(chemin) as fond:
        proc.stdin.write(hud(fond, enr, cache).tobytes())
    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{len(journal)}", flush=True)
for _ in range(int(args.fin * args.fps)):
    proc.stdin.write(final.tobytes())
proc.stdin.close()
proc.wait()

# Le releve de la course part avec la video : `sim/render/` n'est pas versionne,
# et c'est lui que lit la feuille hpe.ipynb pour afficher les decisions.
releve = Path(args.out).with_suffix(".json")
releve.write_text(json.dumps(resume, indent=1))

taille_mo = Path(args.out).stat().st_size / 1e6
duree = args.titre + args.fin + len(journal) / args.fps
print(f"\n{args.out}  {duree:.1f}s  {taille_mo:.1f} Mo")
print(f"{releve}  releve des {resume['traitees']} decisions")
