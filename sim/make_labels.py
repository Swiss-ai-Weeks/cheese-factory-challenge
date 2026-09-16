"""Fabrique les panneaux de sortie de la ligne de tri (textures RGBA).

Isaac Sim ne sait pas dessiner de texte : chaque sortie porte donc un panneau
texture, genere ici avec PIL puis colle sur un quad dans la scene. Les couleurs
sont les memes que celles du HUD de la video.

    .venv/bin/python sim/make_labels.py
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

RACINE = Path(__file__).resolve().parent.parent
SORTIE = RACINE / "sim" / "labels"
POLICE = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Une couleur par destination, partagee par la scene 3D, les panneaux et le HUD.
COULEURS = {
    "bin_hard":      (228, 147,  58),
    "bin_semi_hard": (217, 196,  65),
    "bin_soft":      (127, 176, 105),
    "bin_fresh":     ( 76, 155, 209),
    "bin_blue":      (131, 103, 199),
    "reject":        (193,  85,  78),
}

SOUS_TITRES = {
    "bin_hard": "hard · emmental",
    "bin_semi_hard": "semi-hard · raclette",
    "bin_soft": "soft · goat · processed",
    "bin_fresh": "fresh · cottage · cream",
    "bin_blue": "blue mould",
    "reject": "not cheese · empty · uncertain",
}


def panneau(cle: str, largeur=1024, hauteur=320) -> Image.Image:
    couleur = COULEURS[cle]
    img = Image.new("RGBA", (largeur, hauteur), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([8, 8, largeur - 8, hauteur - 8], radius=36,
                        fill=couleur + (255,), outline=(255, 255, 255, 235), width=7)
    titre = ImageFont.truetype(POLICE, 122)
    petit = ImageFont.truetype(POLICE, 52)
    nom = cle.replace("bin_", "").replace("_", " ").upper() if cle != "reject" else "REJECT"
    d.text((largeur / 2, hauteur / 2 - 34), nom, font=titre, anchor="mm",
           fill=(255, 255, 255, 255))
    d.text((largeur / 2, hauteur / 2 + 82), SOUS_TITRES[cle], font=petit, anchor="mm",
           fill=(255, 255, 255, 225))
    return img


if __name__ == "__main__":
    SORTIE.mkdir(parents=True, exist_ok=True)
    for cle in COULEURS:
        chemin = SORTIE / f"{cle}.png"
        panneau(cle).save(chemin)
        print(f"  {chemin.relative_to(RACINE)}")
    (SORTIE / "colors.json").write_text(json.dumps(COULEURS, indent=1))
    print(f"{len(COULEURS)} panneaux ecrits dans {SORTIE}")
