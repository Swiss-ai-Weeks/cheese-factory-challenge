"""Ligne de tri de fromages dans Isaac Sim, pilotee par le modele.

Une piece arrive sur le tapis d'amenee, s'arrete sous la camera d'inspection, le
modele dit de quel fromage il s'agit, et le deflecteur de la sortie choisie se
leve pour la pousser sur sa voie. Une voie par classe de sortie : les cinq bacs
plus le rejet.

                        ┌── BIN_HARD      ┌── BIN_SOFT      ┌── BIN_BLUE
    ─── [camera] ───────┴────┬────────────┴────┬────────────┴────┬───────
                             └── BIN_SEMI_HARD └── BIN_FRESH     └── REJECT

Rien n'est scripte a l'avance : la voie prise par chaque piece est celle que
renvoie `sim/sort_server.py`, le modele tournant cote hote (le conteneur Isaac
Sim n'a ni torch ni timm). Le poste d'inspection reprend exactement la geometrie
et l'eclairage du rendu d'entrainement — meme tapis, meme assiette, meme focale,
meme distance — sinon le modele verrait un autre domaine que celui sur lequel il
a appris.

    # hote, dans un autre terminal
    .venv/bin/python sim/sort_server.py

    # conteneur
    docker run --rm --gpus all --network host -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
      -v /home/nvidia/hpe/cheese:/workspace -v /home/nvidia/.cache/ov/hub:/var/cache/hub \
      --entrypoint /isaac-sim/python.sh nvcr.io/nvidia/isaac-sim:6.0.1 \
      /workspace/sim/sorting_line.py --out /workspace/sim/render

Sorties : `frames/f*.png` (la vue d'ensemble), `insp/i*.png` (ce que voit la
camera d'inspection), `timeline.jsonl` (un enregistrement par image) et
`resume.json`. Le montage est fait ensuite par `sim/make_video.py`.
"""

import argparse, io, json, math, random, sys, time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--pieces", default="/workspace/sim/demo_pieces.json")
parser.add_argument("--racine-decoupes", default="/workspace/data/processed")
parser.add_argument("--panneaux", default="/workspace/sim/labels")
parser.add_argument("--out", default="/workspace/sim/render")
parser.add_argument("--serveur", default="http://127.0.0.1:8765/predict")
parser.add_argument("--items", type=int, default=0, help="0 = toutes les pieces du fichier")
parser.add_argument("--fps", type=int, default=24)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--insp-res", type=int, default=768,
                    help="resolution de la camera d'inspection (768 = celle de l'entrainement)")
parser.add_argument("--subframes", type=int, default=6, help="echantillons raytracing par image")
parser.add_argument("--dwell", type=float, default=0.9, help="arret sous la camera, en secondes")
parser.add_argument("--vitesse", type=float, default=0.75, help="m/s sur le tapis principal")
parser.add_argument("--vitesse-voie", type=float, default=0.80, help="m/s sur les voies")
parser.add_argument("--espacement", type=float, default=1.25, help="metres entre deux pieces")
parser.add_argument("--max-frames", type=int, default=4000)
parser.add_argument("--seed", type=int, default=3)
parser.add_argument("--preview", action="store_true",
                    help="rend une vue d'ensemble et une inspection, classe, et sort")
parser.add_argument("--cam-azimut", type=float, default=-90.0)
parser.add_argument("--cam-elevation", type=float, default=42.0)
parser.add_argument("--cam-distance", type=float, default=9.2)
parser.add_argument("--cam-focale", type=float, default=18.0)
args = parser.parse_args()

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "renderer": "RaytracedLighting",
                     "width": args.width, "height": args.height})

import numpy as np
import requests
from PIL import Image

import omni.usd, omni.replicator.core as rep
from pxr import UsdGeom, UsdShade, UsdLux, Gf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from usd_kit import (Tapis, boite, disque, emettre, materiau_texture, materiau_uni,
                     orbite, quad, viser)

# --------------------------------------------------------------------------
# plan de la ligne
# --------------------------------------------------------------------------

X_ENTREE   = -2.60         # debut du tapis principal
X_INSPECT  =  0.00         # poste de camera : c'est l'origine du rendu d'entrainement
X_TRI      =  0.70         # debut de la zone de tri (fin des longerons)
X_FIN      =  4.60         # fin du tapis principal
LARG_LIGNE =  1.10         # largeur du tapis d'entrainement, a ne pas changer
ANGLE_VOIE =  60.0         # deviation d'une voie de sortie, en degres
U_VOIE     =  0.45         # ou commence le tapis de la voie, depuis l'axe principal
LONG_VOIE  =  1.20
LARG_VOIE  =  0.80
LONG_LAME  =  1.30         # lame en biais : il lui faut plus que la largeur du tapis
U_BAC      =  2.05         # distance du centre du bac, depuis l'axe principal
RAYON_ASSIETTE = 0.22
Z_SOL      = -0.95

# Une sortie par classe du modele. Les deflecteurs sont echelonnes le long du
# tapis et alternent de cote : deux voies voisines ne se croisent jamais.
# `reject` recoit tout ce qui n'est pas un fromage rangeable — `not_cheese`,
# `empty`, et les pieces sous le seuil de confiance.
VOIES = [
    ("bin_hard",      1.00,  1.0),
    ("bin_semi_hard", 1.60, -1.0),
    ("bin_soft",      2.20,  1.0),
    ("bin_fresh",     2.80, -1.0),
    ("bin_blue",      3.40,  1.0),
    ("reject",        4.00, -1.0),
]
COULEURS = json.loads((Path(args.panneaux) / "colors.json").read_text())
COULEURS = {k: tuple(c / 255.0 for c in v) for k, v in COULEURS.items()}

rng = random.Random(args.seed)
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")

# --------------------------------------------------------------------------
# decor
# --------------------------------------------------------------------------

mat_sol = materiau_uni(stage, "/World/mat_sol", (0.19, 0.20, 0.23), rugosite=0.9)
boite(stage, "/World/sol", (60.0, 60.0, 0.10), (0, 0, Z_SOL - 0.05), mat_sol)

mat_pied = materiau_uni(stage, "/World/mat_pied", (0.26, 0.27, 0.30), rugosite=0.4, metal=0.6)
mat_assiette = materiau_uni(stage, "/World/mat_assiette", (0.90, 0.90, 0.88), rugosite=0.25)


def pieds(chemin, points, hauteur=0.9):
    """Poteaux verticaux jusqu'au sol : sans eux la ligne flotte."""
    for j, (x, y) in enumerate(points):
        boite(stage, f"{chemin}/pied_{j}", (0.07, 0.07, hauteur),
              (x, y, -0.02 - hauteur / 2), mat_pied)


# -- tapis principal, en deux troncons --------------------------------------
# L'amenee reprend la section exacte du tapis d'entrainement, longerons compris :
# c'est ce que voit la camera d'inspection. La zone de tri, elle, n'a pas de
# longerons — les voies de sortie s'y abouchent, une barriere les couperait.
amenee = Tapis(stage, "/World/amenee", longueur=X_TRI - X_ENTREE, largeur=LARG_LIGNE)
amenee.placer(((X_ENTREE + X_TRI) / 2, 0.0, 0.0))
tri = Tapis(stage, "/World/tri", longueur=X_FIN - X_TRI, largeur=LARG_LIGNE, rails=False)
tri.placer(((X_TRI + X_FIN) / 2, 0.0, 0.0))
pieds("/World/amenee", [(x, y) for x in (X_ENTREE + 0.3, X_INSPECT + 0.5)
                        for y in (0.5, -0.5)])
pieds("/World/tri", [(x, y) for x in (2.4, X_FIN - 0.3) for y in (0.5, -0.5)])

# -- voies de sortie --------------------------------------------------------
voies = {}
for k, (cle, x_station, cote) in enumerate(VOIES):
    angle = cote * ANGLE_VOIE
    a = math.radians(angle)
    direction = (math.cos(a), math.sin(a))
    depart = (x_station, 0.0)

    def point(u, lat=0.0):
        return (depart[0] + direction[0] * u - direction[1] * lat,
                depart[1] + direction[1] * u + direction[0] * lat)

    # la voie s'aboutit au bord du tapis principal, quelques millimetres plus
    # bas pour que la jonction ne clignote pas
    milieu = point(U_VOIE + LONG_VOIE / 2)
    tapis = Tapis(stage, f"/World/voie_{cle}", longueur=LONG_VOIE, largeur=LARG_VOIE,
                  rails=False, couleur_dalle=tuple(0.40 * c for c in COULEURS[cle]),
                  couleur_latte=tuple(0.22 * c for c in COULEURS[cle]))
    tapis.placer((milieu[0], milieu[1], -0.008), rotation_z=angle)
    bout = point(U_VOIE + LONG_VOIE)
    pieds(f"/World/voie_{cle}", [point(U_VOIE + LONG_VOIE - 0.12)])

    # deflecteur : la lame qui pousse la piece hors du tapis principal. Elle
    # monte quand le modele a choisi cette sortie, et redescend ensuite.
    mat_lame = materiau_uni(stage, f"/World/mat_lame_{cle}", COULEURS[cle],
                            rugosite=0.35, metal=0.3)
    lame = boite(stage, f"/World/lame_{cle}", (0.05, LONG_LAME, 0.14), (0, 0, 0), mat_lame)
    op_lame = UsdGeom.Xformable(lame).GetOrderedXformOps()[0]

    # bac de reception, ouvert sur le dessus
    cx, cy = point(U_BAC)
    mat_bac = materiau_uni(stage, f"/World/mat_bac_{cle}", COULEURS[cle], rugosite=0.55)
    base = f"/World/bac_{cle}"
    UsdGeom.Xform.Define(stage, base)
    H_BAC = Z_SOL + 0.04          # du sol au bord superieur
    Z_BAC = (Z_SOL + (-0.06)) / 2
    boite(stage, base + "/fond", (0.86, 0.86, 0.04), (cx, cy, Z_SOL + 0.02), mat_bac)
    for j, (sx, sy) in enumerate(((0.43, 0.0), (-0.43, 0.0), (0.0, 0.43), (0.0, -0.43))):
        taille = (0.04, 0.86, 0.89) if sy == 0 else (0.86, 0.04, 0.89)
        boite(stage, base + f"/mur_{j}", taille, (cx + sx, cy + sy, Z_BAC), mat_bac)

    # panneau : Isaac Sim ne sait pas ecrire, le texte est une texture. Tous
    # font face a la camera d'ensemble, sinon ceux du fond seraient de dos.
    panneau = quad(stage, base + "/panneau")
    mat_p, _ = materiau_texture(stage, base + "/mat_panneau",
                                str(Path(args.panneaux) / f"{cle}.png"))
    UsdShade.MaterialBindingAPI(panneau).Bind(mat_p)
    UsdGeom.Xformable(panneau).AddTransformOp().Set(
        Gf.Matrix4d().SetScale(Gf.Vec3d(0.92, 0.29, 1.0)) *
        Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), 90.0)) *
        Gf.Matrix4d().SetTranslate(Gf.Vec3d(cx, cy - 0.44, 0.24)))

    voies[cle] = {"tapis": tapis, "direction": direction, "angle": angle,
                  "x": x_station, "lame": op_lame, "etat_lame": 0.0,
                  "mat_lame": mat_lame}

# -- poste d'inspection ------------------------------------------------------
# La camera d'inspection est virtuelle ; on pose son boitier juste derriere son
# point de vue, et le bras qui le tient part du cote oppose a la camera video.
CIBLE_INSP = Gf.Vec3d(X_INSPECT, 0.0, 0.02)
AZ_INSP, EL_INSP, D_INSP = 22.0, 62.0, 1.15
oeil_insp = orbite(CIBLE_INSP, AZ_INSP, EL_INSP, D_INSP)
boitier = orbite(CIBLE_INSP, AZ_INSP, EL_INSP, D_INSP + 0.22)

mat_portique = materiau_uni(stage, "/World/mat_portique", (0.42, 0.44, 0.48),
                            rugosite=0.3, metal=0.8)
UsdGeom.Xform.Define(stage, "/World/portique")
boite(stage, "/World/portique/boitier", (0.17, 0.15, 0.15),
      (boitier[0], boitier[1], boitier[2]), mat_portique)
boite(stage, "/World/portique/bras", (0.06, 1.05 - boitier[1], 0.06),
      (boitier[0], (1.05 + boitier[1]) / 2, boitier[2] + 0.10), mat_portique)
boite(stage, "/World/portique/mat", (0.08, 0.08, 1.35),
      (boitier[0], 1.05, 0.66), mat_portique)
# --------------------------------------------------------------------------
# lumieres : celles du rendu d'entrainement pour le poste, plus l'ambiance
# necessaire a la vue d'ensemble
# --------------------------------------------------------------------------

dome = UsdLux.DomeLight.Define(stage, "/World/dome")
dome.CreateIntensityAttr(280.0)

cle_poste = UsdLux.SphereLight.Define(stage, "/World/lumiere_poste")
cle_poste.CreateRadiusAttr(0.12)
cle_poste.CreateIntensityAttr(30000.0)
cle_poste.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.94))
UsdGeom.Xformable(cle_poste).AddTranslateOp().Set(Gf.Vec3d(X_INSPECT - 0.35, -0.30, 1.10))

for j, (lx, ly) in enumerate(((1.9, 2.3), (1.9, -2.3), (4.0, 2.3), (4.0, -2.3),
                              (-2.0, 0.0))):
    halle = UsdLux.SphereLight.Define(stage, f"/World/lumiere_halle_{j}")
    halle.CreateRadiusAttr(0.35)
    halle.CreateIntensityAttr(22000.0)
    UsdGeom.Xformable(halle).AddTranslateOp().Set(Gf.Vec3d(lx, ly, 2.6))

# --------------------------------------------------------------------------
# cameras
# --------------------------------------------------------------------------

# vue d'ensemble : c'est l'image de la video
cam_vue = UsdGeom.Camera.Define(stage, "/World/cam_vue")
cam_vue.CreateFocalLengthAttr(args.cam_focale)
cam_vue.CreateClippingRangeAttr(Gf.Vec2f(0.05, 500.0))
op_cam_vue = UsdGeom.Xformable(cam_vue).AddTransformOp()
CIBLE_VUE = Gf.Vec3d(1.55, 0.05, 0.05)
viser(op_cam_vue, orbite(CIBLE_VUE, args.cam_azimut, args.cam_elevation,
                         args.cam_distance), CIBLE_VUE)

# camera d'inspection : meme cadrage que le rendu d'entrainement (assiette
# centree, vue plongeante, focale 20-32 mm, distance 0.85-1.45 m)
cam_insp = UsdGeom.Camera.Define(stage, "/World/cam_insp")
cam_insp.CreateFocalLengthAttr(26.0)
cam_insp.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
viser(UsdGeom.Xformable(cam_insp).AddTransformOp(), oeil_insp, CIBLE_INSP)

# --------------------------------------------------------------------------
# les pieces qui defilent
# --------------------------------------------------------------------------

infos = json.loads(Path(args.pieces).read_text())
if args.items:
    infos = infos[: args.items]
racine_decoupes = Path(args.racine_decoupes)

UsdGeom.Xform.Define(stage, "/World/pieces")
pieces = []
for i, info in enumerate(infos):
    base = f"/World/pieces/p{i:02d}"
    racine = UsdGeom.Xform.Define(stage, base)
    op = UsdGeom.Xformable(racine).AddTransformOp()

    assiette = disque(stage, base + "/assiette", RAYON_ASSIETTE, 0.30, 0.010, 0.030)
    UsdShade.MaterialBindingAPI(assiette).Bind(mat_assiette)

    morceau = quad(stage, base + "/fromage")
    mat_m, _ = materiau_texture(stage, base + "/mat_fromage",
                                str(racine_decoupes / info["path"]))
    UsdShade.MaterialBindingAPI(morceau).Bind(mat_m)

    # meme regle de taille qu'a l'entrainement : la piece occupe une fraction du
    # diametre utile et sa demi-diagonale reste sous le rayon de l'assiette
    ar = info["width"] / info["height"]
    taille = rng.uniform(0.45, 0.72) * (2 * RAYON_ASSIETTE)
    lx, ly = (taille * ar, taille) if ar >= 1 else (taille, taille / ar)
    marge = 0.92 * RAYON_ASSIETTE
    demi_diag = 0.5 * math.hypot(lx, ly)
    if demi_diag > marge:
        f = marge / demi_diag
        lx, ly = lx * f, ly * f
    UsdGeom.Xformable(morceau).AddTransformOp().Set(
        Gf.Matrix4d().SetScale(Gf.Vec3d(lx, ly, 1.0)) *
        Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), rng.uniform(0, 360))) *
        Gf.Matrix4d().SetTranslate(Gf.Vec3d(rng.uniform(-0.03, 0.03),
                                            rng.uniform(-0.03, 0.03), 0.013)))

    pieces.append({"info": info, "op": op, "vis": UsdGeom.Imageable(racine),
                   "voie": None, "decision": None, "detachee": False,
                   "livree": False, "u0": 0.0, "t_detach": 0.0})
    pieces[-1]["vis"].MakeInvisible()

print(f"{len(pieces)} pieces chargees, {len(VOIES)} voies de sortie", flush=True)

# --------------------------------------------------------------------------
# rendu
# --------------------------------------------------------------------------

rp_vue = rep.create.render_product(str(cam_vue.GetPath()), (args.width, args.height))
annot_vue = rep.AnnotatorRegistry.get_annotator("rgb")
annot_vue.attach([rp_vue])

rp_insp = rep.create.render_product(str(cam_insp.GetPath()), (args.insp_res, args.insp_res))
annot_insp = rep.AnnotatorRegistry.get_annotator("rgb")
annot_insp.attach([rp_insp])

sortie = Path(args.out)
(sortie / "frames").mkdir(parents=True, exist_ok=True)
(sortie / "insp").mkdir(parents=True, exist_ok=True)


def pas_de_rendu():
    """Un pas, avec quelques essais : les premieres frames sortent vides."""
    for _ in range(3):
        rep.orchestrator.step(rt_subframes=args.subframes)
        vue, insp = annot_vue.get_data(), annot_insp.get_data()
        if vue is not None and vue.size and insp is not None and insp.size:
            return np.asarray(vue)[:, :, :3], np.asarray(insp)[:, :, :3]
    return None, None


print("prechauffe du moteur de rendu...", flush=True)
for _ in range(8):
    rep.orchestrator.step(rt_subframes=2)


def classer(image_rgb):
    """Envoie l'image de la camera au modele et renvoie sa decision."""
    tampon = io.BytesIO()
    Image.fromarray(image_rgb).save(tampon, format="PNG")
    t0 = time.perf_counter()
    reponse = requests.post(args.serveur, data=tampon.getvalue(), timeout=60,
                            headers={"Content-Type": "image/png"})
    reponse.raise_for_status()
    decision = reponse.json()
    decision["aller_retour_ms"] = (time.perf_counter() - t0) * 1e3
    return decision


def voie_de(decision) -> str:
    """Traduit la sortie du modele en voie physique."""
    return decision["bin"] if decision["status"] == "ok" else "reject"


def poser(piece, x, y, z, cap=0.0):
    piece["op"].Set(
        Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), cap)) *
        Gf.Matrix4d().SetTranslate(Gf.Vec3d(x, y, z)))


def bouger_lame(voie, hauteur):
    """Monte la lame au-dessus du tapis (1) ou la range dessous (0).

    Elle est parallele a sa voie : une piece qui la longe glisse vers la sortie.
    """
    z = -0.30 + 0.37 * hauteur
    voie["lame"].Set(
        Gf.Matrix4d().SetScale(Gf.Vec3d(0.05, LONG_LAME, 0.14)) *
        Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), voie["angle"] - 90.0)) *
        Gf.Matrix4d().SetTranslate(Gf.Vec3d(voie["x"] - 0.10, 0.0, z)))


for voie in voies.values():
    bouger_lame(voie, 0.0)

# --------------------------------------------------------------------------
# mode apercu : une image de chaque camera, une classification, et on sort
# --------------------------------------------------------------------------

if args.preview:
    poser(pieces[0], X_INSPECT, 0.0, 0.0)
    pieces[0]["vis"].MakeVisible()
    for i, voie in enumerate(voies.values()):
        bouger_lame(voie, 1.0 if i % 2 == 0 else 0.0)
    for i in range(1, len(pieces)):
        if i <= len(VOIES):                       # une piece par voie de sortie
            v = voies[VOIES[i - 1][0]]
            u = U_VOIE + 0.20 + 0.28 * i
            poser(pieces[i], v["x"] + v["direction"][0] * u, v["direction"][1] * u,
                  0.0, cap=v["angle"])
        else:                                     # les suivantes font la queue
            poser(pieces[i], X_INSPECT - (i - len(VOIES)) * args.espacement, 0.0, 0.0)
        pieces[i]["vis"].MakeVisible()
    vue, insp = pas_de_rendu()
    if vue is None:
        print("rendu vide", flush=True); app.close(); raise SystemExit(1)
    Image.fromarray(vue).save(sortie / "apercu_vue.png")
    Image.fromarray(insp).save(sortie / "apercu_insp.png")
    try:
        d = classer(insp)
        print(f"attendu {infos[0]['bin']}/{infos[0]['label']}  ->  "
              f"{d['status']} {d['bin']} {d['cheese_type']} "
              f"({d['bin_confidence']:.3f})", flush=True)
    except Exception as exc:
        print(f"serveur injoignable : {exc}", flush=True)
    print(f"apercu ecrit dans {sortie}", flush=True)
    app.close()
    raise SystemExit(0)

# --------------------------------------------------------------------------
# deroulement de la ligne
# --------------------------------------------------------------------------

DT = 1.0 / args.fps
X_DEPART = -1.45
compteurs = {cle: 0 for cle, _, _ in VOIES}
journal = open(sortie / "timeline.jsonl", "w")

t, avance, i_courant, etat, attente = 0.0, 0.0, 0, "avance", 0.0
nouvelle = False
derniere = None            # derniere decision, affichee par le HUD
insp_courante = None
t0_total = time.perf_counter()
k = 0

while k < args.max_frames:
    if i_courant >= len(pieces) and all(p["livree"] for p in pieces):
        break

    # -- avance du tapis, arret pile sous la camera -------------------------
    if etat == "avance":
        pas = args.vitesse * DT
        if i_courant < len(pieces):
            x = X_DEPART - i_courant * args.espacement + avance
            if x + pas >= X_INSPECT:
                avance += max(0.0, X_INSPECT - x)
                etat, attente, nouvelle = "inspection", args.dwell, True
            else:
                avance += pas
        else:
            avance += pas
    else:
        attente -= DT

    amenee.defiler(avance)
    tri.defiler(avance)
    for cle, _, _ in VOIES:
        voies[cle]["tapis"].defiler(args.vitesse_voie * t)

    # -- position de chaque piece -------------------------------------------
    for i, p in enumerate(pieces):
        if p["livree"]:
            continue
        if not p["detachee"]:
            x = X_DEPART - i * args.espacement + avance
            if p["voie"] is not None and x >= voies[p["voie"]]["x"]:
                p["detachee"] = True
                p["u0"] = x - voies[p["voie"]]["x"]
                p["t_detach"] = t
            else:
                (p["vis"].MakeVisible if x >= X_ENTREE else p["vis"].MakeInvisible)()
                poser(p, x, 0.0, 0.0)
                continue
        voie = voies[p["voie"]]
        u = p["u0"] + args.vitesse_voie * (t - p["t_detach"])
        dx, dy = voie["direction"]
        # apres le bout de la voie, la piece bascule dans le bac
        chute = min(1.0, max(0.0, (u - (U_VOIE + LONG_VOIE)) / (U_BAC - U_VOIE - LONG_VOIE)))
        if chute >= 1.0:
            p["livree"] = True
            p["vis"].MakeInvisible()
            compteurs[p["voie"]] += 1
            continue
        poser(p, voie["x"] + dx * u, dy * u, -0.50 * chute ** 2, cap=voie["angle"])

    # -- deflecteurs : leves tant que leur piece n'a pas quitte le tapis -----
    for cle, _, _ in VOIES:
        voie = voies[cle]
        cible = 1.0 if any((not p["livree"]) and (not p["detachee"]) and p["voie"] == cle
                           for p in pieces) else 0.0
        voie["etat_lame"] += (cible - voie["etat_lame"]) * min(1.0, DT / 0.10)
        bouger_lame(voie, voie["etat_lame"])
        emettre(voie["mat_lame"],
                tuple(0.9 * c * voie["etat_lame"] for c in COULEURS[cle]))

    # -- rendu ---------------------------------------------------------------
    vue, insp = pas_de_rendu()
    if vue is None:
        print(f"  image {k} vide apres 3 essais", flush=True)
        t += DT; k += 1
        continue

    # -- classification, sur la premiere image d'arret -----------------------
    if nouvelle:
        nouvelle = False
        p = pieces[i_courant]
        nom_insp = f"i{i_courant:02d}.png"
        Image.fromarray(insp).save(sortie / "insp" / nom_insp)
        try:
            decision = classer(insp)
        except Exception as exc:
            print(f"  serveur injoignable ({exc}) : piece envoyee au rejet", flush=True)
            decision = {"status": "erreur", "bin": None, "bin_confidence": 0.0,
                        "cheese_type": "?", "type_confidence": 0.0, "topk_types": [],
                        "latency_ms": 0.0, "aller_retour_ms": 0.0}
        attendu = "reject" if p["info"]["bin"] == "not_cheese" else p["info"]["bin"]
        p["voie"] = voie_de(decision)
        p["decision"] = decision
        insp_courante = nom_insp
        derniere = {"piece": i_courant, "uid": p["info"]["uid"],
                    "verite_label": p["info"]["label"], "verite_voie": attendu,
                    "voie": p["voie"], "juste": p["voie"] == attendu,
                    "insp": nom_insp, **decision}
        print(f"[{i_courant + 1}/{len(pieces)}] {p['info']['label']:22s} -> "
              f"{decision['status']:10s} {p['voie']:14s} "
              f"{decision['bin_confidence']:.3f} "
              f"{'OK' if derniere['juste'] else 'RATE (attendu ' + attendu + ')'}",
              flush=True)

    # -- ecriture -------------------------------------------------------------
    Image.fromarray(vue).save(sortie / "frames" / f"f{k:05d}.png")
    journal.write(json.dumps({
        "f": k, "t": round(t, 4), "etat": etat, "piece": i_courant,
        "inspection": etat == "inspection",
        "insp": insp_courante, "derniere": derniere,
        "compteurs": dict(compteurs),
        "traitees": sum(1 for p in pieces if p["decision"]),
        "justes": sum(1 for p in pieces if p["decision"] and p["voie"] ==
                      ("reject" if p["info"]["bin"] == "not_cheese" else p["info"]["bin"])),
    }) + "\n")
    journal.flush()

    if etat == "inspection" and attente <= 0:
        etat, i_courant = "avance", i_courant + 1

    t += DT
    k += 1
    if k % 25 == 0:
        ecoule = time.perf_counter() - t0_total
        print(f"  image {k}  t={t:5.1f}s  {ecoule / k:.2f}s/image", flush=True)

journal.close()

traitees = [p for p in pieces if p["decision"]]
justes = [p for p in traitees
          if p["voie"] == ("reject" if p["info"]["bin"] == "not_cheese" else p["info"]["bin"])]
resume = {
    "images": k, "fps": args.fps, "duree_s": round(k / args.fps, 2),
    "pieces": len(pieces), "traitees": len(traitees), "justes": len(justes),
    "compteurs": compteurs,
    "voies": [cle for cle, _, _ in VOIES],
    "decisions": [{"piece": i, "uid": p["info"]["uid"], "label": p["info"]["label"],
                   "verite_voie": ("reject" if p["info"]["bin"] == "not_cheese"
                                   else p["info"]["bin"]),
                   "voie": p["voie"], "decision": p["decision"]}
                  for i, p in enumerate(pieces)],
}
(sortie / "resume.json").write_text(json.dumps(resume, indent=1))
print(f"\n{k} images ({k / args.fps:.1f}s), {len(justes)}/{len(traitees)} pieces "
      f"rangees dans le bon bac", flush=True)
print(json.dumps(compteurs), flush=True)
app.close()
