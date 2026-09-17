"""La ligne complete : la camera decide, le bras execute.

Deux modeles, deux metiers, aucun ne connait le travail de l'autre.

    camera ──▶ sim_type13 (hote, HTTP) ──▶ numero de voie ──▶ politique FR3 ──▶ geste

Le classifieur de fromages ne sait pas qu'un bras existe ; il rend un bac. La
politique du bras n'a jamais vu de fromage ; elle recoit un entier et sait
saisir une assiette qui defile, la porter et la poser a plat sur la voie
demandee sans la renverser. Ce fichier est le point de rencontre.

    # hote
    .venv/bin/python sim/sort_server.py

    # conteneur
    docker run --rm --gpus '"device=0"' --network host -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \\
      -v /home/nvidia/hpe/cheese:/workspace -v ~/.cache/ov/hub:/var/cache/hub \\
      -e PYTHONPATH=/isaac-sim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle \\
      --entrypoint /isaac-sim/python.sh nvcr.io/nvidia/isaac-sim:6.0.1 \\
      /workspace/sim/pick_line.py --out /workspace/sim/rendu_bras

`--scripte` remplace la politique par l'automate de reference : utile pour
mettre la scene au point sans attendre la fin de l'entrainement.

Sorties : `frames/f*.png`, `insp/i*.png`, `timeline.jsonl`, `resume.json` —
le meme format que `sorting_line.py`, pour le meme montage video.
"""

import argparse, io, json, math, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--politique", default="/workspace/runs/pick_fr3/best.pt")
ap.add_argument("--scripte", action="store_true", help="automate au lieu de la politique")
ap.add_argument("--pieces", default="/workspace/sim/demo_pieces.json")
ap.add_argument("--racine-decoupes", default="/workspace/data/processed")
ap.add_argument("--panneaux", default="/workspace/sim/labels")
ap.add_argument("--out", default="/workspace/sim/rendu_bras")
ap.add_argument("--serveur", default="http://127.0.0.1:8765/predict")
ap.add_argument("--items", type=int, default=0, help="0 = toutes les pieces du fichier")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--height", type=int, default=720)
ap.add_argument("--insp-res", type=int, default=768)
ap.add_argument("--subframes", type=int, default=4)
ap.add_argument("--dwell", type=float, default=0.8, help="arret sous la camera, en secondes")
ap.add_argument("--vitesse", type=float, default=0.35, help="m/s du tapis d'amenee")
# L'assiette naît hors cadre et entre dans l'image en defilant : on ne la voit
# jamais apparaitre. Le point de depart est en amont du bord gauche du champ.
ap.add_argument("--x-depart", type=float, default=-1.85,
                help="ou l'assiette est deposee sur le tapis, hors champ")
ap.add_argument("--max-frames", type=int, default=4000)
ap.add_argument("--delai-cycle", type=int, default=750, help="pas max pour traiter une piece")
ap.add_argument("--preview", action="store_true", help="une image de chaque camera, puis sortie")
ap.add_argument("--cam-azimut", type=float, default=-124.0)
ap.add_argument("--cam-elevation", type=float, default=33.0)
ap.add_argument("--cam-distance", type=float, default=3.45)
ap.add_argument("--cam-focale", type=float, default=20.0)
args = ap.parse_args()

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "renderer": "RaytracedLighting",
                     "width": args.width, "height": args.height})

import numpy as np
import requests
import torch
from PIL import Image

import omni.replicator.core as rep
from pxr import Gf, UsdGeom, UsdLux, UsdShade

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pick_cell import (Automate, Cellule, HAUT_ASSIETTE, LARG_TAPIS, LARG_VOIE,
                       LONG_VOIE, N_VOIES, RAYON_ASSIETTE, X_PRISE, Y_TAPIS,
                       Z_BASE, Z_MORCEAU, FOND_ASSIETTE, LONG_TAPIS,
                       X_ENTREE, ecart_lacet, lacet_de, voies_xy)
from usd_kit import (Tapis, boite, materiau_texture, materiau_uni, orbite, quad,
                     viser)

X_INSPECT = -0.62          # poste de camera, en amont de la fenetre de prise
VOIES = ["bin_hard", "bin_semi_hard", "bin_soft", "bin_fresh", "bin_blue", "reject"]

# --------------------------------------------------------------------------
# la cellule, puis les ajouts qui ne servent qu'a l'image
# --------------------------------------------------------------------------

cellule = Cellule(n_env=1, dt_physique=1 / 60, sous_pas=2, espacement=6.0,
                  auto_collisions=False, graine=5)
cellule.duree_max = 10 ** 6            # c'est la ligne qui rythme, pas l'episode
stage = cellule.stage
base = "/World/env_0"
O = cellule.origines[0].tolist()       # origine monde de la cellule

couleurs = json.loads((Path(args.panneaux) / "colors.json").read_text())
couleurs = {k: tuple(c / 255.0 for c in v) for k, v in couleurs.items()}

# le morceau garde sa physique mais disparait de l'image : ce que la camera doit
# voir, c'est la decoupe posee au fond de l'assiette, comme a l'entrainement
UsdGeom.Imageable(stage.GetPrimAtPath(base + "/morceau/col")).MakeInvisible()

fromage = quad(stage, base + "/assiette/fromage")
mat_fromage, tex_fromage = materiau_texture(stage, base + "/assiette/mat_fromage")
UsdShade.MaterialBindingAPI(fromage).Bind(mat_fromage)
op_fromage = UsdGeom.Xformable(fromage).AddTransformOp()

# Tapis visuels. La cellule d'entrainement se contente de dalles statiques :
# pour l'image il faut les vrais tapis d'`usd_kit` — dalle, longerons et lattes
# qui defilent — et c'est aussi ce que le classifieur a vu a l'entrainement.
# les dalles de la cellule gardent leurs colliders mais sortent de l'image :
# un collider invisible collisionne toujours
for chemin in [base + "/amenee"] + [base + f"/voie_{k}" for k in range(N_VOIES)]:
    UsdGeom.Imageable(stage.GetPrimAtPath(chemin)).MakeInvisible()

amenee = Tapis(stage, base + "/tapis_amenee", longueur=LONG_TAPIS,
               largeur=LARG_TAPIS)
amenee.placer((0.0, Y_TAPIS, -0.001))

tapis_voies = []
for k, (x, y) in enumerate(voies_xy()):
    cle = VOIES[k]
    # plus etroit que la voie physique : a 62 cm du bras, six voies de 32 cm se
    # recouvrent et l'ensemble se lit comme une seule nappe
    t = Tapis(stage, base + f"/tapis_voie_{k}", longueur=0.72, largeur=0.22,
              rails=False,
              couleur_dalle=tuple(0.22 * c for c in couleurs[cle]),
              couleur_latte=tuple(0.11 * c for c in couleurs[cle]))
    t.placer((x * 1.06, y * 1.06, -0.001), rotation_z=math.degrees(math.atan2(y, x)))
    tapis_voies.append(t)

    # panneau : Isaac Sim ne sait pas ecrire, le texte est une texture. Tous
    # font face a la camera d'ensemble, sinon ceux du fond seraient de dos.
    panneau = quad(stage, base + f"/panneau_{k}")
    mat_p, _ = materiau_texture(stage, base + f"/mat_pan_{k}",
                                str(Path(args.panneaux) / f"{cle}.png"))
    UsdShade.MaterialBindingAPI(panneau).Bind(mat_p)
    r = 0.95
    UsdGeom.Xformable(panneau).AddTransformOp().Set(
        Gf.Matrix4d().SetScale(Gf.Vec3d(0.34, 0.107, 1.0)) *
        Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), 90.0)) *
        Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), args.cam_azimut + 90.0)) *
        Gf.Matrix4d().SetTranslate(Gf.Vec3d(r * math.cos(math.radians(math.degrees(
            math.atan2(y, x)))), r * math.sin(math.radians(math.degrees(
                math.atan2(y, x)))), 0.13)))

# poste d'inspection : le boitier et son portique, hors du champ de la camera
CIBLE_INSP = Gf.Vec3d(O[0] + X_INSPECT, O[1] + Y_TAPIS, O[2] + FOND_ASSIETTE + 0.003)
# La distance suit la taille de l'assiette : ce qui doit rester constant,
# c'est la taille APPARENTE de la decoupe, c'est-a-dire le cadrage sur lequel
# le classifieur a ete entraine.
AZ_INSP, EL_INSP, D_INSP = 150.0, 62.0, 0.43 * (2 * RAYON_ASSIETTE) / 0.180
oeil_insp = orbite(CIBLE_INSP, AZ_INSP, EL_INSP, D_INSP)
boitier = orbite(CIBLE_INSP, AZ_INSP, EL_INSP, D_INSP + 0.16)
mat_portique = materiau_uni(stage, "/World/mat_portique", (0.42, 0.44, 0.48), 0.3, 0.8)
boite(stage, "/World/portique/boitier", (0.10, 0.09, 0.09),
      (boitier[0], boitier[1], boitier[2]), mat_portique)
boite(stage, "/World/portique/bras", (0.05, 0.75, 0.05),
      (boitier[0], boitier[1] + 0.37, boitier[2] + 0.06), mat_portique)
boite(stage, "/World/portique/mat", (0.07, 0.07, 1.30),
      (boitier[0], boitier[1] + 0.74, O[2] + 0.35), mat_portique)

cle_poste = UsdLux.SphereLight.Define(stage, "/World/lumiere_poste")
cle_poste.CreateRadiusAttr(0.05)
cle_poste.CreateIntensityAttr(4200.0)
cle_poste.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.94))
UsdGeom.Xformable(cle_poste).AddTranslateOp().Set(
    Gf.Vec3d(CIBLE_INSP[0] - 0.22, CIBLE_INSP[1] - 0.18, CIBLE_INSP[2] + 0.45))

for j, (lx, ly) in enumerate(((1.6, 1.6), (1.6, -1.6), (-1.6, 1.6), (-1.6, -1.6))):
    halle = UsdLux.SphereLight.Define(stage, f"/World/lumiere_halle_{j}")
    halle.CreateRadiusAttr(0.35)
    halle.CreateIntensityAttr(9000.0)
    UsdGeom.Xformable(halle).AddTranslateOp().Set(Gf.Vec3d(O[0] + lx, O[1] + ly, O[2] + 2.4))

# -- cameras ---------------------------------------------------------------
cam_vue = UsdGeom.Camera.Define(stage, "/World/cam_vue")
cam_vue.CreateFocalLengthAttr(args.cam_focale)
cam_vue.CreateClippingRangeAttr(Gf.Vec2f(0.05, 500.0))
CIBLE_VUE = Gf.Vec3d(O[0], O[1] + 0.05, O[2] + 0.16)
viser(UsdGeom.Xformable(cam_vue).AddTransformOp(),
      orbite(CIBLE_VUE, args.cam_azimut, args.cam_elevation, args.cam_distance), CIBLE_VUE)

# camera d'inspection : le cadrage exact des rendus d'entrainement
cam_insp = UsdGeom.Camera.Define(stage, "/World/cam_insp")
cam_insp.CreateFocalLengthAttr(26.0)
cam_insp.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
viser(UsdGeom.Xformable(cam_insp).AddTransformOp(), oeil_insp, CIBLE_INSP)

rp_vue = rep.create.render_product(str(cam_vue.GetPath()), (args.width, args.height))
annot_vue = rep.AnnotatorRegistry.get_annotator("rgb")
annot_vue.attach([rp_vue])
rp_insp = rep.create.render_product(str(cam_insp.GetPath()), (args.insp_res, args.insp_res))
annot_insp = rep.AnnotatorRegistry.get_annotator("rgb")
annot_insp.attach([rp_insp])

sortie = Path(args.out)
(sortie / "frames").mkdir(parents=True, exist_ok=True)
(sortie / "insp").mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# qui commande le bras
# --------------------------------------------------------------------------

automate = Automate(cellule)
politique = None
if not args.scripte:
    import torch.nn as nn

    ck = torch.load(args.politique, map_location="cuda:0", weights_only=False)

    class Politique(nn.Module):
        def __init__(self, n_obs, n_act):
            super().__init__()
            def tronc(s):
                return nn.Sequential(nn.Linear(n_obs, 512), nn.ELU(),
                                     nn.Linear(512, 256), nn.ELU(),
                                     nn.Linear(256, 128), nn.ELU(), nn.Linear(128, s))
            self.acteur, self.critique = tronc(n_act), tronc(1)
            self.log_ecart = nn.Parameter(torch.zeros(n_act))

    politique = Politique(ck["n_obs"], ck["n_act"]).to("cuda:0")
    politique.load_state_dict(ck["politique"], strict=False)
    politique.eval()
    moy, var = ck["norm"]["moy"].to("cuda:0"), ck["norm"]["var"].to("cuda:0")
    print(f"politique chargee : iteration {ck['iteration']}, "
          f"reussite {ck['reussite']:.1%}", flush=True)


attente_bras = torch.zeros(1, 5, device=cellule.dev)
attente_bras[:, 4] = 1.0            # pince ouverte, bras immobile


def retour_repos():
    """Ramene le bras a sa pose de repos, pince ouverte, sans le teleporter.

    Ce n'est pas une coquetterie de mise en scene. A l'entrainement, CHAQUE
    episode commence bras au repos : laisser le bras la ou le cycle precedent
    l'a laisse, c'est interroger la politique hors de la distribution qu'elle a
    vue. L'automate, lui, y restait carrement bloque — quatre cents pas a
    chercher le chemin du retour depuis les voies de sortie.
    """
    p_tcp, q_tcp = cellule.poses_tcp()
    delta = ((cellule.p_repos.unsqueeze(0) - p_tcp) / 0.020).clamp(-1, 1)
    cible_lacet = torch.full((1,), cellule.lacet_repos, device=cellule.dev)
    dlac = (ecart_lacet(cible_lacet, lacet_de(q_tcp)) / 0.15).clamp(-1, 1)
    return torch.stack([delta[:, 0], delta[:, 1], delta[:, 2], dlac,
                        torch.ones(1, device=cellule.dev)], dim=1)


def au_repos():
    p_tcp, _ = cellule.poses_tcp()
    return float((p_tcp[0] - cellule.p_repos).norm()) < 0.04


def actions():
    """Le geste du bras, par la politique ou par l'automate."""
    if politique is None:
        return automate.agir()
    obs = ((cellule.observer() - moy) / (var.sqrt() + 1e-5)).clamp(-8, 8)
    with torch.no_grad():
        return politique.acteur(obs)                 # moyenne, sans bruit


# --------------------------------------------------------------------------
# les pieces qui defilent
# --------------------------------------------------------------------------

infos = json.loads(Path(args.pieces).read_text())
if args.items:
    infos = infos[: args.items]
racine_decoupes = Path(args.racine_decoupes)
print(f"{len(infos)} pieces, {N_VOIES} voies, "
      f"{'automate' if politique is None else 'politique'} aux commandes", flush=True)


def habiller(info):
    """Pose la decoupe du fromage au fond de l'assiette, taille comme au rendu."""
    tex_fromage.GetInput("file").Set(str(racine_decoupes / info["path"]))
    ar = info["width"] / info["height"]
    taille = 0.55 * (2 * RAYON_ASSIETTE)
    lx, ly = (taille * ar, taille) if ar >= 1 else (taille, taille / ar)
    demi_diag = 0.5 * math.hypot(lx, ly)
    marge = 0.92 * RAYON_ASSIETTE
    if demi_diag > marge:
        f = marge / demi_diag
        lx, ly = lx * f, ly * f
    op_fromage.Set(Gf.Matrix4d().SetScale(Gf.Vec3d(lx, ly, 1.0)) *
                   Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), 37.0)) *
                   Gf.Matrix4d().SetTranslate(Gf.Vec3d(0, 0, FOND_ASSIETTE + 0.001)))


def placer_piece(x):
    """Remet l'assiette et son morceau au depart de la ligne.

    Le bras, lui, n'est pas replace : il finit son geste et repart de la ou il
    est, comme sur une vraie ligne. Le remettre a sa pose de repos serait une
    teleportation bien visible a l'image.
    """
    ids = torch.zeros(1, dtype=torch.long, device=cellule.dev)
    automate.reinitialiser(ids)
    cellule.tenue[ids] = 0
    cellule.pas_ep[ids] = 0
    # On pose la piece nous-memes : il faut donc desamorcer le second temps de
    # remise de la cellule, qui sinon rejouerait par-dessus la pose memorisee
    # a la construction et ramenerait l'assiette 15 cm plus loin.
    cellule.frais_cpt[ids] = 0
    cellule.rejouer[ids] = False
    cellule.a_pincer[ids] = False
    pos = torch.tensor([[x, Y_TAPIS, 0.002]], device=cellule.dev) + cellule.origines
    droit = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=cellule.dev)
    cellule.plats.set_world_poses(pos, droit)
    cellule.plats.set_velocities(torch.zeros(1, 6, device=cellule.dev))
    mor = pos.clone(); mor[:, 2] = Z_MORCEAU + cellule.origines[0, 2]
    cellule.morceaux.set_world_poses(mor, droit)
    cellule.morceaux.set_velocities(torch.zeros(1, 6, device=cellule.dev))
    cellule.remise_ass[ids] = pos
    cellule.remise_mor[ids] = mor


def rendre():
    """Un pas de rendu : les premieres images sortent vides, on reessaie."""
    for _ in range(3):
        rep.orchestrator.step(rt_subframes=args.subframes, pause_timeline=False)
        vue, insp = annot_vue.get_data(), annot_insp.get_data()
        if vue is not None and vue.size and insp is not None and insp.size:
            return np.asarray(vue)[:, :, :3], np.asarray(insp)[:, :, :3]
    return None, None


def classer(image_rgb):
    tampon = io.BytesIO()
    Image.fromarray(image_rgb).save(tampon, format="PNG")
    t0 = time.perf_counter()
    r = requests.post(args.serveur, data=tampon.getvalue(), timeout=60,
                      headers={"Content-Type": "image/png"})
    r.raise_for_status()
    d = r.json()
    d["aller_retour_ms"] = (time.perf_counter() - t0) * 1e3
    return d


print("prechauffe du moteur de rendu...", flush=True)
for _ in range(8):
    rep.orchestrator.step(rt_subframes=2, pause_timeline=False)

# --------------------------------------------------------------------------
# apercu
# --------------------------------------------------------------------------

if args.preview:
    habiller(infos[0])
    cellule.vitesse[:] = 0.0          # l'apercu juge un cadrage, pas un defilement
    placer_piece(X_INSPECT)
    for _ in range(30):
        cellule.agir(actions()); cellule.pas()
    pa_, _ = cellule.pose_assiette()
    print("assiette au poste :", [round(float(v), 4) for v in pa_[0]],
          " camera visee :", [round(float(v), 4) for v in CIBLE_INSP], flush=True)
    vue, insp = rendre()
    if vue is None:
        print("rendu vide", flush=True); app.close(); raise SystemExit(1)
    Image.fromarray(vue).save(sortie / "apercu_vue.png")
    Image.fromarray(insp).save(sortie / "apercu_insp.png")
    try:
        d = classer(insp)
        print(f"attendu {infos[0]['bin']}/{infos[0]['label']} -> {d['status']} "
              f"{d['bin']} ({d['bin_confidence']:.3f})", flush=True)
    except Exception as exc:
        print(f"serveur injoignable : {exc}", flush=True)
    print(f"apercu ecrit dans {sortie}", flush=True)
    app.close()
    raise SystemExit(0)

# --------------------------------------------------------------------------
# la ligne
# --------------------------------------------------------------------------

DT = 1.0 / args.fps
journal = open(sortie / "timeline.jsonl", "w")
compteurs = {v: 0 for v in VOIES}
resultats = []
etat, attente, horloge, retour_k = "amenee", 0.0, 0, 0
avance = 0.0
i_piece, k, t = 0, 0, 0.0
derniere, insp_courante = None, None
t0_total = time.perf_counter()

habiller(infos[0])
placer_piece(args.x_depart)
cellule.vitesse[:] = args.vitesse

while k < args.max_frames and i_piece < len(infos):
    p_ass, _ = cellule.pose_assiette()
    x_ass = float(p_ass[0, 0])

    if etat == "amenee":
        # le tapis avance jusqu'a amener l'assiette sous la camera, puis s'arrete
        if x_ass >= X_INSPECT:
            cellule.vitesse[:] = 0.0
            etat, attente = "inspection", args.dwell
    elif etat == "inspection":
        attente -= DT
    elif etat == "retour":
        retour_k += 1
        if au_repos() or retour_k > 120:
            habiller(infos[i_piece])
            placer_piece(args.x_depart)
            cellule.vitesse[:] = args.vitesse
            etat, derniere, insp_courante = "amenee", None, None
    elif etat == "avance":
        # le tapis amene l'assiette dans la fenetre de prise, puis s'arrete :
        # c'est un poste d'indexation, et c'est la situation dans laquelle la
        # politique a ete entrainee
        if x_ass >= X_PRISE:
            cellule.vitesse[:] = 0.0
            etat, horloge = "tri", 0
    else:                                   # tri : la politique a la main
        horloge += 1

    # Le bras ne travaille que sur la phase de tri : pendant l'amenee et
    # l'inspection il attend, consigne figee. Sinon il va cueillir l'assiette
    # avant qu'elle n'arrive sous la camera, et le poste ne sert plus a rien.
    if etat == "tri":
        geste = actions()
    elif etat == "retour":
        geste = retour_repos()
    else:
        geste = attente_bras
    cellule.agir(geste)
    cellule.pas()

    avance += float(cellule.vitesse[0]) * DT
    amenee.defiler(avance)
    for t_ in tapis_voies:
        t_.defiler(0.6 * t)

    vue, insp = rendre()
    if vue is None:
        print(f"  image {k} vide", flush=True)
        t += DT; k += 1
        continue

    if etat == "inspection" and derniere is None:
        nom_insp = f"i{i_piece:02d}.png"
        Image.fromarray(insp).save(sortie / "insp" / nom_insp)
        try:
            d = classer(insp)
        except Exception as exc:
            print(f"  serveur injoignable ({exc}) : rejet", flush=True)
            d = {"status": "erreur", "bin": None, "bin_confidence": 0.0,
                 "cheese_type": "?", "type_confidence": 0.0, "topk_types": [],
                 "latency_ms": 0.0, "aller_retour_ms": 0.0}
        voie = d["bin"] if d["status"] == "ok" else "reject"
        cellule.cible[:] = VOIES.index(voie)
        info = infos[i_piece]
        attendu = "reject" if info["bin"] == "not_cheese" else info["bin"]
        insp_courante = nom_insp
        derniere = {"piece": i_piece, "uid": info["uid"], "verite_label": info["label"],
                    "verite_voie": attendu, "voie": voie, "juste": voie == attendu,
                    "insp": nom_insp, **d}
        print(f"[{i_piece + 1}/{len(infos)}] {info['label']:22s} -> {d['status']:10s} "
              f"{voie:14s} {d['bin_confidence']:.3f} "
              f"{'OK' if derniere['juste'] else 'RATE (attendu ' + attendu + ')'}", flush=True)

    if etat == "inspection" and attente <= 0:
        # On repart, mais seulement jusqu'a la fenetre de prise. S'arreter sous
        # la camera laissait l'assiette a 62 cm en amont du bras, c'est-a-dire
        # au bord de son espace de travail : l'automate n'y arrivait jamais.
        etat = "avance"
        cellule.vitesse[:] = args.vitesse

    fini = False
    if etat == "tri":
        r, term, infos_ep = cellule.evaluer()
        posee = bool(infos_ep["posee"][0])
        rate = bool(infos_ep["renversee"][0] or infos_ep["perdue"][0])
        if posee or rate or horloge > args.delai_cycle:
            issue = "posee" if posee else ("ratee" if rate else "delai")
            if posee:
                compteurs[VOIES[int(cellule.cible[0])]] += 1
            resultats.append({**(derniere or {}), "issue": issue,
                              "voie_finale": VOIES[int(cellule.cible[0])]})
            print(f"        -> {issue} sur {VOIES[int(cellule.cible[0])]} "
                  f"({horloge} pas)", flush=True)
            i_piece += 1
            fini = True
            if i_piece < len(infos):
                # on laisse l'assiette posee a l'image, le temps que le bras
                # remonte : c'est le moment ou l'on VOIT le resultat du tri
                etat, retour_k = "retour", 0
                cellule.vitesse[:] = 0.0

    Image.fromarray(vue).save(sortie / "frames" / f"f{k:05d}.png")
    p_ass, _ = cellule.pose_assiette()
    journal.write(json.dumps({
        "f": k, "t": round(t, 4), "etat": etat, "piece": i_piece,
        "inspection": etat == "inspection", "insp": insp_courante,
        "derniere": derniere, "compteurs": dict(compteurs),
        "traitees": len(resultats),
        # ce que fait le bras, pour le bandeau de la video
        "tenue": bool(cellule.tenue[0] > 0.5),
        "z_assiette": round(float(p_ass[0, 2]), 4),
        "cycle": horloge,
        "ratees": sum(1 for r_ in resultats if r_.get("issue") != "posee"),
        "justes": sum(1 for r_ in resultats if r_.get("juste")),
        "posees": sum(1 for r_ in resultats if r_.get("issue") == "posee"),
    }) + "\n")
    journal.flush()

    t += DT
    k += 1
    if k % 25 == 0:
        ecoule = time.perf_counter() - t0_total
        print(f"  image {k}  t={t:5.1f}s  {ecoule / k:.2f}s/image  "
              f"{etat:10s} bras {Automate.NOMS[int(automate.etat[0])]:11s}"
              f" assiette {[round(float(v), 3) for v in p_ass[0]]}", flush=True)

journal.close()
posees = sum(1 for r_ in resultats if r_["issue"] == "posee")
justes = sum(1 for r_ in resultats if r_.get("juste"))
resume = {"images": k, "fps": args.fps, "duree_s": round(k / args.fps, 2),
          "pieces": len(resultats), "traitees": len(resultats),   # lu par make_video
          "posees": posees, "justes": justes,
          "compteurs": compteurs, "voies": VOIES,
          "pilote": "automate" if politique is None else "politique",
          "resultats": resultats}
(sortie / "resume.json").write_text(json.dumps(resume, indent=1))
print(f"\n{k} images ({k / args.fps:.1f}s) — {posees}/{len(resultats)} assiettes posees, "
      f"{justes}/{len(resultats)} dans le bon bac", flush=True)
app.close()
