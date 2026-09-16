"""Rend des pieces de fromage posees dans une assiette sur un tapis roulant.

Chaque decoupe RGBA (silhouette issue des polygones de Food Recognition) est
appliquee sur un quad pose au fond d'une assiette, et la scene est rendue sous
plusieurs angles avec randomisation de domaine (lumiere, camera, materiaux).

    /isaac-sim/python.sh render_belt.py --cutouts N --views M --out DIR

Les N*M images heritent du bac de la decoupe source. Le champ `group` du
manifeste porte l'uid de la decoupe : toutes les vues d'une meme piece doivent
rester dans le meme split, sinon on remesure une fuite.
"""

import argparse, json, math, random, sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--cutouts", type=int, default=8, help="nombre de pieces distinctes")
parser.add_argument("--views", type=int, default=3, help="vues par piece")
parser.add_argument("--out", default="/workspace/sim/out")
parser.add_argument("--res", type=int, default=768)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--manifest", default="/workspace/data/processed/cutouts/manifest.csv")
parser.add_argument("--root", default="/workspace/data/processed")
parser.add_argument("--min-fill", type=float, default=0.0,
                    help="ecarte les silhouettes en lamelle (part de la boite couverte "
                         "par le polygone). NB : ne detecte pas les decoupes qui "
                         "emportent du pain, dont le polygone est au contraire bien plein")
parser.add_argument("--max-ar", type=float, default=0.0,
                    help="rapport d'aspect maximal, 0 = pas de limite")
parser.add_argument("--elev-min", type=float, default=30.0)
parser.add_argument("--elev-max", type=float, default=78.0)
parser.add_argument("--shard", default="0/1", help="i/n : decoupe le travail entre GPU")
parser.add_argument("--all", action="store_true", help="toutes les decoupes")
parser.add_argument("--empty", type=int, default=0,
                    help="rend N images de tapis VIDE (classe `empty`) au lieu de "
                         "parcourir les decoupes. Le modele doit savoir dire qu'il "
                         "n'y a rien a ramasser.")
parser.add_argument("--skip-existing", action="store_true",
                    help="passe les pieces dont toutes les vues sont deja sur disque")
args = parser.parse_args()

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "renderer": "RaytracedLighting",
                     "width": args.res, "height": args.res})

import omni.usd, omni.replicator.core as rep
from pxr import UsdGeom, UsdShade, UsdLux, Gf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from usd_kit import Tapis, disque, materiau_texture, materiau_uni, quad, teinter, viser

rng = random.Random(args.seed)
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
monde = UsdGeom.Xform.Define(stage, "/World")


# tapis roulant : dalle sombre, longerons et lattes transversales
tapis = Tapis(stage, "/World/tapis")
mat_tapis = tapis.mat_dalle

RAYON_ASSIETTE = 0.22          # rayon utile ou peut tenir une piece
assiette = disque(stage, "/World/assiette", RAYON_ASSIETTE, 0.30, 0.010, 0.030)
mat_assiette = materiau_uni(stage, "/World/mat_assiette", (0.92, 0.92, 0.90), rugosite=0.25)
UsdShade.MaterialBindingAPI(assiette).Bind(mat_assiette)

# la piece de fromage
piece = quad(stage, "/World/fromage")
mat_piece, tex_piece = materiau_texture(stage, "/World/mat_fromage")
UsdShade.MaterialBindingAPI(piece).Bind(mat_piece)
op_piece = UsdGeom.Xformable(piece).AddTransformOp()
vis_piece = UsdGeom.Imageable(piece)
vis_assiette = UsdGeom.Imageable(assiette)

# lumieres
dome = UsdLux.DomeLight.Define(stage, "/World/dome")
dome.CreateIntensityAttr(300.0)
cle = UsdLux.SphereLight.Define(stage, "/World/cle")
cle.CreateRadiusAttr(0.12); cle.CreateIntensityAttr(30000.0)
op_cle = UsdGeom.Xformable(cle).AddTranslateOp()

# camera
cam = UsdGeom.Camera.Define(stage, "/World/camera")
cam.CreateFocalLengthAttr(28.0)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
op_cam = UsdGeom.Xformable(cam).AddTransformOp()


# --------------------------------------------------------------------------
# rendu
# --------------------------------------------------------------------------

rp = rep.create.render_product(str(cam.GetPath()), (args.res, args.res))
annot = rep.AnnotatorRegistry.get_annotator("rgb")
annot.attach([rp])

import csv
racine = Path(args.root)
lignes = [r for r in csv.DictReader(open(args.manifest))]

avant = len(lignes)
if args.min_fill > 0:
    lignes = [r for r in lignes if float(r["fill_ratio"]) >= args.min_fill]
if args.max_ar > 0:
    def _ar(r):
        w, h = float(r["width"]), float(r["height"])
        return max(w / h, h / w)
    lignes = [r for r in lignes if _ar(r) <= args.max_ar]
print(f"{len(lignes)}/{avant} decoupes retenues apres filtres", flush=True)

# ATTENTION : le melange doit etre IDENTIQUE dans tous les shards, sinon chacun
# decoupe un ordre different et ils se recouvrent au lieu de se partager le
# travail (un quart des pieces rendu deux fois, un quart jamais). La graine de
# partage est donc figee ; `--seed` ne pilote que la randomisation de domaine.
random.Random(1234).shuffle(lignes)
if not args.all:
    lignes = lignes[: args.cutouts]

i_shard, n_shard = (int(x) for x in args.shard.split("/"))
choisies = lignes[i_shard::n_shard]
if args.skip_existing:
    avant_reprise = len(choisies)
    choisies = [r for r in choisies
                if not all((Path(args.out) / f"{r['bin']}__{r['uid']}__v{v}.png").exists()
                           for v in range(args.views))]
    print(f"reprise : {avant_reprise - len(choisies)} pieces deja completes, "
          f"{len(choisies)} a faire", flush=True)
print(f"shard {i_shard}/{n_shard} : {len(choisies)} pieces x {args.views} vues", flush=True)

sortie = Path(args.out); sortie.mkdir(parents=True, exist_ok=True)
import numpy as np
from PIL import Image

# Les premieres frames echouent tant que le moteur n'a pas fini de monter
# ("renderer failed to advance to the scheduled frame") : on les brule a vide.
print("prechauffe du moteur de rendu...", flush=True)
for _ in range(8):
    rep.orchestrator.step(rt_subframes=4)

def randomiser_scene():
    """Tirage commun a toutes les vues : camera, lumieres, teintes."""
    azimut = rng.uniform(0, 360)
    elevation = rng.uniform(args.elev_min, args.elev_max)
    dist = rng.uniform(0.85, 1.45)
    ar_ = math.radians(azimut); el = math.radians(elevation)
    oeil = Gf.Vec3d(dist * math.cos(el) * math.cos(ar_),
                    dist * math.cos(el) * math.sin(ar_),
                    dist * math.sin(el))
    viser(op_cam, oeil, Gf.Vec3d(0, 0, 0.02))
    cam.GetFocalLengthAttr().Set(rng.uniform(20.0, 32.0))
    al = math.radians(rng.uniform(0, 360)); el2 = math.radians(rng.uniform(40, 85))
    dl = rng.uniform(0.8, 1.6)
    op_cle.Set(Gf.Vec3d(dl * math.cos(el2) * math.cos(al),
                        dl * math.cos(el2) * math.sin(al), dl * math.sin(el2)))
    cle.GetIntensityAttr().Set(rng.uniform(12000, 60000))
    dome.GetIntensityAttr().Set(rng.uniform(120, 700))
    teinte = rng.uniform(0.88, 1.0)
    cle.CreateColorAttr(Gf.Vec3f(1.0, teinte, rng.uniform(teinte, 1.0)))
    g = rng.uniform(0.80, 0.97)
    teinter(mat_assiette, (g, g, g * rng.uniform(0.96, 1.0)))
    n = rng.uniform(0.05, 0.18)
    teinter(mat_tapis, (n, n, n * rng.uniform(1.0, 1.2)))
    return azimut, elevation


def rendre():
    """Un pas de rendu, avec quelques essais si le moteur n'a pas suivi."""
    for _ in range(3):
        rep.orchestrator.step(rt_subframes=12)
        img = annot.get_data()
        if img is not None and img.size:
            return np.asarray(img)[:, :, :3]
    return None


meta = []

if args.empty:
    # Tapis vide : la moitie des images sans rien du tout, l'autre moitie avec une
    # assiette vide. Le robot doit distinguer "rien a ramasser" de "objet inconnu".
    vis_piece.MakeInvisible()
    for k in range(args.empty):
        avec_assiette = (k % 2 == 0)
        (vis_assiette.MakeVisible if avec_assiette else vis_assiette.MakeInvisible)()
        azimut, elevation = randomiser_scene()
        arr = rendre()
        if arr is None:
            print(f"  vue vide ratee #{k}", flush=True)
            continue
        uid = f"empty{k:05d}"
        nom = f"empty__{uid}__v0.png"
        Image.fromarray(arr).save(sortie / nom)
        meta.append({"file": nom, "bin": "empty", "label": "empty",
                     "group": uid, "view": 0, "split_src": "",
                     "azimut": round(azimut, 1), "elevation": round(elevation, 1)})
        if (k + 1) % 100 == 0:
            print(f"[{k+1}/{args.empty}] tapis vide", flush=True)
    with open(sortie / f"manifest_empty{i_shard}.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"\n{len(meta)} images de tapis vide ecrites dans {sortie}")
    app.close()
    raise SystemExit(0)

vis_piece.MakeVisible(); vis_assiette.MakeVisible()
for i, r in enumerate(choisies):
    chemin_tex = racine / r["path"]
    tex_piece.GetInput("file").Set(str(chemin_tex))

    with Image.open(chemin_tex) as im:
        ar = im.width / im.height
    # la piece occupe une fraction du diametre utile, et ne peut pas deborder :
    # sa demi-diagonale doit rester sous le rayon de l'assiette
    taille = rng.uniform(0.35, 0.75) * (2 * RAYON_ASSIETTE)
    lx, ly = (taille * ar, taille) if ar >= 1 else (taille, taille / ar)
    demi_diag = 0.5 * math.hypot(lx, ly)
    marge = 0.92 * RAYON_ASSIETTE
    if demi_diag > marge:
        k = marge / demi_diag
        lx, ly = lx * k, ly * k

    for v in range(args.views):
        # --- randomisation de domaine ---
        rot = rng.uniform(0, 360)
        libre = max(0.0, 0.92 * RAYON_ASSIETTE - 0.5 * math.hypot(lx, ly))
        dx, dy = rng.uniform(-libre, libre), rng.uniform(-libre, libre)
        op_piece.Set(
            Gf.Matrix4d().SetScale(Gf.Vec3d(lx, ly, 1.0)) *
            Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), rot)) *
            Gf.Matrix4d().SetTranslate(Gf.Vec3d(dx, dy, 0.013)))

        azimut = rng.uniform(0, 360)
        elevation = rng.uniform(args.elev_min, args.elev_max)          # vue plongeante, comme une cam de tapis
        dist = rng.uniform(0.85, 1.45)
        ar_ = math.radians(azimut); el = math.radians(elevation)
        oeil = Gf.Vec3d(dist * math.cos(el) * math.cos(ar_),
                        dist * math.cos(el) * math.sin(ar_),
                        dist * math.sin(el))
        viser(op_cam, oeil, Gf.Vec3d(dx * 0.5, dy * 0.5, 0.02))
        cam.GetFocalLengthAttr().Set(rng.uniform(20.0, 32.0))

        al = math.radians(rng.uniform(0, 360)); el2 = math.radians(rng.uniform(40, 85))
        dl = rng.uniform(0.8, 1.6)
        op_cle.Set(Gf.Vec3d(dl * math.cos(el2) * math.cos(al),
                            dl * math.cos(el2) * math.sin(al),
                            dl * math.sin(el2)))
        cle.GetIntensityAttr().Set(rng.uniform(12000, 60000))
        dome.GetIntensityAttr().Set(rng.uniform(120, 700))
        teinte = rng.uniform(0.88, 1.0)
        cle.CreateColorAttr(Gf.Vec3f(1.0, teinte, rng.uniform(teinte, 1.0)))
        g = rng.uniform(0.80, 0.97)
        teinter(mat_assiette, (g, g, g * rng.uniform(0.96, 1.0)))
        n = rng.uniform(0.05, 0.18)
        teinter(mat_tapis, (n, n, n * rng.uniform(1.0, 1.2)))

        img = None
        for essai in range(3):
            rep.orchestrator.step(rt_subframes=12)
            img = annot.get_data()
            if img is not None and img.size:
                break
        if img is None or not img.size:
            print(f"  vue vide pour {r['uid']} apres 3 essais", flush=True)
            continue
        arr = np.asarray(img)[:, :, :3]
        nom = f"{r['bin']}__{r['uid']}__v{v}.png"
        Image.fromarray(arr).save(sortie / nom)
        meta.append({"file": nom, "bin": r["bin"], "label": r["label"],
                     "group": r["uid"], "view": v, "split_src": r["split"],
                     "azimut": round(azimut, 1), "elevation": round(elevation, 1)})
    print(f"[{i+1}/{len(choisies)}] {r['bin']:16s} {r['uid']} -> {args.views} vues", flush=True)

with open(sortie / f"manifest_{i_shard}.json", "w") as fh:
    json.dump(meta, fh, indent=1)
print(f"\n{len(meta)} images ecrites dans {sortie}")
app.close()
