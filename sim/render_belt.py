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
from pxr import Usd, UsdGeom, UsdShade, UsdLux, Sdf, Gf

rng = random.Random(args.seed)
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
monde = UsdGeom.Xform.Define(stage, "/World")


# --------------------------------------------------------------------------
# materiaux
# --------------------------------------------------------------------------

def materiau_uni(chemin, couleur, rugosite=0.5, metal=0.0):
    mat = UsdShade.Material.Define(stage, chemin)
    sh = UsdShade.Shader.Define(stage, chemin + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*couleur))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rugosite)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def materiau_texture(chemin):
    """UsdPreviewSurface alimente par une texture RGBA : rgb -> diffuse, a -> opacite.

    L'alpha de la decoupe devient le masque de decoupe du quad, donc la piece a
    la silhouette du fromage et pas celle d'un rectangle.
    """
    mat = UsdShade.Material.Define(stage, chemin)
    sh = UsdShade.Shader.Define(stage, chemin + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    sh.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.5)

    lecteur = UsdShade.Shader.Define(stage, chemin + "/uv")
    lecteur.CreateIdAttr("UsdPrimvarReader_float2")
    lecteur.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    lecteur.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    tex = UsdShade.Shader.Define(stage, chemin + "/tex")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        lecteur.ConnectableAPI(), "result")
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    tex.CreateOutput("a", Sdf.ValueTypeNames.Float)

    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        tex.ConnectableAPI(), "rgb")
    sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(
        tex.ConnectableAPI(), "a")
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat, tex


# --------------------------------------------------------------------------
# geometrie
# --------------------------------------------------------------------------

def quad(chemin, demi=0.5):
    """Quad unitaire dans le plan XY, avec UV, destine a porter la decoupe."""
    m = UsdGeom.Mesh.Define(stage, chemin)
    m.CreatePointsAttr([Gf.Vec3f(-demi, -demi, 0), Gf.Vec3f(demi, -demi, 0),
                        Gf.Vec3f(demi, demi, 0), Gf.Vec3f(-demi, demi, 0)])
    m.CreateFaceVertexCountsAttr([4])
    m.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    m.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)
    UsdGeom.PrimvarsAPI(m).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    ).Set([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)])
    m.CreateDoubleSidedAttr(True)
    return m


def disque(chemin, rayon_int, rayon_ext, z_fond, z_bord, segments=96):
    """Assiette : disque plat + rebord releve, maille finement.

    Le cylindre USD par defaut se tesselle en une dizaine de facettes et se voit
    comme un decagone dans le rendu. On construit donc le maillage a la main :
    un eventail central, puis un anneau qui remonte vers le bord.
    """
    pts, faces, counts, sts = [], [], [], []
    pts.append(Gf.Vec3f(0, 0, z_fond)); sts.append(Gf.Vec2f(0.5, 0.5))
    for i in range(segments):
        a = 2 * math.pi * i / segments
        c, si = math.cos(a), math.sin(a)
        pts.append(Gf.Vec3f(rayon_int * c, rayon_int * si, z_fond))
        sts.append(Gf.Vec2f(0.5 + 0.25 * c, 0.5 + 0.25 * si))
    for i in range(segments):
        a = 2 * math.pi * i / segments
        c, si = math.cos(a), math.sin(a)
        pts.append(Gf.Vec3f(rayon_ext * c, rayon_ext * si, z_bord))
        sts.append(Gf.Vec2f(0.5 + 0.5 * c, 0.5 + 0.5 * si))
    for i in range(segments):                       # eventail central
        faces += [0, 1 + i, 1 + (i + 1) % segments]; counts.append(3)
    for i in range(segments):                       # anneau du rebord
        a, b = 1 + i, 1 + (i + 1) % segments
        c_, d_ = 1 + segments + i, 1 + segments + (i + 1) % segments
        faces += [a, b, d_, c_]; counts.append(4)

    m = UsdGeom.Mesh.Define(stage, chemin)
    m.CreatePointsAttr(pts)
    m.CreateFaceVertexCountsAttr(counts)
    m.CreateFaceVertexIndicesAttr(faces)
    m.CreateSubdivisionSchemeAttr("none")
    UsdGeom.PrimvarsAPI(m).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(sts)
    m.CreateDoubleSidedAttr(True)
    return m


# tapis roulant : dalle sombre, plus deux longerons pour qu'on le lise comme un tapis
tapis = UsdGeom.Cube.Define(stage, "/World/tapis")
tapis.CreateSizeAttr(1.0)
UsdGeom.Xformable(tapis).AddTransformOp().Set(
    Gf.Matrix4d().SetScale(Gf.Vec3d(4.0, 1.1, 0.04)) *
    Gf.Matrix4d().SetTranslate(Gf.Vec3d(0, 0, -0.02)))
mat_tapis = materiau_uni("/World/mat_tapis", (0.09, 0.09, 0.10), rugosite=0.85)
UsdShade.MaterialBindingAPI(tapis).Bind(mat_tapis)

mat_rail = materiau_uni("/World/mat_rail", (0.30, 0.31, 0.33), rugosite=0.35, metal=0.7)
for cote, y in (("g", 0.60), ("d", -0.60)):
    rail = UsdGeom.Cube.Define(stage, f"/World/rail_{cote}")
    rail.CreateSizeAttr(1.0)
    UsdGeom.Xformable(rail).AddTransformOp().Set(
        Gf.Matrix4d().SetScale(Gf.Vec3d(4.0, 0.05, 0.10)) *
        Gf.Matrix4d().SetTranslate(Gf.Vec3d(0, y, 0.01)))
    UsdShade.MaterialBindingAPI(rail).Bind(mat_rail)

# lattes transversales du tapis, pour donner de la texture au fond
mat_latte = materiau_uni("/World/mat_latte", (0.13, 0.13, 0.15), rugosite=0.75)
for i in range(26):
    latte = UsdGeom.Cube.Define(stage, f"/World/latte_{i}")
    latte.CreateSizeAttr(1.0)
    UsdGeom.Xformable(latte).AddTransformOp().Set(
        Gf.Matrix4d().SetScale(Gf.Vec3d(0.02, 1.1, 0.012)) *
        Gf.Matrix4d().SetTranslate(Gf.Vec3d(-1.95 + i * 0.155, 0, 0.002)))
    UsdShade.MaterialBindingAPI(latte).Bind(mat_latte)

RAYON_ASSIETTE = 0.22          # rayon utile ou peut tenir une piece
assiette = disque("/World/assiette", RAYON_ASSIETTE, 0.30, 0.010, 0.030)
mat_assiette = materiau_uni("/World/mat_assiette", (0.92, 0.92, 0.90), rugosite=0.25)
UsdShade.MaterialBindingAPI(assiette).Bind(mat_assiette)

# la piece de fromage
piece = quad("/World/fromage")
mat_piece, tex_piece = materiau_texture("/World/mat_fromage")
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


def viser(op, oeil, cible, haut=Gf.Vec3d(0, 0, 1)):
    """Oriente un prim vers `cible` : la transforme camera est l'inverse de la vue."""
    op.Set(Gf.Matrix4d().SetLookAt(oeil, cible, haut).GetInverse())


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
    mat_assiette.GetPrim().GetChild("Shader").GetAttribute("inputs:diffuseColor").Set(
        Gf.Vec3f(g, g, g * rng.uniform(0.96, 1.0)))
    n = rng.uniform(0.05, 0.18)
    mat_tapis.GetPrim().GetChild("Shader").GetAttribute("inputs:diffuseColor").Set(
        Gf.Vec3f(n, n, n * rng.uniform(1.0, 1.2)))
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
        mat_assiette.GetPrim().GetChild("Shader").GetAttribute("inputs:diffuseColor").Set(
            Gf.Vec3f(g, g, g * rng.uniform(0.96, 1.0)))
        n = rng.uniform(0.05, 0.18)
        mat_tapis.GetPrim().GetChild("Shader").GetAttribute("inputs:diffuseColor").Set(
            Gf.Vec3f(n, n, n * rng.uniform(1.0, 1.2)))

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
