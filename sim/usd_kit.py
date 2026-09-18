"""Primitives USD partagees par les scenes Isaac Sim du projet.

A importer **apres** `SimulationApp`, sinon `pxr` n'est pas encore disponible.
Le tapis, l'assiette et le quad texture sont ceux qui ont servi a fabriquer le
jeu d'entrainement : la scene de tri les reutilise telles quelles, pour que la
camera d'inspection voie exactement le domaine sur lequel le modele a appris.
"""

from __future__ import annotations

import math

from pxr import UsdGeom, UsdShade, Sdf, Gf


# --------------------------------------------------------------------------
# materiaux
# --------------------------------------------------------------------------

def materiau_uni(stage, chemin, couleur, rugosite=0.5, metal=0.0):
    mat = UsdShade.Material.Define(stage, chemin)
    sh = UsdShade.Shader.Define(stage, chemin + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*couleur))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rugosite)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def teinter(materiau, couleur):
    """Change la couleur diffuse d'un materiau uni deja cree."""
    materiau.GetPrim().GetChild("Shader").GetAttribute("inputs:diffuseColor").Set(
        Gf.Vec3f(*couleur))


def emettre(materiau, couleur):
    """Allume l'emissif d'un materiau uni : sert a surligner une voie active."""
    sh = UsdShade.Shader(materiau.GetPrim().GetChild("Shader"))
    entree = sh.GetInput("emissiveColor") or sh.CreateInput(
        "emissiveColor", Sdf.ValueTypeNames.Color3f)
    entree.Set(Gf.Vec3f(*couleur))


def materiau_texture(stage, chemin, fichier: str = ""):
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
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(fichier)
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

def quad(stage, chemin, demi=0.5):
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


def disque(stage, chemin, rayon_int, rayon_ext, z_fond, z_bord, segments=96):
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


def assiette(stage, chemin, rayon=0.090, hauteur=0.045, epaisseur=0.008,
             fond=0.008, segments=96):
    """Assiette **prehensible** : un vrai volume ferme, pas une coque.

    `disque()` ci-dessus est une surface d'epaisseur nulle : parfaite pour un
    rendu, impossible a saisir — PhysX n'a rien a pincer et la pince traverse.
    Celle-ci est un plateau creux : un socle plein, une paroi verticale de
    `epaisseur` qui retient ce qu'on transporte.

    A 68 mm de diametre pour 80 mm d'ouverture, la pince du FR3 ENJAMBE
    l'assiette et serre sur le socle, de part et d'autre. C'est ce qui rend la
    prise tenable : la ligne de serrage passe par l'axe du centre de masse, et
    l'assiette pend a plat. La saisir par le bord, comme le laissait croire la
    premiere version de cette fonction, la fait pendre a un rayon de son centre
    de masse — elle bascule, et ce qu'elle porte tombe.

           |<--------- 2 * rayon (68 mm) --------->|
        >|#|#                                   #|#|<  <- les DEUX mors
           #.#                                    #.#   ^ hauteur
           #######################################.##   | fond (plein)
           |<-->| epaisseur

    Le maillage est ferme (paroi exterieure, couronne du bord, paroi
    interieure, dessus du fond, dessous) pour que la decomposition convexe de
    PhysX lui donne un collider correct. Comme `disque()`, il est construit a la
    main : un cylindre USD se tesselle en decagone visible.
    """
    n = segments
    pts, sts, faces, counts = [], [], [], []
    r_int = rayon - epaisseur

    def anneau(r, z):
        """Ajoute n points sur un cercle, renvoie l'index du premier."""
        base = len(pts)
        for i in range(n):
            a = 2 * math.pi * i / n
            c, s_ = math.cos(a), math.sin(a)
            pts.append(Gf.Vec3f(r * c, r * s_, z))
            sts.append(Gf.Vec2f(0.5 + 0.5 * (r / rayon) * c,
                                0.5 + 0.5 * (r / rayon) * s_))
        return base

    A = anneau(rayon, 0.0)          # bas exterieur
    B = anneau(rayon, hauteur)      # haut exterieur
    C = anneau(r_int, hauteur)      # haut interieur
    D = anneau(r_int, fond)         # naissance du fond

    c_haut = len(pts); pts.append(Gf.Vec3f(0, 0, fond)); sts.append(Gf.Vec2f(0.5, 0.5))
    c_bas = len(pts); pts.append(Gf.Vec3f(0, 0, 0.0)); sts.append(Gf.Vec2f(0.5, 0.5))

    for i in range(n):
        j = (i + 1) % n
        faces += [A + i, A + j, B + j, B + i]; counts.append(4)   # paroi exterieure
        faces += [B + i, B + j, C + j, C + i]; counts.append(4)   # couronne du bord
        faces += [C + i, C + j, D + j, D + i]; counts.append(4)   # paroi interieure
        faces += [c_haut, D + i, D + j]; counts.append(3)         # dessus du fond
        faces += [c_bas, A + j, A + i]; counts.append(3)          # dessous

    m = UsdGeom.Mesh.Define(stage, chemin)
    m.CreatePointsAttr(pts)
    m.CreateFaceVertexCountsAttr(counts)
    m.CreateFaceVertexIndicesAttr(faces)
    m.CreateSubdivisionSchemeAttr("none")
    m.CreateExtentAttr([Gf.Vec3f(-rayon, -rayon, 0.0), Gf.Vec3f(rayon, rayon, hauteur)])
    UsdGeom.PrimvarsAPI(m).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(sts)
    return m


def boite(stage, chemin, taille, centre, materiau=None):
    """Cube unitaire mis a l'echelle et place : mur, dalle, poteau."""
    c = UsdGeom.Cube.Define(stage, chemin)
    c.CreateSizeAttr(1.0)
    UsdGeom.Xformable(c).AddTransformOp().Set(
        Gf.Matrix4d().SetScale(Gf.Vec3d(*taille)) *
        Gf.Matrix4d().SetTranslate(Gf.Vec3d(*centre)))
    if materiau is not None:
        UsdShade.MaterialBindingAPI(c).Bind(materiau)
    return c


def boite_orientee(stage, chemin, taille, centre, materiau=None,
                   rotation=(0.0, 0.0, 0.0)):
    """Boite placee avec une rotation XYZ en degres.

    Ce petit primitive suffit pour les toboggans, renforts et pare-chocs de la
    ligne sans introduire de maillages lourds. La transformation reste locale
    au Xform parent, ce qui permet de reorienter un module industriel complet.
    """
    c = UsdGeom.Cube.Define(stage, chemin)
    c.CreateSizeAttr(1.0)
    rx, ry, rz = rotation
    rotation_q = (Gf.Rotation(Gf.Vec3d(1, 0, 0), rx) *
                  Gf.Rotation(Gf.Vec3d(0, 1, 0), ry) *
                  Gf.Rotation(Gf.Vec3d(0, 0, 1), rz))
    UsdGeom.Xformable(c).AddTransformOp().Set(
        Gf.Matrix4d().SetScale(Gf.Vec3d(*taille)) *
        Gf.Matrix4d().SetRotate(rotation_q) *
        Gf.Matrix4d().SetTranslate(Gf.Vec3d(*centre)))
    if materiau is not None:
        UsdShade.MaterialBindingAPI(c).Bind(materiau)
    return c


def module_reception(stage, chemin, centre, angle, couleur, z_sol,
                     mat_inox, mat_tote, mat_sombre):
    """Cree un module de reception alimentaire compact et reutilisable.

    Le repere local pointe dans le sens de sortie du convoyeur (+X). Le bac
    amovible reste sous le niveau de transfert, tandis qu'un toboggan incline
    guide la piece depuis le bout du tapis. La couleur de classe n'est qu'un
    code visuel (rive et bandeau), pas la matiere de tout le bac.
    """
    racine = UsdGeom.Xform.Define(stage, chemin)
    UsdGeom.Xformable(racine).AddTransformOp().Set(
        Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), angle)) *
        Gf.Matrix4d().SetTranslate(Gf.Vec3d(*centre)))

    mat_accent = materiau_uni(stage, chemin + "/mat_accent", couleur,
                              rugosite=0.42, metal=0.15)

    # Chassis inox : quatre montants, traverses et pieds reglables.
    for i, (x, y) in enumerate(((-0.31, -0.31), (-0.31, 0.31),
                                (0.31, -0.31), (0.31, 0.31))):
        boite(stage, f"{chemin}/chassis/montant_{i}", (0.045, 0.045, 0.68),
              (x, y, z_sol + 0.36), mat_inox)
        boite(stage, f"{chemin}/chassis/pied_{i}", (0.105, 0.105, 0.025),
              (x, y, z_sol + 0.013), mat_sombre)
    for i, y in enumerate((-0.31, 0.31)):
        boite(stage, f"{chemin}/chassis/traverse_{i}", (0.70, 0.045, 0.045),
              (0.0, y, z_sol + 0.12), mat_inox)

    # Tote amovible, neutre et ouvert. Sa face avant porte le code couleur.
    z_tote = z_sol + 0.30
    boite(stage, chemin + "/tote/fond", (0.60, 0.58, 0.035),
          (0.02, 0.0, z_sol + 0.08), mat_tote)
    boite(stage, chemin + "/tote/avant", (0.035, 0.58, 0.46),
          (-0.28, 0.0, z_tote), mat_tote)
    boite(stage, chemin + "/tote/arriere", (0.035, 0.58, 0.46),
          (0.32, 0.0, z_tote), mat_tote)
    for i, y in enumerate((-0.29, 0.29)):
        boite(stage, f"{chemin}/tote/cote_{i}", (0.60, 0.035, 0.46),
              (0.02, y, z_tote), mat_tote)
    boite(stage, chemin + "/tote/bandeau", (0.025, 0.46, 0.075),
          (-0.301, 0.0, z_sol + 0.38), mat_accent)

    # Entonnoir et rive de transfert, ouverts vers le convoyeur (-X).
    boite_orientee(stage, chemin + "/chute/plaque", (0.50, 0.64, 0.025),
                   (-0.43, 0.0, -0.16), mat_inox, rotation=(0, -18, 0))
    for i, y in enumerate((-0.34, 0.34)):
        boite_orientee(stage, f"{chemin}/chute/rive_{i}", (0.50, 0.035, 0.16),
                       (-0.43, y, -0.09), mat_inox, rotation=(0, -18, 0))
        boite(stage, f"{chemin}/chute/accent_{i}", (0.055, 0.045, 0.11),
              (-0.68, y, -0.01), mat_accent)

    # Pare-chocs basse energie et poignee du tote.
    boite(stage, chemin + "/bumper", (0.09, 0.72, 0.13),
          (0.43, 0.0, z_sol + 0.18), mat_sombre)
    boite(stage, chemin + "/tote/poignee", (0.035, 0.25, 0.035),
          (0.345, 0.0, z_sol + 0.43), mat_inox)
    return racine


def viser(op, oeil, cible, haut=Gf.Vec3d(0, 0, 1)):
    """Oriente un prim vers `cible` : la transforme camera est l'inverse de la vue."""
    op.Set(Gf.Matrix4d().SetLookAt(oeil, cible, haut).GetInverse())


def orbite(centre, azimut, elevation, distance):
    """Point sur une sphere autour de `centre`, angles en degres."""
    a, e = math.radians(azimut), math.radians(elevation)
    return Gf.Vec3d(centre[0] + distance * math.cos(e) * math.cos(a),
                    centre[1] + distance * math.cos(e) * math.sin(a),
                    centre[2] + distance * math.sin(e))


# --------------------------------------------------------------------------
# tapis roulant
# --------------------------------------------------------------------------

class Tapis:
    """Dalle sombre, deux longerons, et des lattes transversales qui defilent.

    Les dimensions par defaut sont celles du tapis d'entrainement
    (`render_belt.py`) : 4 m de long, 1.1 m de large, lattes tous les 15.5 cm.
    """

    PAS = 0.155

    def __init__(self, stage, chemin, longueur=4.0, largeur=1.1, rails=True,
                 couleur_dalle=(0.09, 0.09, 0.10),
                 couleur_rail=(0.30, 0.31, 0.33),
                 couleur_latte=(0.13, 0.13, 0.15)):
        self.stage = stage
        self.longueur, self.largeur = longueur, largeur
        self.racine = UsdGeom.Xform.Define(stage, chemin)
        self.op = UsdGeom.Xformable(self.racine).AddTransformOp()
        self.op.Set(Gf.Matrix4d(1.0))

        self.mat_dalle = materiau_uni(stage, chemin + "/mat_dalle", couleur_dalle, 0.85)
        self.mat_rail = materiau_uni(stage, chemin + "/mat_rail", couleur_rail, 0.35, 0.7)
        self.mat_latte = materiau_uni(stage, chemin + "/mat_latte", couleur_latte, 0.75)

        boite(stage, chemin + "/dalle", (longueur, largeur, 0.04),
              (0, 0, -0.02), self.mat_dalle)
        # les voies de sortie s'abouchent au tapis principal : leurs longerons
        # le traverseraient, on les leur retire
        for cote, signe in (("g", 1.0), ("d", -1.0)) if rails else ():
            boite(stage, chemin + f"/rail_{cote}", (longueur, 0.05, 0.10),
                  (0, signe * (largeur / 2 + 0.05), 0.01), self.mat_rail)

        self._x0 = -longueur / 2 + 0.05
        self.n_lattes = int((longueur - 0.1) / self.PAS) + 1
        self._ops = []
        for i in range(self.n_lattes):
            latte = UsdGeom.Cube.Define(stage, chemin + f"/latte_{i}")
            latte.CreateSizeAttr(1.0)
            self._ops.append(UsdGeom.Xformable(latte).AddTransformOp())
            UsdShade.MaterialBindingAPI(latte).Bind(self.mat_latte)
        self.defiler(0.0)

    def defiler(self, avance: float) -> None:
        """Fait avancer les lattes de `avance` metres, en boucle."""
        etendue = self.n_lattes * self.PAS
        for i, op in enumerate(self._ops):
            x = self._x0 + (i * self.PAS + avance) % etendue
            op.Set(Gf.Matrix4d().SetScale(Gf.Vec3d(0.02, self.largeur, 0.012)) *
                   Gf.Matrix4d().SetTranslate(Gf.Vec3d(x, 0, 0.002)))

    def placer(self, translation, rotation_z=0.0) -> None:
        """Pose le tapis : translation du centre, puis cap en degres autour de Z."""
        self.op.Set(Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), rotation_z)) *
                    Gf.Matrix4d().SetTranslate(Gf.Vec3d(*translation)))
