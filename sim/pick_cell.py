"""La cellule de tri : tapis d'amenee, bras FR3, six tapis de sortie.

C'est l'environnement d'entrainement du bras, et **il ne sait rien des
fromages**. Une assiette arrive sur le tapis d'amenee, un numero de voie lui est
donne, et tout son travail est : la saisir par le bord pendant qu'elle defile,
la porter au-dessus du tapis de sortie demande, et l'y poser a plat sans la
renverser. Le morceau qu'elle transporte est un corps rigide : s'il bascule
hors de l'assiette, l'episode est rate. La classe du fromage, c'est l'affaire
du modele de perception (`sim/sort_server.py`) ; ici elle n'est qu'un entier.

                     voie 0   voie 1   voie 2
                        \\       |       /
    ==== amenee ====>   [ FR3 ]            (les six tapis de sortie
                        /       |       \\   sont en arc autour du bras)
                     voie 3   voie 4   voie 5

Les N cellules sont clonees par `GridCloner` et avancent ensemble sur GPU : un
seul `world.step()` fait avancer les 64 bras.

Notes d'API, apprises a la dure
-------------------------------
* Les poses de corps vivantes viennent de `_physics_view.get_link_transforms()`.
  `isaacsim.core.utils.xforms.get_world_pose` lit l'USD, qui reste fige sous le
  pipeline GPU : les deux doigts y sortent au meme point, quoi qu'il arrive.
* Cette vue rend les quaternions en **xyzw**, alors que le reste
  d'`isaacsim.core` les rend en wxyz. Tout ce fichier travaille en xyzw.
* La jacobienne est `(N, corps - 1, 6, ddl)` : la ligne du TCP est son index de
  corps **moins un**, la base fixe n'y figurant pas.
"""

from __future__ import annotations

import math

import torch

import omni.usd
from isaacsim.core.api import World
from isaacsim.core.cloner import GridCloner
from isaacsim.core.prims import Articulation, RigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

from usd_kit import Tapis, assiette, boite, materiau_uni, quad

# --------------------------------------------------------------------------
# plan de la cellule, en metres, repere local d'une cellule
# --------------------------------------------------------------------------

# Le bras est perche sur un socle, au-dessus du plan des tapis : c'est la
# disposition d'une vraie cellule, et elle degage la base du tapis (sinon le
# longeron traverse le socle) tout en elargissant la fenetre de prise. Le zero
# des z reste la surface des tapis.
Z_BASE = 0.28           # hauteur de la base du bras au-dessus des tapis
# Hauteur de transport. Elle doit degager le SOCLE, dont le sommet est a Z_BASE :
# porter l'assiette a 0.29 la faisait accrocher le haut du socle en passant du
# tapis aux voies, et le bras s'y arc-boutait jusqu'a l'expiration du cycle.
Z_TRANSPORT = Z_BASE + 0.17
# Le tapis passe devant le bras, pas au bout de son bras. A 0.66 m d'axe, le
# point de prise etait a 0.74 m de la base et 14 cm sous elle : le poignet ne
# pouvait plus a la fois s'y placer et pointer vers le bas, l'IK arbitrait, et
# l'approche plafonnait a 3 cm d'erreur laterale sans jamais converger. A
# 0.45 m tout rentre dans la partie confortable de l'espace de travail. La
# largeur suit : un tapis de 0.55 m est a l'echelle d'une assiette de 64 mm,
# et le bord reste a 7.5 cm du socle.
Y_TAPIS = 0.45          # axe du tapis d'amenee
LARG_TAPIS = 0.55       # largeur du tapis
# Le tapis deborde largement du champ de la camera, des deux cotes. Ce n'est pas
# de la figuration : c'est ce qui permet a l'assiette de NAITRE hors cadre et
# d'entrer dans l'image en defilant, au lieu d'apparaitre par magie au milieu du
# plan.
LONG_TAPIS = 5.00
X_ENTREE = -2.10        # ou l'assiette apparait, hors du champ de la camera
X_PRISE = -0.05         # milieu de la fenetre atteignable
X_SORTIE = 0.50         # au-dela, elle est hors de portee : episode perdu
R_VOIE = 0.60           # rayon de l'arc des tapis de sortie
N_VOIES = 6
ANGLE_VOIE0 = -148.0    # degres, 0 = +x ; les voies sont du cote oppose au tapis
ANGLE_VOIE1 = -32.0
LARG_VOIE = 0.32
LONG_VOIE = 0.60

# L'assiette est dimensionnee pour que la pince l'ENJAMBE, et non pour qu'elle
# la pince par le bord. Ce n'est pas un detail d'esthetique, c'est la condition
# de faisabilite de la tache : un plateau de 180 mm saisi par son bord pend a
# 90 mm de son centre de masse, soit 0.12 N.m de couple sur deux mors
# paralleles. Il bascule, le morceau tombe, et aucun reglage de recompense ne
# rattrape ca — 29 M de pas d'entrainement l'ont paye pour le savoir. A 68 mm
# de diametre pour 80 mm d'ouverture, les mors serrent deux points
# diametralement opposes : la ligne de prise passe par le centre de masse,
# l'assiette pend a plat, et le lacet n'a plus aucune importance sur un disque.
RAYON_ASSIETTE = 0.032  # bord exterieur ; la pince se referme dessus des DEUX cotes
HAUT_ASSIETTE = 0.024   # paroi assez haute pour retenir le morceau
FOND_ASSIETTE = 0.012   # socle PLEIN : c'est la-dessus que les mors serrent
EP_ASSIETTE = 0.005
EP_COLLIDER = 0.008     # paroi vue par PhysX : plus epaisse que le visuel
MASSE_ASSIETTE = 0.05
MASSE_MORCEAU = 0.008

OUVERT = 0.040          # course d'un doigt, pince ouverte
# La consigne de fermeture bute sur l'assiette au lieu de viser zero. Commander
# zero, c'est demander 34 mm de course au-dela du contact : le drive pousse,
# le solveur cede, et les mors s'enfoncent progressivement jusqu'au centre de
# l'assiette — tenue correcte, mais image impossible. Avec 4 mm d'interference
# la prise est franche et les mors restent ou on les voit.
FERME = RAYON_ASSIETTE - 0.004

# Largeur de pince lue (somme des deux doigts) selon ce qu'elle fait. La bande
# "tenue" est sans ambiguite : fermee sur du vide, la pince tombe a zero ;
# fermee sur l'assiette, elle bute a son diametre. C'est un capteur de prise
# gratuit, et bien plus franc que l'ancien critere sur une paroi de 14 mm.
COTE_MORCEAU = 0.012    # cube, taille native : pas d'echelle sur un collider
# le morceau repose sur le fond de l'assiette : bas de l'assiette + fond + demi-cote
Z_MORCEAU = 0.002 + FOND_ASSIETTE + COTE_MORCEAU / 2 + 0.001

LARG_OUVERTE = 2 * OUVERT               # 0.080

# Orientation de l'embase. Elle n'est pas decorative : l'axe 1 du FR3 est
# limite a +-157 deg, donc il existe un secteur de 46 deg, exactement a
# l'oppose de la face du bras, ou aucune pose n'est atteignable. Monte face au
# tapis (+90 deg), ce secteur tombait au milieu de l'arc des voies : `bin_soft`
# et `bin_fresh` etaient hors d'atteinte, et l'automate y restait bloque
# jusqu'a l'expiration du cycle. On tourne donc l'embase pour loger l'angle
# mort dans l'intervalle libre entre la derniere voie et le tapis. Tout est
# alors a moins de 119 deg de la face du bras, avec 38 deg de marge.
ANGLE_BASE = -29.0
# pose de repos : le TCP regarde vers le bas, au-dessus du tapis d'amenee
REPOS = (math.radians(90.0 - ANGLE_BASE), -0.5, 0.0, -2.3, 0.0, 1.85, 0.79)


def voies_xy() -> list[tuple[float, float]]:
    """Centre de chaque tapis de sortie, dans le repere de la cellule."""
    pts = []
    for k in range(N_VOIES):
        f = k / (N_VOIES - 1)
        a = math.radians(ANGLE_VOIE0 + f * (ANGLE_VOIE1 - ANGLE_VOIE0))
        pts.append((R_VOIE * math.cos(a), R_VOIE * math.sin(a)))
    return pts


# --------------------------------------------------------------------------
# quaternions (xyzw), en torch et par lot
# --------------------------------------------------------------------------

def q_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack([aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw,
                        aw * bw - ax * bx - ay * by - az * bz], dim=-1)


def q_conj(q: torch.Tensor) -> torch.Tensor:
    return q * torch.tensor([-1.0, -1.0, -1.0, 1.0], device=q.device)


def q_rot(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Applique la rotation q au vecteur v."""
    u, w = q[..., :3], q[..., 3:]
    return (v + 2.0 * torch.linalg.cross(u, torch.linalg.cross(u, v) + w * v))


def q_erreur(q_cible: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Vecteur rotation menant de q a q_cible (petit angle : 2 * partie vectorielle)."""
    d = q_mul(q_cible, q_conj(q))
    d = torch.where(d[..., 3:4] < 0, -d, d)      # chemin court
    return 2.0 * d[..., :3]


def lacet_de(q: torch.Tensor) -> torch.Tensor:
    """Lacet d'un quaternion xyzw, dans la convention de `q_lacet`."""
    return torch.atan2(2 * (q[..., 3] * q[..., 2] + q[..., 0] * q[..., 1]),
                       1 - 2 * (q[..., 1] ** 2 + q[..., 2] ** 2))


def ecart_lacet(lacet: torch.Tensor, cible: torch.Tensor) -> torch.Tensor:
    """Ecart de lacet modulo pi : une pince a deux doigts est symetrique."""
    e = (lacet - cible + math.pi / 2) % math.pi - math.pi / 2
    return e


def q_lacet(lacet: torch.Tensor) -> torch.Tensor:
    """Pince vers le bas (retournee autour de x) plus un lacet autour de z."""
    demi = lacet * 0.5
    qz = torch.stack([torch.zeros_like(demi), torch.zeros_like(demi),
                      torch.sin(demi), torch.cos(demi)], dim=-1)
    qx = torch.tensor([1.0, 0.0, 0.0, 0.0], device=lacet.device).expand_as(qz)
    return q_mul(qz, qx)


# --------------------------------------------------------------------------
# la cellule
# --------------------------------------------------------------------------

class Cellule:
    """N cellules identiques, clonees et simulees ensemble sur GPU."""

    def __init__(self, n_env=64, device="cuda:0", dt_physique=1 / 120, sous_pas=4,
                 espacement=4.0, rendu=False, graine=0, ccd=True, auto_collisions=True):
        self.n = n_env
        self.dev = torch.device(device)
        self.sous_pas = sous_pas
        self.dt = dt_physique * sous_pas
        self.gen = torch.Generator(device=self.dev).manual_seed(graine)
        self.ccd = ccd
        self.auto_collisions = auto_collisions

        self.monde = World(stage_units_in_meters=1.0, backend="torch", device=device,
                           physics_dt=dt_physique, rendering_dt=dt_physique * sous_pas)
        ctx = self.monde.get_physics_context()
        ctx.enable_gpu_dynamics(True)
        ctx.set_solver_type("TGS")
        # Les tampons GPU de PhysX sont dimensionnes pour une scene, pas pour un
        # millier. Trop petits, PhysX n'echoue pas : il PERD des contacts, en le
        # disant dans un flot d'erreurs noye au milieu du journal. A 1024
        # cellules le morceau traversait le fond de son assiette et tombait —
        # `tenue` restait a 0.4 % et l'apprentissage ne demarrait jamais, alors
        # que la meme scene a 256 cellules montait a 16 % en cinq iterations.
        # On dimensionne donc les tampons sur le nombre de cellules.
        for nom, valeur in (
                ("gpu_total_aggregate_pairs_capacity", 64 * self.n + 8192),
                ("gpu_found_lost_pairs_capacity", 64 * self.n + 8192),
                ("gpu_found_lost_aggregate_pairs_capacity", 64 * self.n + 8192),
                ("gpu_max_rigid_contact_count", 256 * self.n + 65536),
                ("gpu_max_rigid_patch_count", 64 * self.n + 16384),
                ("gpu_collision_stack_size", 128 * 1024 * 1024),
                ("gpu_heap_capacity", 128 * 1024 * 1024)):
            regle = getattr(ctx, "set_" + nom, None)
            if regle is not None:
                regle(int(valeur))
        self.stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)

        self._decor()
        self._batir_cellule("/World/env_0")

        cloner = GridCloner(spacing=espacement)
        cibles = cloner.generate_paths("/World/env", self.n)
        cloner.clone(source_prim_path="/World/env_0", prim_paths=cibles,
                     replicate_physics=True, base_env_path="/World",
                     copy_from_source=False)

        self.bras = Articulation("/World/env_.*/fr3", name="fr3")
        self.plats = RigidPrim("/World/env_.*/assiette", name="assiette")
        self.morceaux = RigidPrim("/World/env_.*/morceau", name="morceau")
        for v in (self.bras, self.plats, self.morceaux):
            self.monde.scene.add(v)
        self.monde.reset()

        # Les origines de cellule se lisent sur les prims, dans l'ordre propre a
        # chaque vue. Les positions rendues par le cloner sont dans SON ordre a
        # lui, qui n'est pas l'ordre lexicographique de `env_.*` : s'en servir
        # remet chaque assiette dans la cellule d'un autre robot, ce qui ne se
        # voit pas en coordonnees locales et fait tomber l'assiette a cote du
        # tapis.
        self.origines = self._origines(self.bras)
        for vue in (self.plats, self.morceaux):
            assert torch.equal(self._origines(vue), self.origines), \
                "les vues ne parcourent pas les cellules dans le meme ordre"

        self.i_tcp = self.bras.body_names.index("fr3_hand_tcp")
        self.vue = self.bras._physics_view       # seule source de poses vivantes
        # Les gains du modele USD (520/46) laissent le bras flechir de 49 mm
        # sous son propre poids ; a 2000/120 il n'en reste que 13, et le
        # servo-visuel de `agir` absorbe le reste.
        self.bras.set_gains(
            kps=torch.tensor([[2000.0] * 7 + [1500.0] * 2], device=self.dev).repeat(self.n, 1),
            kds=torch.tensor([[120.0] * 7 + [80.0] * 2], device=self.dev).repeat(self.n, 1))
        self.limites = self.bras.get_dof_limits().to(self.dev)

        self.voies = torch.tensor(voies_xy(), device=self.dev, dtype=torch.float32)
        self.repos = torch.tensor(REPOS, device=self.dev).repeat(self.n, 1)

        # etat d'episode
        z = lambda: torch.zeros(self.n, device=self.dev)
        self.cible = torch.zeros(self.n, dtype=torch.long, device=self.dev)
        self.vitesse = z() + 0.15
        self.pas_ep, self.tenue, self.posee = z(), z(), z()
        # potentiel du pas precedent : la recompense en est la variation
        self.phi_prec = z()
        self.gamma = 0.99
        self.derniere_action = torch.zeros(self.n, 5, device=self.dev)
        self.pince_ouverte = torch.ones(self.n, device=self.dev)
        # Une cellule fraichement remise est gelee, puis REPOSEE un pas plus
        # tard : voir agir(). Le compteur vaut 2 a la remise, 1 apres le
        # premier pas (c'est la qu'on rejoue la pose), 0 ensuite.
        self.frais_cpt = torch.zeros(self.n, dtype=torch.long, device=self.dev)
        self.remise_ass = torch.zeros(self.n, 3, device=self.dev)
        self.remise_mor = torch.zeros(self.n, 3, device=self.dev)
        # Seuls les departs qui TELEPORTENT le bras ont besoin du second temps
        self.rejouer = torch.zeros(self.n, dtype=torch.bool, device=self.dev)
        # cellules a qui l'on glisse l'assiette en main au second temps
        self.a_pincer = torch.zeros(self.n, dtype=torch.bool, device=self.dev)
        self.duree_max = 420
        self.part_prise = 0.0        # part des episodes qui demarrent assiette en main

        # pose du TCP a la pose de repos : elle sert de consigne de depart a
        # chaque remise. Elle se lit apres un pas, les transformees de corps
        # etant en retard d'un pas sur l'ecriture des articulations.
        self.p_cmd = torch.zeros(self.n, 3, device=self.dev)
        self.lacet_cmd = torch.zeros(self.n, device=self.dev)
        self.reinitialiser(torch.arange(self.n, device=self.dev))
        # La pose de repos se mesure sur un bras ecrit EXACTEMENT a REPOS, et
        # apres l'avoir laisse s'y installer. Deux pieges, tous deux payes :
        # une lecture immediate rend la pose d'AVANT l'ecriture (les
        # transformees de corps ont un pas de retard), et une lecture prise
        # apres une remise ordinaire mesure une cellule BRUITEE — 0.02 rad sur
        # sept axes, soit 2 a 4 cm de TCP. Comme c'est cette pose qui sert a
        # poser l'assiette des departs "deja en main", l'erreur se paie en
        # assiette lachee a cote des mors.
        nominal = torch.cat([self.repos,
                             torch.full((self.n, 2), OUVERT, device=self.dev)], dim=1)
        self.bras.set_joint_positions(nominal)
        self.bras.set_joint_velocities(torch.zeros(self.n, 9, device=self.dev))
        self.bras.set_joint_position_targets(nominal)
        for _ in range(40):
            self.monde.step(render=False)
        p_repos, q_repos = self.poses_tcp()
        self.p_repos = p_repos[0].clone()
        self.lacet_repos = float(lacet_de(q_repos[0]))
        self.reinitialiser(torch.arange(self.n, device=self.dev))

    def _origines(self, vue) -> torch.Tensor:
        """Origine monde de chaque cellule, dans l'ordre de parcours de `vue`."""
        xs = []
        for chemin in vue.prim_paths:
            env = "/".join(chemin.split("/")[:3])          # /World/env_k
            t = UsdGeom.Xformable(self.stage.GetPrimAtPath(env)) \
                .ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
            xs.append([t[0], t[1], t[2]])
        return torch.tensor(xs, device=self.dev, dtype=torch.float32)

    # ---------------------------------------------------------------- decor
    def _decor(self):
        from pxr import UsdLux
        dome = UsdLux.DomeLight.Define(self.stage, "/World/dome")
        dome.CreateIntensityAttr(400.0)
        cle = UsdLux.SphereLight.Define(self.stage, "/World/cle")
        cle.CreateRadiusAttr(0.4); cle.CreateIntensityAttr(28000.0)
        UsdGeom.Xformable(cle).AddTranslateOp().Set(Gf.Vec3d(0.6, 0.0, 2.4))

        self.mat_phys = UsdShade.Material.Define(self.stage, "/World/mat_phys")
        m = UsdPhysics.MaterialAPI.Apply(self.mat_phys.GetPrim())
        m.CreateStaticFrictionAttr(1.3)
        m.CreateDynamicFrictionAttr(1.1)
        m.CreateRestitutionAttr(0.0)

    def _statique(self, prim):
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            self.mat_phys, UsdShade.Tokens.weakerThanDescendants, "physics")

    # ------------------------------------------------------------- batiment
    def _batir_cellule(self, base: str):
        st = self.stage
        UsdGeom.Xform.Define(st, base)

        mat_sol = materiau_uni(st, base + "/mat_sol", (0.18, 0.19, 0.22), rugosite=0.9)
        mat_tapis = materiau_uni(st, base + "/mat_tapis", (0.10, 0.10, 0.12), rugosite=0.85)
        mat_pince = materiau_uni(st, base + "/mat_pince", (0.86, 0.87, 0.85), rugosite=0.3)

        # 3.2 m : au-dela, les sols des cellules voisines se recouvrent
        sol = boite(st, base + "/sol", (3.2, 2.8, 0.10), (0, 0, -0.97), mat_sol)
        self._statique(sol.GetPrim())

        # tapis d'amenee : une dalle statique. Le defilement est joue en
        # imposant la vitesse de l'assiette, PhysX n'ayant pas de tapis roulant.
        amenee = boite(st, base + "/amenee", (LONG_TAPIS, LARG_TAPIS, 0.04),
                       (0.0, Y_TAPIS, -0.02), mat_tapis)
        self._statique(amenee.GetPrim())

        # les six tapis de sortie, en arc autour du bras
        for k, (x, y) in enumerate(voies_xy()):
            a = math.degrees(math.atan2(y, x))
            t = boite(st, base + f"/voie_{k}", (LONG_VOIE, LARG_VOIE, 0.04),
                      (0, 0, 0), materiau_uni(st, base + f"/mat_voie_{k}",
                                              (0.14, 0.15, 0.18), rugosite=0.8))
            UsdGeom.Xformable(t).GetOrderedXformOps()[0].Set(
                Gf.Matrix4d().SetScale(Gf.Vec3d(LONG_VOIE, LARG_VOIE, 0.04)) *
                Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), a)) *
                Gf.Matrix4d().SetTranslate(Gf.Vec3d(x, y, -0.02)))
            self._statique(t.GetPrim())

        # socle du bras : sa base affleure la surface des tapis
        socle = boite(st, base + "/socle", (0.20, 0.20, 0.95 + Z_BASE),
                      (0, 0, (Z_BASE - 0.95) / 2),
                      materiau_uni(st, base + "/mat_socle", (0.30, 0.31, 0.34),
                                   rugosite=0.4, metal=0.6))
        self._statique(socle.GetPrim())

        # le bras, oriente pour que voies et tapis tiennent tous dans sa course
        add_reference_to_stage(
            f"{get_assets_root_path()}/Isaac/Robots/FrankaRobotics/FrankaFR3/fr3.usd",
            base + "/fr3")
        ops = {o.GetOpName(): o for o in
               UsdGeom.Xformable(st.GetPrimAtPath(base + "/fr3")).GetOrderedXformOps()}
        ops["xformOp:translate"].Set(Gf.Vec3d(0, 0, Z_BASE))
        demi = math.radians(ANGLE_BASE) / 2
        ops["xformOp:orient"].Set(Gf.Quatd(math.cos(demi), 0, 0, math.sin(demi)))
        PhysxSchema.PhysxArticulationAPI.Apply(st.GetPrimAtPath(base + "/fr3")) \
            .CreateEnabledSelfCollisionsAttr(self.auto_collisions)

        # Les doigts du modele mettent 1.7 s a se fermer : entre l'action et sa
        # consequence, la politique ne peut plus faire le lien. On leve leur
        # vitesse maximale (0.4 m/s), ce qui reste plausible pour une pince
        # industrielle. Cela se regle sur le joint, avant le demarrage de la
        # physique : a chaud, le backend GPU refuse l'ecriture.
        # Compensation de gravite, comme sur un vrai FR3 : le controleur d'une
        # cellule industrielle annule le poids du bras, l'operateur commande une
        # pose et le bras y va. Sans elle, le bras s'installe plusieurs
        # centimetres SOUS sa consigne — 14 cm mesures en articulaire, 18 mm
        # residuels meme avec la consigne cartesienne absolue. Ce n'est pas un
        # detail de confort : la pose de repos devient impossible a connaitre,
        # donc l'assiette des departs "deja en main" est posee a cote des mors
        # et tombe entre eux. L'assiette et le morceau, eux, gardent leur poids.
        for prim in Usd.PrimRange(st.GetPrimAtPath(base + "/fr3")):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(True)

        for prim in Usd.PrimRange(st.GetPrimAtPath(base + "/fr3")):
            if "finger_joint" in prim.GetName():
                PhysxSchema.PhysxJointAPI.Apply(prim).CreateMaxJointVelocityAttr(0.25)
                # La consigne de fermeture est zero : sur une assiette de 68 mm
                # cela fait 34 mm d'erreur par doigt, et le drive d'origine y
                # repond par des centaines de newtons. Une assiette de 50 g
                # part alors comme un noyau de cerise a la moindre asymetrie.
                # On plafonne l'effort comme le ferait une vraie pince : 12
                # N tiennent largement l'assiette, sans l'ejecter.
                for jeu in ("linear", "transX", "transY", "transZ"):
                    d = UsdPhysics.DriveAPI.Get(prim, jeu)
                    if d:
                        d.CreateMaxForceAttr(12.0)

        # -- l'assiette : un visuel lisse, des colliders convexes explicites --
        # La decomposition convexe d'une paroi de 8 mm coute cher et laisse
        # passer : les coques generees sont fines et l'assiette traversait le
        # tapis. On donne donc a PhysX des formes qu'il gere exactement — un
        # cylindre pour le fond, seize paves pour la couronne — et on garde le
        # maillage a 96 segments pour l'image seulement.
        racine_ass = UsdGeom.Xform.Define(st, base + "/assiette")
        UsdGeom.Xformable(racine_ass).AddTranslateOp().Set(
            Gf.Vec3d(X_ENTREE, Y_TAPIS, 0.001))
        vis = assiette(st, base + "/assiette/visuel", rayon=RAYON_ASSIETTE,
                       hauteur=HAUT_ASSIETTE, epaisseur=EP_ASSIETTE,
                       fond=FOND_ASSIETTE)
        UsdShade.MaterialBindingAPI(vis).Bind(mat_pince)

        # Le socle va d'un bord a l'autre et il est PLEIN. C'est ce qui rend la
        # prise possible : a 0.4 m/s les mors avancent de 13 mm par pas de
        # controle, donc ils traversaient purement et simplement une paroi de
        # 6 mm entre deux tests de collision — largeur lue 0.068 puis 0.041 au
        # pas suivant, sans jamais rien toucher. Sur un cylindre plein il n'y a
        # rien a traverser : le contact est garanti quelle que soit la vitesse.
        fond = UsdGeom.Cylinder.Define(st, base + "/assiette/col_fond")
        fond.CreateRadiusAttr(RAYON_ASSIETTE); fond.CreateHeightAttr(FOND_ASSIETTE)
        fond.CreateAxisAttr("Z")
        fond.CreateExtentAttr([Gf.Vec3f(-RAYON_ASSIETTE, -RAYON_ASSIETTE, -FOND_ASSIETTE / 2),
                               Gf.Vec3f(RAYON_ASSIETTE, RAYON_ASSIETTE, FOND_ASSIETTE / 2)])
        UsdGeom.Xformable(fond).AddTranslateOp().Set(Gf.Vec3d(0, 0, FOND_ASSIETTE / 2))
        self._collider(fond.GetPrim())

        # La paroi de collision est plus epaisse que la paroi vue : a 8 mm les
        # doigts s'y enfoncent de 3.5 mm, la prise devient molle et l'assiette
        # bascule en cinq pas. Le collider deborde de 3 mm de chaque cote, ce
        # qui ne se voit pas et donne une prise franche.
        # La couronne retient le morceau. Ses paves sont des cubes a taille
        # NATIVE, seulement tournes et places : une echelle non uniforme sur le
        # collider d'un corps rigide le fait deriver, et seize paves etires
        # faisaient glisser l'assiette hors des mors en une dizaine de pas.
        # C'est la meme lecon que pour la racine du morceau, un cran plus bas
        # dans l'arbre.
        n_mur = 16
        cote = HAUT_ASSIETTE - FOND_ASSIETTE
        r_mur = RAYON_ASSIETTE - cote / 2
        for k in range(n_mur):
            a_mur = 360.0 * k / n_mur
            mur = UsdGeom.Cube.Define(st, base + f"/assiette/col_mur_{k:02d}")
            mur.CreateSizeAttr(cote)
            d = cote / 2
            mur.CreateExtentAttr([Gf.Vec3f(-d, -d, -d), Gf.Vec3f(d, d, d)])
            UsdGeom.Xformable(mur).AddTransformOp().Set(
                Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), a_mur)) *
                Gf.Matrix4d().SetTranslate(Gf.Vec3d(
                    r_mur * math.cos(math.radians(a_mur)),
                    r_mur * math.sin(math.radians(a_mur)),
                    (HAUT_ASSIETTE + FOND_ASSIETTE) / 2)))
            self._collider(mur.GetPrim())

        self._corps_rigide(racine_ass.GetPrim(), MASSE_ASSIETTE)

        # Le morceau transporte : c'est lui qui dit si l'assiette a ete renversee.
        # Sa racine ne porte QUE la translation — un corps rigide dont la
        # transformee contient une echelle est ejecte au premier pas (mesure :
        # +0.85 m en 40 pas, puis il quitte la scene). L'echelle doit vivre sur
        # un fils, et `boite()` la met toujours dans la transformee.
        racine_mor = UsdGeom.Xform.Define(st, base + "/morceau")
        UsdGeom.Xformable(racine_mor).AddTranslateOp().Set(
            Gf.Vec3d(X_ENTREE, Y_TAPIS, Z_MORCEAU))
        # Cube a taille NATIVE, sans op d'echelle : le commentaire ci-dessus vaut
        # aussi pour les fils. Une echelle non uniforme sur le collider d'un
        # corps rigide le fait deriver — mesure : le morceau accelerait vers le
        # haut a 8.7 g, quittait l'assiette en cinq pas et l'episode etait
        # declare renverse. Un cube de 18 mm tient dans l'assiette et se lit
        # comme un morceau de fromage.
        m = UsdGeom.Cube.Define(st, base + "/morceau/col")
        m.CreateSizeAttr(COTE_MORCEAU)
        d = COTE_MORCEAU / 2
        m.CreateExtentAttr([Gf.Vec3f(-d, -d, -d), Gf.Vec3f(d, d, d)])
        UsdShade.MaterialBindingAPI(m).Bind(
            materiau_uni(st, base + "/mat_morceau", (0.93, 0.84, 0.45), rugosite=0.6))
        self._collider(m.GetPrim())
        self._corps_rigide(racine_mor.GetPrim(), MASSE_MORCEAU)

    def _collider(self, prim):
        """Forme de collision : invisible si elle double un visuel."""
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(0.005)
        px.CreateRestOffsetAttr(0.0005)
        if prim.GetName().startswith("col_"):        # double un visuel lisse
            UsdGeom.Imageable(prim).MakeInvisible()
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            self.mat_phys, UsdShade.Tokens.weakerThanDescendants, "physics")

    def _corps_rigide(self, prim, masse):
        """Corps rigide dont les colliders sont deja poses (sur lui ou ses fils).

        CCD activee : l'assiette est mince et le tapis fait 4 cm, sans elle un
        pas rapide la fait traverser.
        """
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(masse)
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateEnableCCDAttr(self.ccd)
        rb.CreateSolverPositionIterationCountAttr(12)
        rb.CreateMaxDepenetrationVelocityAttr(1.0)

    # ----------------------------------------------------------------- etat
    def poses_tcp(self):
        """Position et quaternion (xyzw) du TCP, dans le repere de chaque cellule."""
        T = self.vue.get_link_transforms()[:, self.i_tcp]
        return T[:, :3] - self.origines, T[:, 3:7]

    def pose_assiette(self):
        p, q = self.plats.get_world_poses()          # isaacsim rend wxyz
        q = q[:, [1, 2, 3, 0]]                       # -> xyzw
        return p - self.origines, q

    def pose_morceau(self):
        p, _ = self.morceaux.get_world_poses()
        return p - self.origines

    def largeur_pince(self):
        q = self.bras.get_joint_positions()
        return q[:, 7] + q[:, 8]

    def point_de_prise(self):
        """L'axe de l'assiette, a mi-hauteur de paroi, et le lacet qui va avec.

        La pince enjambe l'assiette entiere : le point vise est donc son
        **centre**, pas son bord. Les deux mors se referment sur deux points
        diametralement opposes, la ligne de prise passe par l'axe du centre de
        masse et l'assiette pend a plat — c'est ce qui rend la tache faisable.

        Le lacet rendu est le lacet **vise**, et il suit l'azimut de l'assiette.
        Sur un disque, aucune orientation de serrage n'est meilleure qu'une
        autre — mais toutes ne sont pas ATTEIGNABLES. Mesure : lacet fige, le
        poignet plafonne a 54 mm au-dessus du tapis et la descente n'aboutit
        jamais ; lacet radial, il descend a 10 mm. La contrainte n'est pas sur
        la prise, elle est sur la cinematique du bras.
        """
        c, _ = self.pose_assiette()
        d = c[:, :2] / c[:, :2].norm(dim=1, keepdim=True).clamp(min=1e-6)
        p = c.clone()
        # Le repere `fr3_hand_tcp` est au plan des pointes : les viser a
        # mi-socle met les mors a cheval sur l'assiette sans toucher le tapis.
        p[:, 2] = c[:, 2] + FOND_ASSIETTE / 2
        return p, torch.atan2(d[:, 1], d[:, 0]) + math.pi / 2

    def assiette_droite(self):
        """Cosinus entre l'axe vertical de l'assiette et la verticale du monde."""
        _, q = self.pose_assiette()
        haut = q_rot(q, torch.tensor([0.0, 0.0, 1.0], device=self.dev).expand(self.n, 3))
        return haut[:, 2]

    # ------------------------------------------------------------ dynamique
    def _tapis(self):
        """Fait defiler ce qui repose sur le tapis d'amenee.

        PhysX n'a pas de tapis roulant. Imposer une vitesse aux corps poses
        dessus ne suffit pas : le frottement du tapis (1.3) l'annule des le
        sous-pas suivant, et l'assiette n'avance pas d'un millimetre — mesure
        faite, 0.16 m/s commandes, 0.000 m parcourus. On impose donc l'AVANCE
        elle-meme, position et vitesse ensemble. C'est d'ailleurs ce que fait
        une bande qui entraine : elle deplace, elle ne pousse pas.

        On ne touche a rien quand le tapis est a l'arret : reecrire une pose
        lue — qui a un pas de retard — annulerait le geste du bras en train de
        saisir l'assiette.
        """
        da = self.vitesse * self.dt
        for vue in (self.plats, self.morceaux):
            p, q = vue.get_world_poses()
            loc = p - self.origines
            dessus = (loc[:, 2] < 0.075) & (loc[:, 1] > Y_TAPIS - LARG_TAPIS / 2) & \
                     (loc[:, 1] < Y_TAPIS + LARG_TAPIS / 2) & (self.tenue < 0.5) & \
                     (self.vitesse > 1e-4)
            if not bool(dessus.any()):
                continue
            p = p.clone()
            p[dessus, 0] += da[dessus]
            vue.set_world_poses(p, q)
            # Les vitesses lues ont un pas de retard : juste apres une remise,
            # les recopier ressusciterait l'ancienne. On remet donc a zero les
            # cellules fraichement remises au lieu de relire ce qu'elles disent.
            v = vue.get_velocities().clone()
            v[dessus, 0] = self.vitesse[dessus]
            v[dessus, 1] *= 0.5
            v[self.frais_cpt > 0] = 0.0
            vue.set_velocities(v)

    def ik(self, p_cible, q_cible, amortissement=0.06, pas_max=0.12, gain=0.7):
        """Un pas d'IK amortie : l'ecart de pose devient un increment articulaire."""
        p, q = self.poses_tcp()
        err = torch.cat([(p_cible - p).clamp(-0.05, 0.05), q_erreur(q_cible, q)], dim=1)
        J = self.bras.get_jacobians()[:, self.i_tcp - 1, :, :7]
        A = J @ J.transpose(1, 2) + amortissement ** 2 * torch.eye(6, device=self.dev)
        dq = J.transpose(1, 2) @ torch.linalg.solve(A, err.unsqueeze(-1))
        # `gain` < 1 : un pas d'IK entier fait osciller le TCP de +-2 cm
        return (gain * dq.squeeze(-1)).clamp(-pas_max, pas_max)

    def agir(self, actions: torch.Tensor):
        """actions = (dx, dy, dz, dlacet, pince) dans [-1, 1].

        Les trois premieres composantes deplacent une **consigne cartesienne
        absolue**, pas le TCP mesure. La difference n'est pas cosmetique : une
        consigne relative (`mesure + delta`) annule l'erreur a chaque pas, donc
        elle ne peut pas lutter contre l'affaissement du bras — le TCP coulait
        de 14 cm sous la consigne et s'y stabilisait. Avec une consigne absolue,
        l'erreur persiste tant que le bras n'est pas arrive, et l'IK la comble.
        """
        a = actions.clamp(-1.0, 1.0)
        self.derniere_action = a
        p, q = self.poses_tcp()

        self.p_cmd = self.p_cmd + a[:, :3] * 0.020
        # Le plancher de la consigne doit passer SOUS le point de prise : celui-ci
        # est a 8 mm du tapis (mi-socle d'une assiette posee a 2 mm), donc un
        # plancher a 10 mm rendait la descente litteralement impossible et
        # l'automate restait bloque en phase 1.
        self.p_cmd[:, 2] = self.p_cmd[:, 2].clamp(0.003, Z_BASE + 0.55)
        # la consigne ne s'eloigne jamais beaucoup du bras : sinon elle file
        # devant lui et l'IK tire a fond dans une direction inatteignable
        ecart = self.p_cmd - p
        trop = ecart.norm(dim=1, keepdim=True).clamp(min=1e-6)
        recale = p + ecart * (trop.clamp(max=0.10) / trop)
        frais = self.frais_cpt > 0
        self.p_cmd = torch.where(frais.unsqueeze(1), self.p_cmd, recale)
        self.lacet_cmd = self.lacet_cmd + a[:, 3] * 0.15

        # Les transformees de corps ont un pas de retard sur l'ecriture des
        # articulations : juste apres une remise, `poses_tcp()` rend encore la
        # pose d'AVANT. S'en servir recale la consigne sur une position qui
        # n'existe plus et le bras part d'un coup — ce qui arrachait l'assiette
        # des departs "deja en main". Une cellule fraiche ne bouge donc pas
        # pendant un pas, le temps que la physique rattrape.
        dq = self.ik(self.p_cmd, q_lacet(self.lacet_cmd))
        dq = torch.where(frais.unsqueeze(1), torch.zeros_like(dq), dq)
        # Deuxieme temps de la remise. Au pas qui suit une ecriture, PhysX voit
        # encore les corps du bras a leur pose d'AVANT : il genere des contacts
        # avec des doigts qui sont, dans la scene, ailleurs — souvent pile ou
        # le morceau vient de naitre, qui repart alors a 5.8 m/s. On repose
        # donc assiette et morceau une seconde fois, une fois les transformees
        # du bras rattrapees. Mesure : sans ce second temps, aucun episode
        # "assiette deja en main" ne survivait a sa deuxieme seconde.
        second = (self.frais_cpt == 1)
        rejoue = (second & self.rejouer).nonzero(as_tuple=False).squeeze(-1)
        if rejoue.numel():
            self._poser(rejoue, self.remise_ass[rejoue], self.remise_mor[rejoue])

        # Depart "assiette deja en main", second temps. Le premier temps a
        # repose le bras au repos ; un pas plus tard il y est vraiment et ses
        # transformees sont a jour. C'est LA qu'on peut glisser l'assiette
        # entre les mors. L'inverse — teleporter le bras autour d'une assiette
        # deja posee — fait balayer aux doigts tout le volume qui les separe de
        # leur nouvelle pose et catapulte le morceau a 7 m/s.
        pince = (second & self.a_pincer).nonzero(as_tuple=False).squeeze(-1)
        if pince.numel():
            k = pince.numel()
            p_ass = self.p_repos.unsqueeze(0).repeat(k, 1)
            p_ass[:, 2] -= FOND_ASSIETTE / 2
            p_mor = p_ass.clone()
            p_mor[:, 2] += FOND_ASSIETTE + COTE_MORCEAU / 2 + 0.001
            self._poser(pince, p_ass + self.origines[pince],
                        p_mor + self.origines[pince])
            q_bras = self.bras.get_joint_positions()[pince][:, :7]
            doigts = torch.full((k, 2), RAYON_ASSIETTE + 0.002, device=self.dev)
            self.bras.set_joint_positions(torch.cat([q_bras, doigts], dim=1),
                                          indices=pince)
            self.pince_ouverte[pince] = 0.0
            self.a_pincer[pince] = False
        self.frais_cpt = (self.frais_cpt - 1).clamp(min=0)
        q_bras = self.bras.get_joint_positions()[:, :7] + dq
        q_bras = torch.max(torch.min(q_bras, self.limites[:, :7, 1]), self.limites[:, :7, 0])
        # Pince a etat, avec hysteresis. Tirer l'ouverture au sort a chaque pas
        # la fait papilloner : la probabilite de rester fermee dix pas de suite
        # est nulle, donc aucune prise ne dure et la politique n'apprend jamais
        # qu'il faut serrer. Ici l'action est un ORDRE — ouvrir, fermer, ou ne
        # rien changer — comme sur une vraie pince.
        un, zero = torch.ones_like(self.pince_ouverte), torch.zeros_like(self.pince_ouverte)
        self.pince_ouverte = torch.where(a[:, 4] > 0.3, un,
                                         torch.where(a[:, 4] < -0.3, zero,
                                                     self.pince_ouverte))
        cible_doigt = torch.where(self.pince_ouverte > 0.5,
                                  torch.full_like(self.pince_ouverte, OUVERT),
                                  torch.full_like(self.pince_ouverte, FERME))
        doigts = cible_doigt.unsqueeze(1).expand(self.n, 2)
        self.bras.set_joint_position_targets(torch.cat([q_bras, doigts], dim=1))

    def pas(self):
        # Une avance par pas de controle : `_tapis` ecrit des poses lues, qui
        # ont un pas de retard, donc les empiler sous-pas par sous-pas ferait
        # begayer l'assiette.
        self._tapis()
        for _ in range(self.sous_pas):
            self.monde.step(render=False)
        self.pas_ep += 1

    # --------------------------------------------------------- observations
    def observer(self) -> torch.Tensor:
        q = self.bras.get_joint_positions()
        qd = self.bras.get_joint_velocities()
        p_tcp, q_tcp = self.poses_tcp()
        p_ass, q_ass = self.pose_assiette()
        v_ass = self.plats.get_velocities()[:, :3]
        p_prise, lacet_prise = self.point_de_prise()
        voie = self.voies[self.cible]
        chaud = torch.zeros(self.n, N_VOIES, device=self.dev)
        chaud.scatter_(1, self.cible.unsqueeze(1), 1.0)
        depose = torch.cat([voie, torch.full((self.n, 1), 0.10, device=self.dev)], dim=1)
        return torch.cat([
            q[:, :7] / 3.0, qd[:, :7] / 10.0, self.largeur_pince().unsqueeze(1) / OUVERT,
            self.pince_ouverte.unsqueeze(1),
            p_tcp, q_tcp,
            p_ass - p_tcp, q_ass, v_ass,
            p_prise - p_tcp,
            ecart_lacet(lacet_de(q_tcp), lacet_prise).unsqueeze(1),
            depose - p_tcp, chaud,
            self.tenue.unsqueeze(1), self.vitesse.unsqueeze(1),
        ], dim=1)

    @property
    def n_obs(self) -> int:
        return 7 + 7 + 1 + 1 + 3 + 4 + 3 + 4 + 3 + 3 + 1 + 3 + N_VOIES + 1 + 1

    n_act = 5

    # ---------------------------------------------------------- recompenses
    def evaluer(self):
        """Recompense de progres, puis les fins d'episode. Renvoie (r, fini, info).

        Deux phases, decrites par un seul potentiel. Assiette libre, il croit
        quand le TCP se rapproche du point de prise ; assiette en main, il fait
        un bond puis croit quand l'assiette se rapproche de sa voie, a plat. La
        recompense rendue est la VARIATION de ce potentiel, pas sa valeur.
        """
        p_tcp, q_tcp = self.poses_tcp()
        p_ass, _ = self.pose_assiette()
        p_mor = self.pose_morceau()
        p_prise, lacet_prise = self.point_de_prise()
        voie = self.voies[self.cible]
        droite = self.assiette_droite()
        v_ass = self.plats.get_velocities()[:, :3]

        proche = (p_ass - p_tcp).norm(dim=1) < 0.045
        serre = self.pince_ouverte < 0.5
        self.tenue = (serre & proche & (p_ass[:, 2] > 0.020)).float()

        d_prise = (p_prise - p_tcp).norm(dim=1)
        d_xy = (p_prise[:, :2] - p_tcp[:, :2]).norm(dim=1)
        d_voie = (p_ass[:, :2] - voie).norm(dim=1)
        e_lacet = ecart_lacet(lacet_de(q_tcp), lacet_prise).abs()

        # -- fins d'episode ------------------------------------------------
        dedans = (p_mor[:, :2] - p_ass[:, :2]).norm(dim=1) < RAYON_ASSIETTE
        pose_ok = ((d_voie < 0.11) & (p_ass[:, 2] < 0.02) & (droite > 0.93) &
                   (v_ass.norm(dim=1) < 0.08) & ~serre & dedans)
        renversee = (droite < 0.30) | (p_mor[:, 2] < -0.05) | \
                    (~dedans & (p_ass[:, 2] > 0.06))
        perdue = (p_ass[:, 2] < -0.20) | (p_ass[:, 0] > X_SORTIE + 0.35)
        # Les poses lues ont un pas de retard sur les ecritures de remise : a la
        # premiere evaluation d'un episode, le morceau est encore a sa position
        # d'avant. Sans ce delai de grace, tout depart "assiette en main" etait
        # declare renverse aussitot, quelle que soit la physique.
        jeune = self.pas_ep < 3
        renversee = renversee & ~jeune
        perdue = perdue & ~jeune
        trop_tard = self.pas_ep >= self.duree_max
        rate = renversee | perdue

        # -- recompense de PROGRES, pas d'etat -----------------------------
        # Un bareme dense qui paie l'etat courant se retourne contre la tache :
        # porter l'assiette rapportait 10 par pas, indefiniment, et la poser
        # rapportait 80 une fois en arretant l'episode. Flaner valait donc une
        # dizaine de fois plus que reussir, et la politique apprenait
        # exactement cela — 29 M de pas passes a se garer au-dessus du point de
        # prise sans jamais serrer. On note donc un POTENTIEL, et la
        # recompense est sa variation : avancer paie, s'arreter ne paie rien,
        # et le bonus terminal n'a plus de concurrent. Mettre le potentiel a
        # zero sur les fins reelles est ce qui garantit que la politique
        # optimale ne change pas (Ng, Harada & Russell, 1999).
        # Les gains se lisent en "combien vaut un metre parcouru". Trop faibles,
        # le signal se noie : a 6 par metre, rapprocher l'assiette de sa voie
        # rapportait 0.04 par pas, moins que le cout d'action, et la politique
        # ne bougeait pas. Le palier "pince serree sur l'assiette" fait le pont
        # entre les deux phases : sans lui, serrer ne rapporte rien tant que
        # l'assiette n'a pas decolle, et rien ne pousse a essayer.
        # Le potentiel dit "assiette dans les mors", et non "assiette en l'air" :
        # sinon, descendre l'assiette sous 2 cm pour la poser la fait sortir de
        # la phase de transport et coute 65 AVANT que la prime de depose
        # n'arrive. Poser devenait un gouffre a franchir a l'aveugle.
        tient = (d_prise < 0.025) & serre
        porte = serre & proche
        # La distance qui compte est celle a la POSE FINALE, en trois dimensions.
        # Mesurée a plat, elle laissait la politique porter l'assiette a 71 cm
        # de haut : a cette hauteur le bras est presque vertical et ne peut
        # plus s'etendre lateralement, donc elle calait a 38 cm de la voie sans
        # que rien ne lui dise de descendre. En 3D, descendre EST se rapprocher.
        # La cible suit le geste, pas la ligne droite : on porte a hauteur de
        # transport tant qu'on n'est pas au-dessus de la voie, on ne descend
        # qu'ensuite. Viser directement le point de pose en trois dimensions
        # creusait un minimum local imparable — descendre coute moins cher au
        # bras que contourner son propre socle, donc la politique posait
        # l'assiette a mi-hauteur au-dessus du tapis et n'en bougeait plus.
        z_cible = torch.where(d_voie < 0.15,
                              torch.full_like(d_voie, 0.008),
                              torch.full_like(d_voie, Z_TRANSPORT))
        d_pose = torch.sqrt(d_voie.pow(2) + (p_ass[:, 2] - z_cible).pow(2))
        phi = torch.where(
            porte,
            60.0 - 40.0 * d_pose + 5.0 * droite,                  # porter puis poser
            -40.0 * d_prise - 4.0 * e_lacet + 12.0 * tient.float(),   # aller chercher
        )
        phi_fin = torch.where(rate | pose_ok, torch.zeros_like(phi), phi)
        r = self.gamma * phi_fin - self.phi_prec
        r = torch.where(self.pas_ep <= 1, torch.zeros_like(r), r)
        self.phi_prec = phi

        # La prime de depose doit dominer le potentiel qu'elle fait tomber a
        # zero : a la voie, phi vaut environ 65, donc une prime de 100 ne
        # laissait que 35 — trop peu face a l'option "continuer a porter".
        r = r + 200.0 * pose_ok.float() - 20.0 * rate.float()
        r = r - 0.02 * self.derniere_action.pow(2).sum(dim=1)
        # ne pas raser le tapis : on balaierait l'assiette au lieu de la prendre
        r = r - 0.15 * ((p_tcp[:, 2] < 0.085) & (d_xy > 0.05)).float()

        fini = pose_ok | rate | trop_tard
        self.posee = pose_ok.float()
        return r, fini, {"posee": pose_ok, "renversee": renversee, "perdue": perdue,
                         "tronque": trop_tard, "tenue": self.tenue > 0.5}

    # -------------------------------------------------------------- remises
    def reinitialiser(self, ids: torch.Tensor):
        """Remet des cellules a zero, sans jamais teleporter un bras qui tient.

        Deux departs coexistent. Le depart ORDINAIRE replace le bras au repos
        et fait naitre l'assiette sur le tapis : c'est la tache complete.
        Le depart PINCE, lui, ne touche pas au bras — il referme seulement les
        mors sur une assiette posee sous la consigne cartesienne courante.

        Ce detail a coute cher. Teleporter le bras au repos pour lui mettre une
        assiette en main fait balayer aux doigts, en un pas, tout le volume qui
        les separe de leur nouvelle pose ; le morceau qui vient d'y naitre est
        catapulte a 7 m/s. Et la pose lue juste apres une ecriture etant celle
        d'AVANT, on ne peut meme pas poser l'assiette la ou le bras est
        vraiment. En ne bougeant pas le bras, les deux pieges disparaissent
        d'un coup : `p_cmd` est un etat qu'on possede, sans retard, et la
        compensation de gravite garantit que le TCP s'y trouve.
        """
        if ids.numel() == 0:
            return
        n = ids.numel()

        # Qui partira assiette en main ? Le tirage est fait ici, l'assiette
        # n'est glissee dans les mors qu'au pas suivant (voir agir).
        if self.part_prise > 0:
            deja = torch.rand(n, device=self.dev, generator=self.gen) < self.part_prise
        else:
            deja = torch.zeros(n, dtype=torch.bool, device=self.dev)

        # Bras au repos. Le bruit articulaire donne de la variete aux departs
        # ordinaires, mais pas a ceux qui recevront une assiette : ceux-la
        # doivent avoir leur TCP exactement a la pose de repos, puisque c'est
        # la qu'on posera l'assiette.
        q = self.repos[ids].clone()
        bruit = torch.randn(n, 7, device=self.dev, generator=self.gen) * 0.02
        q = q + bruit * (~deja).unsqueeze(1).float()
        plein = torch.cat([q, torch.full((n, 2), OUVERT, device=self.dev)], dim=1)
        self.bras.set_joint_positions(plein, indices=ids)
        self.bras.set_joint_velocities(torch.zeros(n, 9, device=self.dev), indices=ids)
        self.bras.set_joint_position_targets(plein, indices=ids)

        # a l'arret l'assiette doit naitre dans la fenetre atteignable ; en
        # marche, assez en amont pour que le bras la voie venir
        uu = lambda a, b: torch.rand(n, device=self.dev, generator=self.gen) * (b - a) + a
        amont = (self.vitesse[ids] * 3.0).clamp(0.0, X_PRISE - X_ENTREE)
        x = X_PRISE - amont + uu(-0.12, 0.12)
        y = Y_TAPIS + uu(-0.05, 0.05)
        pos = torch.stack([x, y, torch.full_like(x, 0.002)], dim=1) + self.origines[ids]
        mor = pos.clone()
        mor[:, 0] += uu(-0.004, 0.004)
        mor[:, 1] += uu(-0.004, 0.004)
        mor[:, 2] = Z_MORCEAU + self.origines[ids][:, 2]
        self._poser(ids, pos, mor)

        self.pince_ouverte[ids] = 1.0
        if hasattr(self, "p_repos"):
            self.p_cmd[ids] = self.p_repos
            self.lacet_cmd[ids] = self.lacet_repos

        self.cible[ids] = torch.randint(0, N_VOIES, (n,), device=self.dev,
                                        generator=self.gen)
        self.frais_cpt[ids] = 2
        self.rejouer[ids] = ~deja
        self.a_pincer[ids] = deja
        self.pas_ep[ids] = 0
        self.tenue[ids] = 0
        self.posee[ids] = 0
        self.phi_prec[ids] = 0
        self.derniere_action[ids] = 0

    def _poser(self, ids, p_ass, p_mor):
        """Ecrit assiette et morceau, a plat, immobiles."""
        k = ids.numel()
        q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.dev).repeat(k, 1)
        zero6 = torch.zeros(k, 6, device=self.dev)
        self.plats.set_world_poses(p_ass, q, indices=ids)
        self.plats.set_velocities(zero6, indices=ids)
        self.morceaux.set_world_poses(p_mor, q, indices=ids)
        self.morceaux.set_velocities(zero6, indices=ids)
        self.remise_ass[ids] = p_ass
        self.remise_mor[ids] = p_mor


# --------------------------------------------------------------------------
# l'automate de reference
# --------------------------------------------------------------------------

class Automate:
    """Prise scriptee, dans le meme espace d'action que la politique.

    C'est le temoin : il prouve que la tache est faisable et sert de repli si
    la politique n'est pas au point. Il ne regarde qu'une chose a la fois
    (approcher, descendre, serrer, lever, porter, poser) avec des tolerances
    fixes, donc il cale des que la scene s'ecarte du cas nominal — la ou une
    politique apprise rattrape.
    """

    NOMS = ["approche", "alignement", "descente", "serrage", "levee",
            "transport", "depose", "degagement"]

    def __init__(self, cellule: "Cellule"):
        self.c = cellule
        self.etat = torch.zeros(cellule.n, dtype=torch.long, device=cellule.dev)
        self.horloge = torch.zeros(cellule.n, device=cellule.dev)

    def reinitialiser(self, ids):
        self.etat[ids] = 0
        self.horloge[ids] = 0

    def agir(self) -> torch.Tensor:
        c = self.c
        dev, N = c.dev, c.n
        p_tcp, q_tcp = c.poses_tcp()
        p_ass, _ = c.pose_assiette()
        p_prise, lacet_vise = c.point_de_prise()
        voie = c.voies[c.cible]
        lacet = lacet_de(q_tcp)

        cible = p_prise.clone()
        pince = torch.ones(N, device=dev)
        d_xy = (p_tcp[:, :2] - p_prise[:, :2]).norm(dim=1)
        lac_ok = ecart_lacet(lacet, lacet_vise).abs() < 0.12

        # La descente se fait en trois temps, et non d'un seul elan. La pince
        # ouverte fait 80 mm pour une assiette de 64 : 8 mm de jeu par cote.
        # Un seuil unique oblige a choisir entre descendre de travers (on
        # heurte le bord et on chasse l'assiette) et ne jamais descendre (le
        # bras tourne au-dessus jusqu'a l'expiration). En etageant, chaque
        # palier se resserre a hauteur ou l'erreur ne coute encore rien.
        m = self.etat == 0                                   # approche, 13 cm
        cible[m] = p_prise[m] + torch.tensor([0.0, 0.0, 0.13], device=dev)
        self.etat[m & (d_xy < 0.015) & lac_ok] = 1

        m = self.etat == 1                                   # alignement, 5 cm
        cible[m] = p_prise[m] + torch.tensor([0.0, 0.0, 0.050], device=dev)
        self.etat[m & (d_xy < 0.005) & lac_ok] = 2
        self.etat[m & (d_xy > 0.030)] = 0                    # on a derive

        m = self.etat == 2                                   # descente a mi-socle
        cible[m] = p_prise[m]
        entre = m & (d_xy < 0.010) & ((p_tcp[:, 2] - p_prise[:, 2]).abs() < 0.005)
        self.etat[entre] = 3
        self.horloge[entre] = 0              # seulement ceux qui ARRIVENT

        m = self.etat == 3                                   # serrage
        cible[m] = p_prise[m]
        pince[m] = -1.0
        self.horloge[m] += 1
        self.etat[m & (self.horloge > 30)] = 4

        m = self.etat == 4                                   # levee
        pince[m] = -1.0
        cible[m] = p_prise[m] + torch.tensor([0.0, 0.0, Z_TRANSPORT], device=dev)
        self.etat[m & (p_ass[:, 2] > Z_TRANSPORT - 0.06)] = 5

        # Transport direct vers la voie, a hauteur constante. Une trajectoire en
        # arc autour de l'axe du bras, qui evite pourtant le repli sur le socle,
        # s'est revelee pire dans la cellule de rendu : le bras y partait a
        # l'oppose de la consigne. Le trajet direct reste ce qui passe le plus
        # souvent, et il est de toute facon le temoin — c'est la politique
        # apprise qui doit, elle, trouver son chemin.
        m = self.etat == 5                                   # transport
        pince[m] = -1.0
        cible[m, 0] = voie[m, 0]
        cible[m, 1] = voie[m, 1]
        cible[m, 2] = Z_TRANSPORT
        self.etat[m & ((p_ass[:, :2] - voie).norm(dim=1) < 0.030)] = 6

        m = self.etat == 6                                   # depose
        pince[m] = -1.0
        cible[m, 0] = voie[m, 0]
        cible[m, 1] = voie[m, 1]
        cible[m, 2] = 0.002 + FOND_ASSIETTE / 2 + 0.004
        posee = p_ass[:, 2] < 0.012
        pince[m & posee] = 1.0
        self.etat[m & posee] = 7

        m = self.etat == 7                                   # degagement
        cible[m] = p_tcp[m] + torch.tensor([0.0, 0.0, 0.05], device=dev)

        delta = ((cible - p_tcp) / 0.020).clamp(-1, 1)
        # Une fois l'assiette en main, on leve et on transporte au tiers de la
        # vitesse. A pleine consigne le TCP file a 0.6 m/s : l'assiette bascule
        # dans les mors et le morceau passe par-dessus la paroi. Rien n'oblige
        # a aller vite, et la politique apprise, elle, paie deja ses a-coups
        # par le terme d'action de la recompense.
        # Doux quand on leve et quand on pose — c'est la que l'assiette bascule —
        # mais pas pendant le transport, qui ne fait que tourner a plat et qui,
        # au tiers de la vitesse, ne finissait pas dans le temps d'un cycle.
        lent = torch.where(self.etat == 5, 0.70,
                           torch.where(self.etat >= 4, 0.35, 1.0))
        delta = delta * lent.unsqueeze(1)
        dlac = (ecart_lacet(lacet_vise, lacet) / 0.15).clamp(-1, 1)
        return torch.stack([delta[:, 0], delta[:, 1], delta[:, 2], dlac, pince], dim=1)
