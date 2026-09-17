"""Entraine le bras a poser l'assiette sur la bonne voie. PPO, dans le conteneur.

Le modele n'a rien a voir avec celui des fromages : il ne voit pas d'image, ne
connait aucune classe, et recoit juste un numero de voie. Son metier est
mecanique — saisir une assiette qui defile, la porter, la poser a plat sans
renverser le morceau qu'elle transporte.

    docker run --rm --gpus '"device=0"' -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \\
      -v /home/nvidia/hpe/cheese:/workspace -v ~/.cache/ov/hub:/var/cache/hub \\
      -e PYTHONPATH=/isaac-sim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle \\
      --entrypoint /isaac-sim/python.sh ${ISAAC_SIM_IMAGE:-nvcr.io/nvidia/isaac-sim:6.1.0} \\
      /workspace/sim/train_pick.py --envs 1024 --iters 1500

`torch` n'est pas sur le chemin par defaut du conteneur mais il est dans
l'image : c'est ce que fait la ligne `PYTHONPATH` (extension `ml_archive`).

Sorties, comme pour les tetes de perception : `runs/<nom>/best.pt`,
`runs/<nom>/results.json` (l'historique complet, une entree par iteration).
Le dossier doit exister et etre accessible en ecriture AVANT le lancement, le
conteneur ne tournant pas en root.
"""

import argparse, json, math, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--envs", type=int, default=1024)
ap.add_argument("--iters", type=int, default=1500)
ap.add_argument("--rollout", type=int, default=24, help="pas de controle par iteration")
ap.add_argument("--epoques", type=int, default=5)
ap.add_argument("--lots", type=int, default=4, help="mini-lots par epoque")
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--gamma", type=float, default=0.99)
ap.add_argument("--lam", type=float, default=0.95)
ap.add_argument("--clip", type=float, default=0.2)
ap.add_argument("--entropie", type=float, default=0.004)
ap.add_argument("--kl-cible", type=float, default=0.012, help="0 = pas de pas adaptatif")
ap.add_argument("--duree-max", type=int, default=300, help="pas de controle par episode")
ap.add_argument("--part-prise", type=float, default=0.4,
                help="part des episodes qui demarrent assiette deja pincee")
ap.add_argument("--fin-part-prise", type=int, default=600,
                help="iteration ou cette part tombe a zero")
ap.add_argument("--vitesse-depart", type=float, default=0.0)
ap.add_argument("--vitesse-max", type=float, default=0.28)
ap.add_argument("--seuil-palier", type=float, default=0.55,
                help="taux de reussite au-dela duquel le tapis accelere")
ap.add_argument("--out", default="/workspace/runs/pick_fr3")
ap.add_argument("--reprise", default="", help="checkpoint a reprendre")
ap.add_argument("--graine", type=int, default=0)
ap.add_argument("--sous-pas", type=int, default=2)
ap.add_argument("--dt", type=float, default=1 / 60)
args = ap.parse_args()

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import sys

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pick_cell import Cellule

torch.manual_seed(args.graine)
DEV = "cuda:0"


# --------------------------------------------------------------------------
# politique
# --------------------------------------------------------------------------

class Normaliseur:
    """Moyenne/variance courantes des observations : sans ca, PPO patine."""

    def __init__(self, taille, dev):
        self.moy = torch.zeros(taille, device=dev)
        self.var = torch.ones(taille, device=dev)
        self.n = 1e-4

    def ajouter(self, x):
        m, v, k = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = m - self.moy
        total = self.n + k
        self.moy += delta * k / total
        self.var = (self.var * self.n + v * k + delta.pow(2) * self.n * k / total) / total
        self.n = total

    def __call__(self, x):
        return ((x - self.moy) / (self.var.sqrt() + 1e-5)).clamp(-8, 8)

    def etat(self):
        return {"moy": self.moy, "var": self.var, "n": self.n}

    def charger(self, e):
        self.moy, self.var, self.n = e["moy"].to(DEV), e["var"].to(DEV), e["n"]


class Politique(nn.Module):
    """Acteur-critique separes, ELU, ecart-type appris independant de l'etat."""

    def __init__(self, n_obs, n_act):
        super().__init__()
        def tronc(sortie):
            return nn.Sequential(nn.Linear(n_obs, 512), nn.ELU(),
                                 nn.Linear(512, 256), nn.ELU(),
                                 nn.Linear(256, 128), nn.ELU(),
                                 nn.Linear(128, sortie))
        self.acteur = tronc(n_act)
        self.critique = tronc(1)
        self.log_ecart = nn.Parameter(torch.full((n_act,), -1.0))
        plancher = torch.full((n_act,), math.log(0.30))
        plancher[4] = math.log(0.60)                 # la pince
        self.register_buffer("plancher", plancher)
        # La pince part fermee. Sans ce biais, l'exploration l'ouvre toutes les
        # cinq iterations : aucune prise ne dure, la recompense de transport
        # n'est jamais touchee, et rien ne pousse jamais a serrer. Partir ferme
        # met la politique du bon cote — il ne lui reste qu'a apprendre OU
        # ouvrir, ce qui est une decision locale et facile.
        with torch.no_grad():
            self.acteur[-1].bias[4] = -0.8

    def loi(self, obs):
        # Plancher d'exploration, et un plancher plus haut sur la pince. Ouvrir
        # la pince est une decision de signe : la politique apprend vite a la
        # fermer et sa consigne descend a -1.2, si bien qu'avec un ecart-type
        # libre la probabilite de tenter une ouverture tombe a 1e-5 — l'action
        # qui termine la tache devient litteralement inexplorable. Le plancher
        # la garde a portee de tirage sans rien lui imposer.
        return torch.distributions.Normal(self.acteur(obs),
                                          self.log_ecart.clamp(min=self.plancher).exp())

    def agir(self, obs):
        loi = self.loi(obs)
        a = loi.sample()
        return a, loi.log_prob(a).sum(-1), self.critique(obs).squeeze(-1)

    def evaluer(self, obs, a):
        loi = self.loi(obs)
        return (loi.log_prob(a).sum(-1), loi.entropy().sum(-1),
                self.critique(obs).squeeze(-1))


# --------------------------------------------------------------------------
# mise en place
# --------------------------------------------------------------------------

cellule = Cellule(n_env=args.envs, device=DEV, dt_physique=args.dt,
                  sous_pas=args.sous_pas, graine=args.graine, auto_collisions=False)
cellule.duree_max = args.duree_max
cellule.gamma = args.gamma          # la recompense de progres l'utilise
N, T = cellule.n, args.rollout

pol = Politique(cellule.n_obs, cellule.n_act).to(DEV)
opti = torch.optim.Adam(pol.parameters(), lr=args.lr)
norm = Normaliseur(cellule.n_obs, DEV)
lr = args.lr
vitesse_max = args.vitesse_depart
depart_iter = 0
historique = []

if args.reprise:
    ck = torch.load(args.reprise, map_location=DEV, weights_only=False)
    pol.load_state_dict(ck["politique"]); norm.charger(ck["norm"])
    vitesse_max = ck.get("vitesse_max", vitesse_max)
    depart_iter = ck.get("iteration", 0)
    historique = ck.get("historique", [])
    print(f"reprise de {args.reprise} a l'iteration {depart_iter}", flush=True)

sortie = Path(args.out); sortie.mkdir(parents=True, exist_ok=True)
import pick_cell as _pc
print(f"{N} cellules, {cellule.n_obs} observations, {cellule.n_act} actions", flush=True)
print(f"temoin: masse={_pc.MASSE_ASSIETTE} collider={_pc.EP_COLLIDER} "
      f"part_prise={cellule.part_prise} fichier={_pc.__file__}", flush=True)


def tirer_vitesses(ids):
    """Chaque episode tire sa vitesse de tapis sous le palier courant."""
    if ids.numel():
        cellule.vitesse[ids] = torch.rand(ids.numel(), device=DEV) * vitesse_max


tous = torch.arange(N, device=DEV)
tirer_vitesses(tous)
cellule.reinitialiser(tous)
# Les episodes demarrent tous au meme instant : sans ca ils expirent aussi tous
# ensemble et les remises arrivent par vagues, ce qui hache le signal
# d'apprentissage. On vieillit chaque cellule d'un tirage different.
cellule.pas_ep = torch.randint(0, args.duree_max, (N,), device=DEV).float()
obs = cellule.observer()

# tampons
o_buf = torch.zeros(T, N, cellule.n_obs, device=DEV)
a_buf = torch.zeros(T, N, cellule.n_act, device=DEV)
lp_buf = torch.zeros(T, N, device=DEV)
v_buf = torch.zeros(T, N, device=DEV)
r_buf = torch.zeros(T, N, device=DEV)
f_buf = torch.zeros(T, N, device=DEV)
tr_buf = torch.zeros(T, N, device=DEV)

meilleur = -1.0
t_debut = time.perf_counter()

for it in range(depart_iter, args.iters):
    t0 = time.perf_counter()
    # la part d'episodes demarres assiette en main s'efface avec l'entrainement :
    # a la fin, tout episode part du tapis
    cellule.part_prise = args.part_prise * max(0.0, 1.0 - it / max(1, args.fin_part_prise))
    stats = {"posee": 0, "renversee": 0, "perdue": 0, "tronque": 0, "episodes": 0,
             "tenue": 0.0, "ferme": 0.0, "z_max": 0.0, "c_droite": 0,
             "c_morceau_bas": 0, "c_morceau_sorti": 0, "c_ass_bas": 0, "c_tenait": 0}

    for t in range(T):
        norm.ajouter(obs)
        o = norm(obs)
        with torch.no_grad():
            a, lp, v = pol.agir(o)
        cellule.agir(a)
        cellule.pas()
        r, fini, info = cellule.evaluer()

        o_buf[t], a_buf[t], lp_buf[t], v_buf[t] = o, a, lp, v
        r_buf[t], f_buf[t] = r, fini.float()
        tr_buf[t] = info["tronque"].float()
        stats["tenue"] += float(info["tenue"].float().mean())
        stats["ferme"] += float(1.0 - cellule.pince_ouverte.mean())
        stats["z_max"] = max(stats["z_max"], float(cellule.pose_assiette()[0][:, 2].max()))

        if it < 2:
            pa = cellule.pose_assiette()[0]
            print(f"   t={t:2d} tenue {int(cellule.tenue.sum()):4d}  hautes "
                  f"{int((pa[:, 2] > 0.3).sum()):4d}  finies {int(fini.sum()):3d}  "
                  f"ferme {float(1 - cellule.pince_ouverte.mean()):.2f}  "
                  f"part_prise {cellule.part_prise:.2f}", flush=True)
        ids = fini.nonzero(as_tuple=False).squeeze(-1)
        if ids.numel():
            for cle in ("posee", "renversee", "perdue", "tronque"):
                stats[cle] += int(info[cle][ids].sum())
            # d'ou viennent les fins ? sans ce detail on optimise a l'aveugle
            d_ = cellule.assiette_droite()[ids]
            pa_ = cellule.pose_assiette()[0][ids]
            pm_ = cellule.pose_morceau()[ids]
            dedans_ = (pm_[:, :2] - pa_[:, :2]).norm(dim=1) < 0.090
            stats["c_droite"] += int((d_ < 0.30).sum())
            stats["c_morceau_bas"] += int((pm_[:, 2] < -0.05).sum())
            stats["c_morceau_sorti"] += int((~dedans_ & (pa_[:, 2] > 0.06)).sum())
            stats["c_ass_bas"] += int((pa_[:, 2] < -0.20).sum())
            stats["c_tenait"] += int((cellule.tenue[ids] > 0.5).sum())
            stats["episodes"] += int(ids.numel())
            tirer_vitesses(ids)
            cellule.reinitialiser(ids)
        obs = cellule.observer()

    # -- avantages (GAE), avec amorcage sur les episodes tronques ----------
    with torch.no_grad():
        derniere_v = pol.critique(norm(obs)).squeeze(-1)
        avantages = torch.zeros_like(r_buf)
        acc = torch.zeros(N, device=DEV)
        for t in reversed(range(T)):
            v_suiv = derniere_v if t == T - 1 else v_buf[t + 1]
            # un episode arrete par le chrono n'est pas un echec : on amorce
            continue_ = 1.0 - f_buf[t]
            cible = r_buf[t] + args.gamma * (v_suiv * continue_ + v_buf[t] * tr_buf[t])
            delta = cible - v_buf[t]
            acc = delta + args.gamma * args.lam * continue_ * acc
            avantages[t] = acc
        retours = avantages + v_buf
        avantages = (avantages - avantages.mean()) / (avantages.std() + 1e-8)

    # -- PPO ---------------------------------------------------------------
    plat = lambda x: x.reshape(T * N, -1).squeeze(-1)
    O, A, LP, AV, RE = (o_buf.reshape(T * N, -1), a_buf.reshape(T * N, -1),
                        plat(lp_buf), plat(avantages), plat(retours))
    taille = T * N // args.lots
    kl_moy = 0.0
    for _ in range(args.epoques):
        perm = torch.randperm(T * N, device=DEV)
        for i in range(args.lots):
            idx = perm[i * taille:(i + 1) * taille]
            lp2, ent, v2 = pol.evaluer(O[idx], A[idx])
            ratio = (lp2 - LP[idx]).exp()
            p1 = ratio * AV[idx]
            p2 = ratio.clamp(1 - args.clip, 1 + args.clip) * AV[idx]
            perte = (-torch.min(p1, p2).mean()
                     + 1.0 * (v2 - RE[idx]).pow(2).mean()
                     - args.entropie * ent.mean())
            opti.zero_grad(set_to_none=True)
            perte.backward()
            nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
            opti.step()
            with torch.no_grad():
                kl_moy = float(((LP[idx] - lp2).exp() - 1 - (LP[idx] - lp2)).mean())

    if args.kl_cible > 0:                      # pas adaptatif, facon Isaac Lab
        if kl_moy > 2.0 * args.kl_cible:
            lr = max(1e-5, lr / 1.5)
        elif kl_moy < 0.5 * args.kl_cible:
            # Plafond a 2e-3 : le pas adaptatif montait jusqu'a 8.5e-3, ou la
            # politique se met a osciller au lieu d'avancer.
            lr = min(2e-3, lr * 1.5)
        for g in opti.param_groups:
            g["lr"] = lr

    # -- journal et palier -------------------------------------------------
    eps = max(1, stats["episodes"])
    reussite = stats["posee"] / eps
    ligne = {"iteration": it, "recompense": round(float(r_buf.mean()), 3),
             "episodes": stats["episodes"], "reussite": round(reussite, 4),
             "renversees": round(stats["renversee"] / eps, 4),
             "perdues": round(stats["perdue"] / eps, 4),
             "part_tenue": round(stats["tenue"] / T, 4),
             "part_fermee": round(stats["ferme"] / T, 4),
             "z_max": round(stats["z_max"], 3),
             "part_prise": round(cellule.part_prise, 3),
             "vitesse_max": round(vitesse_max, 3), "lr": lr, "kl": round(kl_moy, 5),
             "secondes": round(time.perf_counter() - t0, 2)}
    historique.append(ligne)

    if reussite > args.seuil_palier and vitesse_max < args.vitesse_max:
        vitesse_max = min(args.vitesse_max, vitesse_max + 0.02)

    if it % 5 == 0 or it == args.iters - 1:
        ecoule = (time.perf_counter() - t_debut) / 60
        print(f"[{it:5d}] r {ligne['recompense']:+7.2f}  pose {reussite:5.1%}  "
              f"tenue {ligne['part_tenue']:5.1%}  ferme {ligne['part_fermee']:5.1%}  "
              f"z {ligne['z_max']:.2f}  renv {ligne['renversees']:5.1%}  "
              f"perdu {ligne['perdues']:5.1%}  v {vitesse_max:.2f}  "
              f"main {ligne['part_prise']:.2f}  lr {lr:.1e}  {ligne['secondes']:4.1f}s  "
              f"({ecoule:.0f} min)", flush=True)
        print(f"        causes: droite {stats['c_droite']:4d}  morceau_bas "
              f"{stats['c_morceau_bas']:4d}  morceau_sorti {stats['c_morceau_sorti']:4d}  "
              f"assiette_bas {stats['c_ass_bas']:4d}  tenait {stats['c_tenait']:4d}  "
              f"episodes {stats['episodes']:4d}", flush=True)

    if reussite > meilleur and stats["episodes"] > 50:
        meilleur = reussite
        torch.save({"politique": pol.state_dict(), "norm": norm.etat(),
                    "n_obs": cellule.n_obs, "n_act": cellule.n_act,
                    "iteration": it, "reussite": reussite, "vitesse_max": vitesse_max,
                    "historique": historique, "args": vars(args)},
                   sortie / "best.pt")
    if it % 25 == 0 or it == args.iters - 1:
        torch.save({"politique": pol.state_dict(), "norm": norm.etat(),
                    "n_obs": cellule.n_obs, "n_act": cellule.n_act,
                    "iteration": it, "reussite": reussite, "vitesse_max": vitesse_max,
                    "historique": historique, "args": vars(args)},
                   sortie / "last.pt")
        (sortie / "results.json").write_text(json.dumps(
            {"meilleure_reussite": meilleur, "iterations": len(historique),
             "envs": N, "rollout": T, "pas_total": len(historique) * T * N,
             "historique": historique}, indent=1))

print(f"\nmeilleure reussite : {meilleur:.1%}  "
      f"({(time.perf_counter() - t_debut) / 60:.0f} min)", flush=True)
app.close()
