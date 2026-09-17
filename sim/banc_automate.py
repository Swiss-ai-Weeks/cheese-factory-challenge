"""Banc de l'automate de reference : combien d'assiettes posees, et pourquoi pas.

C'est le temoin de faisabilite de la cellule. Si l'automate n'y arrive pas, ce
n'est pas la peine d'accuser l'apprentissage : c'est la scene qui est en cause.
Le detail des causes de fin (basculement, morceau sorti, phase ou l'episode
meurt) est ce qui a permis de trouver, un par un, les defauts de la cellule.

    .../python.sh /workspace/sim/banc_automate.py [vitesse_tapis]
"""
import sys, torch
from pathlib import Path
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pick_cell import Automate, Cellule

n = 64
vit = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
c = Cellule(n_env=n, dt_physique=1/60, sous_pas=2, auto_collisions=False)
c.duree_max = 700
c.part_prise = 0.0
tous = torch.arange(n, device=c.dev)
c.vitesse[:] = vit
c.reinitialiser(tous)
auto = Automate(c)
auto.reinitialiser(tous)

causes = {"droite": 0, "morceau_bas": 0, "morceau_sorti": 0,
          **{f"fin_etat{e}": 0 for e in range(8)}}
cpt = {"posee": 0, "renversee": 0, "perdue": 0, "tronque": 0}
for k in range(1400):
    c.agir(auto.agir()); c.pas()
    r, fini, info = c.evaluer()
    ids = fini.nonzero(as_tuple=False).squeeze(-1)
    if ids.numel():
        for cle in cpt:
            cpt[cle] += int(info[cle][ids].sum())
        pa_, _ = c.pose_assiette(); pm_ = c.pose_morceau()
        d_ = c.assiette_droite()
        dedans_ = (pm_[:, :2] - pa_[:, :2]).norm(dim=1) < 0.034
        causes["droite"] += int((d_[ids] < 0.30).sum())
        causes["morceau_bas"] += int((pm_[ids][:, 2] < -0.05).sum())
        causes["morceau_sorti"] += int((~dedans_[ids] & (pa_[ids][:, 2] > 0.06)).sum())
        for e in range(8):
            causes[f"fin_etat{e}"] += int((auto.etat[ids] == e).sum())
        c.vitesse[ids] = vit
        c.reinitialiser(ids)
        auto.reinitialiser(ids)
    if k % 100 == 0:
        etats = [int((auto.etat == e).sum()) for e in range(8)]
        m1 = auto.etat == 5
        tcp_, q_ = c.poses_tcp(); pp_, lv_ = c.point_de_prise()
        if int(m1.sum()):
            dxy = float((tcp_[m1][:, :2] - pp_[m1][:, :2]).norm(dim=1).mean())
            dz = float((tcp_[m1][:, 2] - pp_[m1][:, 2]).mean())
            from pick_cell import ecart_lacet, lacet_de
            el = ecart_lacet(lacet_de(q_), lv_).abs()
            elm = float(el[m1].mean()); elok = float((el[m1] < 0.12).float().mean())
        else:
            dxy = dz = elm = elok = float("nan")
        print(f"pas {k:4d}  {cpt}  etats {etats}"
              f"  tenue {int(c.tenue.sum()):3d}/{n}"
              f"  transport: dxy {dxy:.4f} elac {elm:.3f}", flush=True)
        if int(m1.sum()):
            q = c.bras.get_joint_positions()[m1][:, :7]
            lim = c.limites[0, :7]
            marge = torch.min(q - lim[:, 0], lim[:, 1] - q)
            print("        axes  q :", [round(float(v), 2) for v in q.mean(0)],
                  " marge min :", [round(float(v), 2) for v in marge.min(0).values],
                  " tcp :", [round(float(v), 3) for v in tcp_[m1].mean(0)], flush=True)
tot = max(1, sum(cpt.values()))
print("causes :", causes, flush=True)
print(f"\nvitesse tapis {vit}  ->  {cpt}  "
      f"reussite {cpt['posee'] / tot:.1%}  sur {tot} episodes", flush=True)
app.close()
