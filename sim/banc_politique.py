"""Que fait la politique courante ? On la regarde porter, pas seulement compter."""
import sys, torch
from pathlib import Path
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch.nn as nn
from pick_cell import Cellule

ck = torch.load(sys.argv[1] if len(sys.argv) > 1 else "/workspace/runs/pick_fr3/last.pt",
                map_location="cuda:0", weights_only=False)
print(f"checkpoint iteration {ck['iteration']}  reussite {ck['reussite']:.3f}", flush=True)


class Politique(nn.Module):
    def __init__(self, n_obs, n_act):
        super().__init__()
        def tronc(s):
            return nn.Sequential(nn.Linear(n_obs, 512), nn.ELU(),
                                 nn.Linear(512, 256), nn.ELU(),
                                 nn.Linear(256, 128), nn.ELU(), nn.Linear(128, s))
        self.acteur, self.critique = tronc(n_act), tronc(1)
        self.log_ecart = nn.Parameter(torch.zeros(n_act))


c = Cellule(n_env=64, dt_physique=1/60, sous_pas=2, auto_collisions=False)
c.duree_max = 400
c.part_prise = 1.0                       # on regarde la moitie "porter et poser"
c.vitesse[:] = 0.0
pol = Politique(ck["n_obs"], ck["n_act"]).to("cuda:0")
pol.load_state_dict(ck["politique"], strict=False); pol.eval()
moy, var = ck["norm"]["moy"].cuda(), ck["norm"]["var"].cuda()
tous = torch.arange(c.n, device=c.dev)
c.reinitialiser(tous)

for k in range(400):
    obs = ((c.observer() - moy) / (var.sqrt() + 1e-5)).clamp(-8, 8)
    with torch.no_grad():
        a = pol.acteur(obs)
    c.agir(a); c.pas()
    r, fini, info = c.evaluer()
    if k % 40 == 0 or k == 399:
        p_ass, _ = c.pose_assiette()
        voie = c.voies[c.cible]
        d_voie = (p_ass[:, :2] - voie).norm(dim=1)
        m = c.tenue > 0.5
        print(f"pas {k:3d}  tenue {int(m.sum()):3d}/{c.n}"
              f"  d_voie {float(d_voie[m].mean()) if int(m.sum()) else float('nan'):.3f}"
              f"  z_ass {float(p_ass[m][:,2].mean()) if int(m.sum()) else float('nan'):+.3f}"
              f"  |a| {float(a.abs().mean()):.3f}  pince {float(a[:,4].mean()):+.2f}"
              f"  r {float(r.mean()):+.2f}", flush=True)
app.close()
