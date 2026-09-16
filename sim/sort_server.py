"""Service de tri : une image de camera entre, une decision de bac sort.

Le conteneur Isaac Sim n'a ni torch ni timm ; le modele vit donc cote hote,
dans `.venv`, et la scene l'interroge en HTTP. C'est exactement le decoupage
attendu sur la ligne reelle : la camera et l'automate d'un cote, la brique de
perception de l'autre.

    .venv/bin/python sim/sort_server.py --checkpoint runs/sim_type13/best.pt

    POST /predict   corps = PNG brut          -> SortResult en JSON
    GET  /health                              -> {"ok": true, ...}
"""

from __future__ import annotations

import argparse, io, json, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "src"))

from PIL import Image                               # noqa: E402
from predict import CheeseSorter                    # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default=str(RACINE / "runs/sim_type13/best.pt"))
ap.add_argument("--host", default="0.0.0.0")
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--min-confidence", type=float, default=0.55)
args = ap.parse_args()

sorter = CheeseSorter(args.checkpoint, min_confidence=args.min_confidence)
sorter.warmup()
verrou = threading.Lock()
compteur = {"n": 0}
print(f"tri pret : {len(sorter.types)} types -> {sorter.bins} "
      f"(seuil {args.min_confidence})", flush=True)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _repondre(self, code: int, charge: dict):
        corps = json.dumps(charge).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(corps)))
        self.end_headers()
        self.wfile.write(corps)

    def do_GET(self):
        if self.path.startswith("/health"):
            self._repondre(200, {"ok": True, "types": sorter.types,
                                 "bins": sorter.bins, "servies": compteur["n"]})
        else:
            self._repondre(404, {"erreur": "route inconnue"})

    def do_POST(self):
        if not self.path.startswith("/predict"):
            return self._repondre(404, {"erreur": "route inconnue"})
        taille = int(self.headers.get("Content-Length", 0))
        brut = self.rfile.read(taille)
        try:
            with Image.open(io.BytesIO(brut)) as img:
                img = img.convert("RGB")
                t0 = time.perf_counter()
                with verrou:                 # un seul passage GPU a la fois
                    res = sorter.predict(img)
        except Exception as exc:             # une frame ratee ne doit pas tuer la ligne
            return self._repondre(500, {"erreur": f"{type(exc).__name__}: {exc}"})
        compteur["n"] += 1
        charge = res.as_dict()
        charge["server_ms"] = (time.perf_counter() - t0) * 1e3
        self._repondre(200, charge)

    def log_message(self, *a):               # pas de bruit sur stderr
        pass


srv = ThreadingHTTPServer((args.host, args.port), Handler)
print(f"ecoute sur http://{args.host}:{args.port}", flush=True)
try:
    srv.serve_forever()
except KeyboardInterrupt:
    pass
