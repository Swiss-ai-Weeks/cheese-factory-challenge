"""Brique perception : image camera -> label de tri, pour le pipeline robot.

Contrat de sortie (stable, c'est ce que consomme le controle du bras) :

    Prediction(
        label      : str    nom du bac, ex. "bin_hard"  (ou "unknown")
        index      : int    index de classe, -1 si unknown
        confidence : float  probabilite softmax de la classe retenue
        topk       : list[tuple[str, float]]
        latency_ms : float
    )

`label == "unknown"` quand la confiance passe sous `min_confidence` : le bras
doit alors laisser passer la piece plutot que la ranger au hasard.

Usage depuis Isaac Sim
----------------------
    from predict import CheeseClassifier

    clf = CheeseClassifier("runs/bin/best.pt", min_confidence=0.55)
    pred = clf.predict(rgb_frame)              # ndarray HxWx3 uint8
    send_to_robot(pred.label, pred.confidence)

    # plusieurs pieces detectees sur le tapis dans la meme frame :
    preds = clf.predict_batch(rgb_frame, boxes=[(x0, y0, x1, y1), ...])
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from torchvision.transforms import v2


@dataclass(frozen=True)
class Prediction:
    label: str
    index: int
    confidence: float
    topk: list
    latency_ms: float

    def as_dict(self) -> dict:
        return asdict(self)


class CheeseClassifier:
    """Enveloppe d'inference autour d'un checkpoint produit par ``train.py``."""

    def __init__(self, checkpoint: str | Path, device: str | None = None,
                 min_confidence: float = 0.0, topk: int = 3,
                 half: bool = True):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)

        self.classes: list[str] = ckpt["classes"]
        self.task: str = ckpt["task"]
        self.img_size: int = ckpt["img_size"]
        self.mean, self.std = ckpt["mean"], ckpt["std"]
        self.min_confidence = min_confidence
        self.topk = min(topk, len(self.classes))

        self.model = timm.create_model(ckpt["model_name"], pretrained=False,
                                       num_classes=len(self.classes))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval().to(self.device, memory_format=torch.channels_last)
        self.dtype = torch.float16 if (half and self.device.type == "cuda") else torch.float32
        self.model.to(self.dtype)

        self.transform = v2.Compose([
            v2.Resize(int(self.img_size * 1.14), antialias=True),
            v2.CenterCrop(self.img_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=self.mean, std=self.std),
        ])

    # -- preparation -------------------------------------------------------

    def _to_pil(self, frame) -> Image.Image:
        if isinstance(frame, Image.Image):
            return frame.convert("RGB")
        arr = np.asarray(frame)
        if arr.dtype != np.uint8:  # Isaac Sim rend parfois en float 0-1
            arr = np.clip(arr * 255.0 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[2] == 4:  # RGBA du capteur camera
            arr = arr[:, :, :3]
        return Image.fromarray(arr, mode="RGB")

    def _tensor(self, frame, box=None) -> torch.Tensor:
        img = self._to_pil(frame)
        if box is not None:
            img = img.crop(tuple(int(v) for v in box))
        return self.transform(img)

    # -- inference ---------------------------------------------------------

    @torch.inference_mode()
    def _forward(self, batch: torch.Tensor) -> np.ndarray:
        batch = batch.to(self.device, self.dtype, memory_format=torch.channels_last)
        logits = self.model(batch).float()
        return torch.softmax(logits, dim=1).cpu().numpy()

    def _decode(self, probs: np.ndarray, latency_ms: float) -> Prediction:
        order = np.argsort(-probs)[: self.topk]
        top = [(self.classes[i], float(probs[i])) for i in order]
        best_idx, best_p = int(order[0]), float(probs[order[0]])
        if best_p < self.min_confidence:
            return Prediction("unknown", -1, best_p, top, latency_ms)
        return Prediction(self.classes[best_idx], best_idx, best_p, top, latency_ms)

    def predict(self, frame, box=None) -> Prediction:
        """Une piece par frame : renvoie le label a transmettre au robot."""
        t0 = time.perf_counter()
        probs = self._forward(self._tensor(frame, box).unsqueeze(0))[0]
        return self._decode(probs, (time.perf_counter() - t0) * 1e3)

    def predict_batch(self, frame, boxes=None, frames=None) -> list[Prediction]:
        """Plusieurs pieces : soit N boites dans une frame, soit N frames."""
        t0 = time.perf_counter()
        if frames is not None:
            tensors = [self._tensor(f) for f in frames]
        elif boxes is not None:
            tensors = [self._tensor(frame, b) for b in boxes]
        else:
            tensors = [self._tensor(frame)]
        if not tensors:
            return []
        probs = self._forward(torch.stack(tensors))
        latency = (time.perf_counter() - t0) * 1e3 / len(tensors)
        return [self._decode(p, latency) for p in probs]

    def warmup(self, iterations: int = 3) -> None:
        """Absorbe le cout de compilation cuDNN avant la premiere piece reelle."""
        dummy = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        for _ in range(iterations):
            self.predict(dummy)


# --------------------------------------------------------------------------
# Tri complet : un seul reseau, deux niveaux de reponse
# --------------------------------------------------------------------------

# Les 5 bacs sont un regroupement strict des types fins. Un modele entraine sur
# les types donne donc le nom ET la decision de tri, sans second reseau.
BIN_OF_TYPE = {
    "hard_cheese": "bin_hard",
    "emmental_cheese": "bin_hard",
    "semi_hard_cheese": "bin_semi_hard",
    "raclette_cheese": "bin_semi_hard",
    "soft_cheese": "bin_soft",
    "goat_cheese_soft": "bin_soft",
    "processed_cheese": "bin_soft",
    "fresh_cheese": "bin_fresh",
    "cottage_cheese": "bin_fresh",
    "cream_cheese": "bin_fresh",
    "blue_mould_cheese": "bin_blue",
}

# Etiquettes qui ne designent aucun bac. Elles ne sont pas des erreurs du modele :
# ce sont des reponses valides, et les seules qui autorisent le pipeline a savoir
# qu'il ne faut pas actionner le bras.
REJECT_STATUS = {
    "empty": "empty",            # rien sur le tapis
    "not_cheese": "not_cheese",  # un objet, mais pas du fromage
}


@dataclass(frozen=True)
class SortResult:
    """Sortie du tri.

    `status` dit au pipeline ce qu'il doit faire, et c'est le champ a tester en
    premier :

        "ok"         un fromage est reconnu, `bin` est exploitable
        "empty"      rien sur le tapis, ne pas actionner le bras
        "not_cheese" un objet est la, mais ce n'est pas du fromage
        "uncertain"  sous le seuil de confiance : laisser passer

    Seul "ok" autorise un rangement. `bin` vaut None dans les trois autres cas,
    pour qu'un appelant distrait ne puisse pas ranger une piece par accident.

    `cheese_type` est le nom fin, a afficher. Le bac est systematiquement plus
    sur que lui : sa probabilite est la somme de celles de ses types.
    """
    status: str
    bin: str | None
    bin_confidence: float
    cheese_type: str
    type_confidence: float
    topk_types: list
    latency_ms: float

    @property
    def actionable(self) -> bool:
        """Vrai seulement si le bras peut ranger la piece."""
        return self.status == "ok"

    def as_dict(self) -> dict:
        return asdict(self)


class CheeseSorter:
    """Classifieur de type fin, dont on deduit le bac de tri.

        sorter = CheeseSorter("runs/sim_type/best.pt", min_confidence=0.55)
        r = sorter.predict(frame)
        if r.actionable:
            place_in_bin(r.bin)                # 'bin_hard'
        else:
            signal(r.status)                   # 'empty' | 'not_cheese' | 'uncertain'

    La decision de bac ne prend pas le type le plus probable pour le remapper :
    elle **somme les probabilites de tous les types d'un meme bac**. Un fromage
    hesitant entre `hard_cheese` et `emmental_cheese` reste franchement dans
    `bin_hard`, alors qu'un argmax aurait affiche une confiance divisee.
    """

    def __init__(self, checkpoint, device=None, min_confidence: float = 0.0,
                 topk: int = 3, half: bool = True,
                 bin_of_type: dict | None = None):
        self._clf = CheeseClassifier(checkpoint, device=device, min_confidence=0.0,
                                     topk=topk, half=half)
        self.min_confidence = min_confidence
        self.topk = topk
        self.types = self._clf.classes
        self.bin_of_type = dict(bin_of_type or BIN_OF_TYPE)

        inconnus = [t for t in self.types
                    if t not in self.bin_of_type and t not in REJECT_STATUS]
        if inconnus:
            raise ValueError(f"types sans bac ni statut attribue : {inconnus}")
        self.bins = sorted(set(self.bin_of_type[t] for t in self.types
                               if t in self.bin_of_type))
        self.reject_types = [t for t in self.types if t in REJECT_STATUS]

    @property
    def img_size(self) -> int:
        return self._clf.img_size

    def warmup(self, iterations: int = 3) -> None:
        self._clf.warmup(iterations)

    def _decode(self, probs: np.ndarray, latency_ms: float) -> SortResult:
        # On agrege par destination : un bac par famille de types, plus une
        # destination par etiquette de rejet.
        par_cible: dict[str, float] = {b: 0.0 for b in self.bins}
        for t in self.reject_types:
            par_cible[t] = 0.0
        for i, t in enumerate(self.types):
            cible = t if t in REJECT_STATUS else self.bin_of_type[t]
            par_cible[cible] += float(probs[i])

        cible = max(par_cible, key=par_cible.get)
        p_cible = par_cible[cible]

        ordre = np.argsort(-probs)[: self.topk]
        top = [(self.types[i], float(probs[i])) for i in ordre]
        i_type = int(ordre[0])
        p_type = float(probs[i_type])

        if p_cible < self.min_confidence:
            return SortResult("uncertain", None, p_cible, self.types[i_type],
                              p_type, top, latency_ms)
        if cible in REJECT_STATUS:
            return SortResult(REJECT_STATUS[cible], None, p_cible, cible,
                              p_type, top, latency_ms)
        return SortResult("ok", cible, p_cible, self.types[i_type],
                          p_type, top, latency_ms)

    def predict(self, frame, box=None) -> SortResult:
        t0 = time.perf_counter()
        probs = self._clf._forward(self._clf._tensor(frame, box).unsqueeze(0))[0]
        return self._decode(probs, (time.perf_counter() - t0) * 1e3)

    def predict_batch(self, frame, boxes=None, frames=None) -> list[SortResult]:
        t0 = time.perf_counter()
        if frames is not None:
            tensors = [self._clf._tensor(f) for f in frames]
        elif boxes is not None:
            tensors = [self._clf._tensor(frame, b) for b in boxes]
        else:
            tensors = [self._clf._tensor(frame)]
        if not tensors:
            return []
        probs = self._clf._forward(torch.stack(tensors))
        latency = (time.perf_counter() - t0) * 1e3 / len(tensors)
        return [self._decode(p, latency) for p in probs]


if __name__ == "__main__":
    import argparse, json

    ap = argparse.ArgumentParser(description="Test rapide du classifieur sur des images")
    ap.add_argument("checkpoint")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--min-confidence", type=float, default=0.0)
    args = ap.parse_args()

    clf = CheeseClassifier(args.checkpoint, min_confidence=args.min_confidence)
    clf.warmup()
    print(f"tache={clf.task}  {len(clf.classes)} classes  entree={clf.img_size}px")
    for path in args.images:
        with Image.open(path) as img:
            pred = clf.predict(img)
        print(f"{Path(path).name:44s} -> {pred.label:18s} {pred.confidence:.3f}  "
              f"{json.dumps(pred.topk)}  {pred.latency_ms:.1f}ms")
