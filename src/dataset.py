"""Dataset PyTorch adosse au manifeste normalise."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

# Regroupement des 12 types de Food Recognition en familles de tri : ce sont les
# bacs que le bras robot doit viser sur le tapis, pas des varietes commerciales.
BIN_OF_FR_LABEL = {
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
    "cheese_generic": "",  # trop vague pour designer un bac : on l'ecarte
}

# Deux etiquettes qui ne designent pas un bac : le bras ne doit rien ramasser.
REJECT_LABELS = {"not_cheese", "empty"}

# Taches disponibles -> (sources concernees, colonne utilisee comme etiquette)
TASKS = {
    # 290 varietes de fromage (photos web)
    "variety": (("cheese_images",), "label"),
    # produit seul : 3 a 4 meules par classe, c'est le seul decoupage de HIDB
    # qui supporte un split par meule et donc une evaluation honnete
    "hidb_product": (("cheese_hidb",), "product"),
    # maturite seule : target vs not-target
    "hidb_ripeness": (("cheese_hidb",), "ripeness"),
    # 6 classes produit x maturite : 1 a 2 meules par classe, donc pas de
    # validation tenable -- conserve pour reference, pas pour mesurer
    "hidb_product_ripeness": (("cheese_hidb",), "label"),
    # 12 types de fromage decoupes dans des scenes de repas
    "fr_cheese_type": (("food_recognition",), "label"),
    # 5 bacs de tri : la tete "ou poser la piece" du demo Isaac Sim
    "bin": (("food_recognition",), "bin"),
    # axe texture commun HIDB + Food Recognition
    "texture": (("cheese_hidb", "food_recognition"), "texture"),
    # rendus Isaac Sim : meme tache que `bin`, mais dans le domaine cible
    "sim_bin": (("sim_belt",), "label"),
    # Tete fromage-seulement d'un routeur en deux etages. Les lignes de rejet
    # sont exclues en laissant cette etiquette vide.
    "sim_bin_cheese": (("sim_belt",), "bin_cheese"),
    # Premier etage du meme routeur : fromage, objet etranger, ou tapis vide.
    "sim_gate": (("sim_belt",), "route_gate"),
    # variante sans les classes de rejet, pour mesurer ce que le rejet coute
    # sur la tache fromage a decoupage identique
    "sim_type_cheese": (("sim_belt",), "fr_type_cheese"),
    # LE modele a livrer : 11 types fins sur les rendus. Le bac s'en deduit en
    # sommant les probabilites des types d'un meme bac (cf. BIN_OF_FR_LABEL),
    # donc un seul reseau donne le nom ET la decision de tri.
    "sim_type": (("sim_belt",), "fr_type"),
    # tout, toutes sources confondues (espace d'etiquettes prefixe par la source)
    "all": (("cheese_hidb", "cheese_images", "food_recognition"), "source_label"),
}


def load_manifest(manifest: Path | None = None) -> list[dict]:
    path = manifest or PROCESSED / "manifest.csv"
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        extra = json.loads(row["extra"]) if row["extra"] else {}
        row["texture"] = extra.get("texture") or ""
        row["product"] = extra.get("product") or ""
        row["ripeness"] = extra.get("ripeness") or ""
        # Keep the capture domain available to samplers.  Historical manifests
        # do not carry this key, so their source name is the stable fallback.
        row["domain"] = extra.get("domain") or row["source"]
        row["source_label"] = f"{row['source']}/{row['label']}"
        row["bin"] = BIN_OF_FR_LABEL.get(row["label"], "") if row["source"] == "food_recognition" else ""
        # Sur les rendus, `label` est deja le bac et le type fin vit dans extra.
        # Les deux classes de rejet (`not_cheese`, `empty`) n'ont pas de type fin :
        # elles sont leur propre etiquette, sinon on entrainerait le modele sur les
        # 400 classes alimentaires d'origine au lieu d'un simple "pas du fromage".
        if row["source"] == "sim_belt":
            row["fr_type"] = (row["label"] if row["label"] in REJECT_LABELS
                              else extra.get("fr_label", ""))
            row["bin_cheese"] = "" if row["label"] in REJECT_LABELS else row["label"]
            row["route_gate"] = row["label"] if row["label"] in REJECT_LABELS else "cheese"
        else:
            row["fr_type"] = ""
            row["bin_cheese"] = ""
            row["route_gate"] = ""
        # meme chose, mais vide sur les classes de rejet : build_label_space
        # ecarte les lignes sans etiquette, donc la tache porte sur les fromages seuls
        row["fr_type_cheese"] = ("" if row["fr_type"] in REJECT_LABELS
                                 else row["fr_type"])
    return rows


def build_label_space(rows: list[dict], task: str, min_per_class: int = 1):
    sources, column = TASKS[task]
    subset = [r for r in rows if r["source"] in sources and r[column]]
    counts: dict[str, int] = {}
    for row in subset:
        counts[row[column]] = counts.get(row[column], 0) + 1
    classes = sorted(c for c, n in counts.items() if n >= min_per_class)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    subset = [r for r in subset if r[column] in class_to_idx]
    return subset, class_to_idx, column


class CheeseDataset(Dataset):
    """Lit les images normalisees (JPEG RGB, grand cote 512)."""

    def __init__(self, rows, class_to_idx, column, split, transform, root=PROCESSED):
        self.rows = [r for r in rows if r["split"] == split]
        self.class_to_idx = class_to_idx
        self.column = column
        self.transform = transform
        self.root = Path(root)
        self.targets = [class_to_idx[r[column]] for r in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(self.root / row["path"]) as img:
            img = img.convert("RGB")
            sample = self.transform(img)
        return sample, self.targets[index]


def build_transforms(img_size: int, mean, std, train: bool):
    if train:
        return v2.Compose([
            v2.RandomResizedCrop(img_size, scale=(0.55, 1.0), ratio=(0.75, 1.333),
                                 antialias=True),
            v2.RandomHorizontalFlip(),
            v2.RandAugment(num_ops=2, magnitude=7),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
            v2.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ])
    return v2.Compose([
        v2.Resize(int(img_size * 1.14), antialias=True),
        v2.CenterCrop(img_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])


def channel_stats() -> tuple[list[float], list[float]]:
    stats = json.loads((PROCESSED / "stats.json").read_text())["channel_stats"]
    return stats["mean"], stats["std"]
