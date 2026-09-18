"""Entrainement d'un classifieur de fromages sur les donnees normalisees.

Exemples
--------
  python src/train.py --task variety --model convnext_tiny --epochs 30
  python src/train.py --task hidb_product_ripeness --img-size 320 --epochs 25
  python src/train.py --task fr_cheese_type --balanced-sampler
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import (PROCESSED, CheeseDataset, TASKS, build_label_space,
                     build_transforms, channel_stats, load_manifest)

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="variety", choices=sorted(TASKS))
    p.add_argument("--model", default="convnext_tiny.fb_in22k_ft_in1k")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--min-per-class", type=int, default=1,
                   help="ecarte les classes sous ce nombre d'images")
    sampling = p.add_mutually_exclusive_group()
    sampling.add_argument("--balanced-sampler", action="store_true",
                          help="echantillonnage inversement proportionnel a la frequence")
    sampling.add_argument(
        "--domain-balanced-sampler", action="store_true",
        help=("equilibre les classes, puis les domaines presents dans chaque classe "
              "(ex. rendu source et camera usine)"),
    )
    p.add_argument("--loss", choices=("cross_entropy", "focal"),
                   default="cross_entropy")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--skip-test", action="store_true",
                   help="n'evalue pas le split test apres l'entrainement")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--init-checkpoint",
                   help="initialise depuis un checkpoint compatible avant adaptation")
    p.add_argument("--out", default=None, help="dossier de sortie (defaut: runs/<task>)")
    p.add_argument("--manifest", default=None,
                   help="manifeste alternatif, ex. data/processed/manifest_sim.csv")
    return p.parse_args()


class FocalLoss(nn.Module):
    """Multi-class focal loss with the same label smoothing as the baseline."""

    def __init__(self, gamma: float, label_smoothing: float = 0.0):
        super().__init__()
        if gamma < 0:
            raise ValueError("focal gamma must be non-negative")
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cross_entropy = F.cross_entropy(
            logits, targets, reduction="none", label_smoothing=self.label_smoothing,
        )
        target_probability = torch.softmax(logits, dim=1).gather(1, targets[:, None]).squeeze(1)
        return (((1.0 - target_probability) ** self.gamma) * cross_entropy).mean()


def domain_balanced_weights(dataset: CheeseDataset, n_classes: int) -> np.ndarray:
    """Give every class equal mass, split equally across its observed domains."""
    targets = np.asarray(dataset.targets)
    domains = np.asarray([row["domain"] for row in dataset.rows], dtype=object)
    weights = np.zeros(len(targets), dtype=np.float64)
    for target in range(n_classes):
        class_mask = targets == target
        class_domains = sorted(set(domains[class_mask]))
        if not class_domains:
            continue
        for domain in class_domains:
            stratum = class_mask & (domains == domain)
            weights[stratum] = 1.0 / (len(class_domains) * int(stratum.sum()))
    if not np.all(weights > 0):
        raise ValueError("domain-balanced sampler found an empty training stratum")
    return weights


@torch.no_grad()
def evaluate(model, loader, device, n_classes, criterion):
    model.eval()
    loss_sum, seen = 0.0, 0
    preds, labels, top5_hits = [], [], 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(images)
            loss = criterion(logits, targets)
        loss_sum += loss.item() * targets.size(0)
        seen += targets.size(0)
        preds.append(logits.argmax(1).cpu())
        labels.append(targets.cpu())
        k = min(5, n_classes)
        top5 = logits.topk(k, dim=1).indices
        top5_hits += (top5 == targets[:, None]).any(1).sum().item()

    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    from sklearn.metrics import f1_score
    return {
        "loss": loss_sum / max(seen, 1),
        "top1": float((preds == labels).mean()),
        "top5": top5_hits / max(seen, 1),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "preds": preds,
        "labels": labels,
    }


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out) if args.out else ROOT / "runs" / args.task
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(Path(args.manifest) if args.manifest else None)
    subset, class_to_idx, column = build_label_space(rows, args.task, args.min_per_class)
    classes = sorted(class_to_idx, key=class_to_idx.get)
    mean, std = channel_stats()

    train_tf = build_transforms(args.img_size, mean, std, train=True)
    eval_tf = build_transforms(args.img_size, mean, std, train=False)
    sets = {
        s: CheeseDataset(subset, class_to_idx, column, s,
                         train_tf if s == "train" else eval_tf)
        for s in ("train", "val", "test")
    }

    print(f"tache      : {args.task}  ({len(classes)} classes)")
    print(f"images     : train={len(sets['train'])} val={len(sets['val'])} test={len(sets['test'])}")
    print(f"normalis.  : mean={mean} std={std}")
    if not len(sets["train"]) or not len(sets["val"]):
        print("split vide, rien a entrainer")
        return 1

    sampler, shuffle = None, True
    if args.balanced_sampler:
        counts = np.bincount(sets["train"].targets, minlength=len(classes))
        weights = 1.0 / np.maximum(counts[sets["train"].targets], 1)
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                        num_samples=len(weights), replacement=True)
        shuffle = False
    elif args.domain_balanced_sampler:
        weights = domain_balanced_weights(sets["train"], len(classes))
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                        num_samples=len(weights), replacement=True)
        shuffle = False

    loaders = {
        "train": DataLoader(sets["train"], batch_size=args.batch_size, shuffle=shuffle,
                            sampler=sampler, num_workers=args.workers, pin_memory=True,
                            drop_last=len(sets["train"]) > args.batch_size,
                            persistent_workers=args.workers > 0),
        **{s: DataLoader(sets[s], batch_size=args.batch_size * 2, shuffle=False,
                         num_workers=args.workers, pin_memory=True,
                         persistent_workers=args.workers > 0)
           for s in ("val", "test") if len(sets[s])},
    }

    model = timm.create_model(args.model,
                              pretrained=not args.no_pretrained and not args.init_checkpoint,
                              num_classes=len(classes))
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        expected = {
            "model_name": args.model,
            "classes": classes,
            "img_size": args.img_size,
        }
        observed = {key: initial.get(key) for key in expected}
        if observed != expected:
            raise ValueError(
                f"incompatible initial checkpoint: expected {expected}, observed {observed}"
            )
        model.load_state_dict(initial["state_dict"])
    model = model.to(device, memory_format=torch.channels_last)

    if args.loss == "focal":
        criterion = FocalLoss(args.focal_gamma, args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    steps = max(1, len(loaders["train"]))
    warmup = args.warmup_epochs * steps
    total = args.epochs * steps

    def lr_at(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)

    history, best = [], {"macro_f1": -1.0, "epoch": -1}
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, loss_sum, seen, correct = time.time(), 0.0, 0, 0
        for images, targets in loaders["train"]:
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images)
                loss = criterion(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            loss_sum += loss.item() * targets.size(0)
            seen += targets.size(0)
            correct += (logits.argmax(1) == targets).sum().item()

        val = evaluate(model, loaders["val"], device, len(classes), criterion)
        entry = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": loss_sum / max(seen, 1),
            "train_top1": correct / max(seen, 1),
            "val_loss": val["loss"],
            "val_top1": val["top1"],
            "val_top5": val["top5"],
            "val_macro_f1": val["macro_f1"],
            "seconds": round(time.time() - t0, 1),
        }
        history.append(entry)
        print(f"epoch {epoch:3d}/{args.epochs}  "
              f"train_loss {entry['train_loss']:.3f} acc {entry['train_top1']:.3f}  |  "
              f"val_loss {entry['val_loss']:.3f} top1 {entry['val_top1']:.3f} "
              f"top5 {entry['val_top5']:.3f} macroF1 {entry['val_macro_f1']:.3f}  "
              f"({entry['seconds']}s)", flush=True)

        if val["macro_f1"] > best["macro_f1"]:
            best = {"macro_f1": val["macro_f1"], "top1": val["top1"], "epoch": epoch}
            torch.save({
                "model_name": args.model,
                "state_dict": model.state_dict(),
                "classes": classes,
                "task": args.task,
                "img_size": args.img_size,
                "mean": mean,
                "std": std,
                "val_metrics": {k: v for k, v in val.items() if k not in ("preds", "labels")},
                "args": vars(args),
            }, out_dir / "best.pt")

    result = {"best_val": best, "history": history}
    if not args.skip_test and "test" in loaders and len(sets["test"]):
        ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        test = evaluate(model, loaders["test"], device, len(classes), criterion)
        from sklearn.metrics import classification_report
        report = classification_report(test["labels"], test["preds"],
                                       labels=list(range(len(classes))),
                                       target_names=classes, zero_division=0,
                                       output_dict=True)
        result["test"] = {k: v for k, v in test.items() if k not in ("preds", "labels")}
        result["per_class"] = report
        print(f"\nTEST  top1 {test['top1']:.4f}  top5 {test['top5']:.4f}  "
              f"macroF1 {test['macro_f1']:.4f}")

    (out_dir / "results.json").write_text(json.dumps(result, indent=2, default=float))
    print(f"\nmeilleur epoch {best['epoch']} (val macroF1 {best['macro_f1']:.4f})")
    print(f"artefacts : {out_dir/'best.pt'}, {out_dir/'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
