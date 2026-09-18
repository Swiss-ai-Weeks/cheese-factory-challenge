"""Evaluate a trained checkpoint on an explicit manifest slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from dataset import CheeseDataset, TASKS, build_transforms, load_manifest
from predict import CheeseClassifier


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", default="sim_bin", choices=sorted(TASKS))
    parser.add_argument("--split", default="test")
    parser.add_argument("--uid-prefix", default="")
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output")
    args = parser.parse_args()

    classifier = CheeseClassifier(args.checkpoint, min_confidence=0.0)
    sources, column = TASKS[args.task]
    rows = [
        row for row in load_manifest(Path(args.manifest))
        if row["source"] in sources
        and row[column] in classifier.classes
        and (not args.uid_prefix or row["uid"].startswith(args.uid_prefix))
    ]
    class_to_idx = {label: index for index, label in enumerate(classifier.classes)}
    transform = build_transforms(
        classifier.img_size, classifier.mean, classifier.std, train=False
    )
    dataset = CheeseDataset(rows, class_to_idx, column, args.split, transform)
    if not dataset:
        raise SystemExit("the requested manifest slice is empty")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    probabilities = []
    labels = []
    for images, targets in loader:
        probabilities.append(classifier._forward(images))
        labels.append(targets.numpy())
    probabilities = np.concatenate(probabilities)
    labels = np.concatenate(labels)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    covered = confidence >= args.min_confidence
    names = classifier.classes
    correct = predictions == labels

    def quantiles(values: np.ndarray) -> dict[str, float] | None:
        if not len(values):
            return None
        points = (0.1, 0.25, 0.5, 0.75, 0.9)
        return {
            f"q{int(point * 100):02d}": float(value)
            for point, value in zip(points, np.quantile(values, points), strict=True)
        }

    per_class_confidence = {}
    for index, name in enumerate(names):
        mask = labels == index
        class_covered = mask & covered
        per_class_confidence[name] = {
            "support": int(mask.sum()),
            "coverage": float(covered[mask].mean()) if mask.any() else None,
            "covered_accuracy": (
                float(correct[class_covered].mean()) if class_covered.any() else None
            ),
            "correct_and_confident_rate": (
                float((correct[mask] & covered[mask]).mean()) if mask.any() else None
            ),
            "confidence_quantiles": quantiles(confidence[mask]),
        }

    foreign_mask = labels == names.index("not_cheese") if "not_cheese" in names else None
    if foreign_mask is not None and foreign_mask.any():
        reject_indices = {
            index for index, name in enumerate(names) if name in {"empty", "not_cheese"}
        }
        actionable = covered & np.asarray(
            [prediction not in reject_indices for prediction in predictions]
        )
        unsafe_foreign = foreign_mask & actionable
        foreign_safety = {
            "support": int(foreign_mask.sum()),
            "unsafe_actionable": int(unsafe_foreign.sum()),
            "safe_rejection_rate": float(1.0 - unsafe_foreign.sum() / foreign_mask.sum()),
        }
    else:
        foreign_safety = None

    result = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "split": args.split,
        "uid_prefix": args.uid_prefix,
        "samples": int(len(labels)),
        "top1": float((predictions == labels).mean()),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "threshold": args.min_confidence,
        "coverage": float(covered.mean()),
        "covered_accuracy": (
            float((predictions[covered] == labels[covered]).mean()) if covered.any() else None
        ),
        "correct_and_confident_rate": float((correct & covered).mean()),
        "confidence_quantiles": {
            "all": quantiles(confidence),
            "correct": quantiles(confidence[correct]),
            "incorrect": quantiles(confidence[~correct]),
        },
        "per_class_confidence": per_class_confidence,
        "foreign_object_safety": foreign_safety,
        "classes": names,
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=list(range(len(names)))
        ).tolist(),
        "per_class": classification_report(
            labels, predictions, labels=list(range(len(names))),
            target_names=names, zero_division=0, output_dict=True,
        ),
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
