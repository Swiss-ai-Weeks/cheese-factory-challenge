from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch
import torch.nn.functional as F


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train import FocalLoss, domain_balanced_weights
from dataset import build_label_space, load_manifest


def test_domain_balancing_is_hierarchical_within_each_class():
    dataset = SimpleNamespace(
        targets=[0, 0, 0, 0, 1, 1, 1],
        rows=[
            {"domain": "source"},
            {"domain": "source"},
            {"domain": "source"},
            {"domain": "camera"},
            {"domain": "source"},
            {"domain": "camera"},
            {"domain": "camera"},
        ],
    )

    weights = domain_balanced_weights(dataset, n_classes=2)

    targets = np.asarray(dataset.targets)
    domains = np.asarray([row["domain"] for row in dataset.rows])
    assert np.isclose(weights[targets == 0].sum(), 1.0)
    assert np.isclose(weights[targets == 1].sum(), 1.0)
    for target in (0, 1):
        assert np.isclose(weights[(targets == target) & (domains == "source")].sum(), 0.5)
        assert np.isclose(weights[(targets == target) & (domains == "camera")].sum(), 0.5)


def test_focal_gamma_zero_matches_cross_entropy():
    logits = torch.tensor([[2.0, 0.5], [0.2, 1.5]])
    targets = torch.tensor([0, 1])

    observed = FocalLoss(gamma=0.0, label_smoothing=0.1)(logits, targets)
    expected = F.cross_entropy(logits, targets, label_smoothing=0.1)

    assert torch.allclose(observed, expected)


def test_two_stage_tasks_derive_labels_without_changing_splits(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "uid,source,label,split,path,extra\n"
        'a,sim_belt,bin_blue,train,a.jpg,"{}"\n'
        'b,sim_belt,not_cheese,val,b.jpg,"{}"\n'
        'c,sim_belt,empty,test,c.jpg,"{}"\n'
    )

    rows = load_manifest(manifest)
    cheese_rows, _, cheese_column = build_label_space(rows, "sim_bin_cheese")
    gate_rows, _, gate_column = build_label_space(rows, "sim_gate")

    assert cheese_column == "bin_cheese"
    assert [(row[cheese_column], row["split"]) for row in cheese_rows] == [
        ("bin_blue", "train")
    ]
    assert gate_column == "route_gate"
    assert [(row[gate_column], row["split"]) for row in gate_rows] == [
        ("cheese", "train"), ("not_cheese", "val"), ("empty", "test")
    ]
