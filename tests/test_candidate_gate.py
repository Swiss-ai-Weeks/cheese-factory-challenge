import copy
import csv

from src.build_p14_candidate_manifest import build
from src.check_candidate_gate import check_candidate


def _fixture():
    report = {
        "checkpoint": "runs/candidate/best.pt",
        "manifest": "balanced.csv",
        "split": "val",
        "samples": 12,
        "threshold": 0.55,
        "macro_f1": 0.7,
        "foreign_object_safety": {"safe_rejection_rate": 1.0},
        "per_class": {"rare": {"recall": 0.5}},
    }
    spec = {
        "version": 1,
        "threshold": 0.55,
        "locked_until_validation_passes": ["test"],
        "suites": {
            "balanced": {
                "manifest": "balanced.csv",
                "manifest_sha256": "balanced-hash",
                "split": "val",
                "samples": 12,
                "minimums": {
                    "macro_f1": 0.6,
                    "foreign_object_safety.safe_rejection_rate": 0.9,
                    "per_class.rare.recall": 0.4,
                },
            },
            "operational": {
                "manifest": "operational.csv",
                "manifest_sha256": "operational-hash",
                "split": "val",
                "samples": 12,
                "minimums": {"macro_f1": 0.6},
            },
        },
    }
    operational = copy.deepcopy(report)
    operational["manifest"] = "operational.csv"
    return spec, {"balanced": report, "operational": operational}


def test_candidate_passes_only_when_every_check_passes():
    spec, reports = _fixture()
    result = check_candidate(
        spec, reports, {"balanced": "balanced-hash", "operational": "operational-hash"},
    )
    assert result["passed"] is True
    assert all(check["passed"] for check in result["checks"])


def test_metric_regression_fails_closed():
    spec, reports = _fixture()
    reports["balanced"]["per_class"]["rare"]["recall"] = 0.39
    result = check_candidate(
        spec, reports, {"balanced": "balanced-hash", "operational": "operational-hash"},
    )
    assert result["passed"] is False
    assert any(
        check["check"] == "per_class.rare.recall" and not check["passed"]
        for check in result["checks"]
    )


def test_manifest_or_checkpoint_drift_fails_closed():
    spec, reports = _fixture()
    reports["operational"]["checkpoint"] = "runs/other/best.pt"
    result = check_candidate(
        spec, reports, {"balanced": "wrong", "operational": "operational-hash"},
    )
    assert result["passed"] is False
    assert any(check["check"] == "manifest_sha256" and not check["passed"] for check in result["checks"])
    assert result["checks"][-1]["check"] == "same_candidate_checkpoint"
    assert result["checks"][-1]["passed"] is False


def test_candidate_manifest_adds_train_empty_without_reading_test(tmp_path):
    fields = ["uid", "label", "split", "group"]
    balanced = tmp_path / "balanced.csv"
    operational = tmp_path / "operational.csv"
    output = tmp_path / "candidate.csv"
    with balanced.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([
            {"uid": "blue-train", "label": "bin_blue", "split": "train", "group": "blue-a"},
            {"uid": "blue-val", "label": "bin_blue", "split": "val", "group": "blue-b"},
        ])
    with operational.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([
            {"uid": "empty-b", "label": "empty", "split": "train", "group": "empty-b"},
            {"uid": "empty-a", "label": "empty", "split": "train", "group": "empty-a"},
            {"uid": "empty-test", "label": "empty", "split": "test", "group": "empty-test"},
        ])

    audit = build(balanced, operational, output, empty_rows=1)
    rows = list(csv.DictReader(output.open()))
    assert audit == {
        "balanced_rows": 2,
        "empty_train_groups": 1,
        "empty_train_rows": 1,
        "rows": 3,
        "test_rows": 0,
    }
    assert [row["uid"] for row in rows] == ["blue-train", "blue-val", "empty-a"]
