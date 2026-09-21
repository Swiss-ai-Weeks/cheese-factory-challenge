import json

import pytest

from src.build_evaluation_report import live_metrics, wilson


def test_wilson_interval_is_bounded_and_not_false_certainty():
    low, high = wilson(11, 11)
    assert 0.70 < low < 1.0
    assert high == pytest.approx(1.0)
    assert wilson(0, 0) is None


def test_live_report_separates_routing_manipulation_and_full_loop(tmp_path):
    evidence = {
        "schema_version": 2,
        "metadata": {
            "commit": "a" * 40,
            "runtime_id": "run-1",
            "scenario": "test",
            "classifier_mode": "model",
            "seed": 2026,
            "config": "sim/factory/config.yaml",
            "config_sha256": "b" * 64,
            "isaac_image": "image@sha256:" + "c" * 64,
            "max_objects": 3,
            "generated_at_epoch": 1.0,
        },
        "metrics": {},
        "records": [
            {
                "object_id": "empty",
                "ground_truth": "empty",
                "detected": False,
                "pick_attempted": False,
                "end_to_end_success": True,
                "cycle_time_s": 0.0,
            },
            {
                "object_id": "a",
                "ground_truth": "hard_cheese",
                "detected": True,
                "classification_correct": True,
                "predicted_bin": "bin_hard",
                "status": "ok",
                "pick_attempted": True,
                "pick_success": True,
                "correct_bin": True,
                "end_to_end_success": True,
                "cycle_time_s": 10.0,
            },
            {
                "object_id": "b",
                "ground_truth": "soft_cheese",
                "detected": True,
                "classification_correct": False,
                "predicted_bin": None,
                "status": "uncertain",
                "pick_attempted": False,
                "pick_success": False,
                "correct_bin": False,
                "end_to_end_success": False,
                "failure_reason": "not_actionable",
                "cycle_time_s": 4.0,
            },
            {
                "object_id": "foreign",
                "ground_truth": "not_cheese",
                "detected": True,
                "classification_correct": True,
                "predicted_bin": None,
                "status": "not_cheese",
                "pick_attempted": False,
                "pick_success": False,
                "correct_bin": True,
                "end_to_end_success": True,
                "cycle_time_s": 2.0,
            },
        ],
    }
    path = tmp_path / "run.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")

    result = live_metrics(path, "model")
    assert result["routing_decision"]["successes"] == 1
    assert result["routing_decision"]["samples"] == 2
    assert result["pick_attempt_coverage"]["successes"] == 1
    assert result["manipulation_given_attempt"]["rate"] == 1.0
    assert result["full_loop_outcome"]["successes"] == 2
    assert result["full_loop_outcome"]["samples"] == 3
    assert result["failure_counts"] == {"not_actionable": 1}


def test_live_report_rejects_wrong_mode(tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"schema_version": 2, "metadata": {"classifier_mode": "showcase"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="expected 'model'"):
        live_metrics(path, "model")
