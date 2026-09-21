"""Build the public evaluation report from committed machine-readable artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
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


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display_path(path: Path) -> str:
    try:
        path = path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        pass
    return str(path).replace("\\", "/")


def wilson(successes: int, samples: int, z: float = 1.959963984540054) -> list[float] | None:
    if samples <= 0:
        return None
    rate = successes / samples
    denominator = 1 + z * z / samples
    centre = rate + z * z / (2 * samples)
    margin = z * math.sqrt(rate * (1 - rate) / samples + z * z / (4 * samples * samples))
    return [
        max(0.0, (centre - margin) / denominator),
        min(1.0, (centre + margin) / denominator),
    ]


def binomial(successes: int, samples: int) -> dict:
    return {
        "successes": successes,
        "samples": samples,
        "rate": successes / samples if samples else None,
        "wilson_95": wilson(successes, samples),
    }


def offline_metrics(path: Path, provenance: dict, model_key: str) -> dict:
    payload = load(path)
    test = payload["test"]
    samples = int(payload["per_class"]["macro avg"]["support"])
    successes = round(float(test["top1"]) * samples)
    model = provenance["models"][model_key]
    return {
        "scope": "saved held-out test split; not live-camera or real-world accuracy",
        "artifact": display_path(path),
        "artifact_sha256": sha256(path),
        "checkpoint_sha256": model["sha256"],
        "training_seed": model["training_seed"],
        "manifest": model["manifest"],
        "top1": binomial(successes, samples),
        "macro_f1": float(test["macro_f1"]),
        "macro_f1_interval": None,
        "interval_note": "Predictions are not saved, so no defensible resampling interval can be regenerated for macro-F1.",
    }


def live_metrics(path: Path, expected_mode: str) -> dict:
    payload = load(path)
    if payload.get("schema_version") != 2:
        raise ValueError(f"{path}: expected evidence schema 2")
    metadata = payload["metadata"]
    if metadata.get("classifier_mode") != expected_mode:
        raise ValueError(f"{path}: expected {expected_mode!r} mode")
    records = payload["records"]
    objects = [row for row in records if row["ground_truth"] != "empty"]
    cheeses = [row for row in objects if row["ground_truth"] in BIN_OF_TYPE]
    detected = [row for row in objects if row.get("detected")]
    attempts = [row for row in cheeses if row.get("pick_attempted")]
    foreign = [row for row in objects if row["ground_truth"] == "not_cheese"]
    empty = [row for row in records if row["ground_truth"] == "empty"]
    failures = Counter((row.get("failure_reason") or row.get("status") or "unknown") for row in objects if not row.get("end_to_end_success"))
    return {
        "scope": "deterministic Isaac Sim scenario; not real-world performance",
        "artifact": display_path(path),
        "artifact_sha256": sha256(path),
        "metadata": metadata,
        "detection": binomial(sum(bool(row.get("detected")) for row in objects), len(objects)),
        "fine_type_classification": binomial(sum(bool(row.get("classification_correct")) for row in detected), len(detected)),
        "routing_decision": binomial(sum(row.get("predicted_bin") == BIN_OF_TYPE[row["ground_truth"]] for row in cheeses), len(cheeses)),
        "pick_attempt_coverage": binomial(len(attempts), len(cheeses)),
        "manipulation_given_attempt": binomial(sum(bool(row.get("pick_success")) for row in attempts), len(attempts)),
        "correct_bin_placement": binomial(sum(bool(row.get("correct_bin")) for row in cheeses), len(cheeses)),
        "safe_foreign_rejection": binomial(sum(not row.get("pick_attempted") and row.get("status") != "ok" for row in foreign), len(foreign)),
        "empty_interval_handling": binomial(sum(not row.get("detected") and not row.get("pick_attempted") for row in empty), len(empty)),
        "full_loop_outcome": binomial(sum(bool(row.get("end_to_end_success")) for row in objects), len(objects)),
        "mean_cycle_time_s": sum(float(row.get("cycle_time_s", 0)) for row in objects) / len(objects),
        "failure_counts": dict(sorted(failures.items())),
    }


def pct(metric: dict) -> str:
    low, high = metric["wilson_95"]
    return f"{metric['successes']}/{metric['samples']} ({metric['rate']:.1%}; 95% Wilson {low:.1%}–{high:.1%})"


def render_markdown(report: dict) -> str:
    lines = [
        "# Factory evaluation results",
        "",
        "> Generated by `src/build_evaluation_report.py`; do not edit metric values by hand.",
        "",
        "Classifier, routing, manipulation and full-loop evidence are deliberately separated. All intervals below are 95% Wilson intervals for binomial proportions. They quantify sampling uncertainty in these bounded datasets/scenarios, not sim-to-real uncertainty.",
        "",
        "## Offline held-out model evidence",
        "",
        "| Component | Sample | Top-1 | Macro-F1 | Seed |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, key in (("fine-type classifier", "type_classifier"), ("direct-bin router", "routing_classifier")):
        row = report["offline"][key]
        lines.append(f"| {label} | {row['top1']['samples']} | {pct(row['top1'])} | {row['macro_f1']:.4f} | {row['training_seed']} |")
    lines += [
        "",
        "Macro-F1 has no interval because the committed legacy artifacts contain aggregate reports, not per-sample predictions. These are held-out dataset results, not live-camera or factory-floor accuracy.",
        "",
        "## Deterministic Isaac integration evidence",
        "",
    ]
    for key, title in (("model", "Trained-model mode"), ("showcase", "Scripted showcase mode")):
        run = report["live"][key]
        m = run["metadata"]
        lines += [
            f"### {title}",
            "",
            f"Commit `{m['commit']}`, seed `{m['seed']}`, scenario `{m['scenario']}`, image `{m['isaac_image']}`.",
            "",
            "| Layer | Result |",
            "|---|---:|",
            f"| camera detection | {pct(run['detection'])} |",
            f"| fine-type classification | {pct(run['fine_type_classification'])} |",
            f"| correct route decision | {pct(run['routing_decision'])} |",
            f"| cheese pick coverage | {pct(run['pick_attempt_coverage'])} |",
            f"| manipulation given an attempt | {pct(run['manipulation_given_attempt'])} |",
            f"| correct-bin placement | {pct(run['correct_bin_placement'])} |",
            f"| safe foreign-object rejection | {pct(run['safe_foreign_rejection'])} |",
            f"| empty-interval handling | {pct(run['empty_interval_handling'])} |",
            f"| full-loop outcome | {pct(run['full_loop_outcome'])} |",
            f"| mean cycle time | {run['mean_cycle_time_s']:.2f} s |",
            "",
            f"Failure counts: `{json.dumps(run['failure_counts'], sort_keys=True)}`.",
            "",
        ]
    lines += [
        "Showcase routing uses known scenario labels and is not classifier evidence. The trained-model run includes real model errors but still uses a small deterministic simulated scenario. Neither establishes real-world safety, reliability or production accuracy.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "infra/isaac-sim/run-evaluation.sh model 11 docs/evidence/p22-model-results.json",
        "infra/isaac-sim/run-evaluation.sh showcase 11 docs/evidence/p22-showcase-results.json",
        ".venv/bin/python src/build_evaluation_report.py",
        "```",
        "",
        "The machine-readable summary is `docs/evidence/p22-evaluation-summary.json`. Source artifact hashes and checkpoint hashes are included there.",
        "",
    ]
    return "\n".join(lines)


def build(type_results: Path, routing_results: Path, model_run: Path, showcase_run: Path, provenance_path: Path) -> dict:
    provenance = load(provenance_path)
    model = live_metrics(model_run, "model")
    showcase = live_metrics(showcase_run, "showcase")
    return {
        "schema_version": 1,
        "confidence_method": "two-sided 95% Wilson score interval; z=1.959963984540054",
        "offline": {
            "type_classifier": offline_metrics(type_results, provenance, "type_classifier"),
            "routing_classifier": offline_metrics(routing_results, provenance, "routing_classifier"),
        },
        "live": {"model": model, "showcase": showcase},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--type-results", type=Path, default=ROOT / "runs/sim_type13/results.json")
    parser.add_argument("--routing-results", type=Path, default=ROOT / "runs/sim_bin_adapt_v2/results.json")
    parser.add_argument("--model-run", type=Path, default=ROOT / "docs/evidence/p22-model-results.json")
    parser.add_argument("--showcase-run", type=Path, default=ROOT / "docs/evidence/p22-showcase-results.json")
    parser.add_argument("--provenance", type=Path, default=ROOT / "config/runtime-provenance.json")
    parser.add_argument("--json-output", type=Path, default=ROOT / "docs/evidence/p22-evaluation-summary.json")
    parser.add_argument("--markdown-output", type=Path, default=ROOT / "docs/evaluation_results.md")
    args = parser.parse_args()
    report = build(args.type_results, args.routing_results, args.model_run, args.showcase_run, args.provenance)
    args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote {args.json_output.relative_to(ROOT)} and {args.markdown_output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
