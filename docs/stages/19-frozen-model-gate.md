# Stage 19 — frozen model-promotion gate

Status: completed without promotion on the RTX PRO 6000 workstation on
2026-09-19. Production remains on `runs/sim_bin_adapt_v2/best.pt`.

## Gate frozen before training

`config/p14_candidate_gate.json` was staged before candidate training with
SHA-256 `e4191e...f06264a5` and Git blob `d7068c...44bc44`. The gate pins:

- the 90-image balanced validation manifest at SHA-256 `b4ea1f...aad4d0`;
- the 300-image operational validation manifest at SHA-256 `8ec626...8e89871`;
- threshold `0.55`, sample counts, split names, and production checkpoint hash;
- minimum aggregate, confident-yield, per-bin, coverage, and rejection metrics.

The balanced suite requires at least 0.4467 top-1, 0.3836 macro-F1, and 0.4022
correct-and-confident rate. Blue, fresh, semi-hard, and soft must also achieve
both the frozen recall and confident-correct floors. The operational suite must
match the accepted baseline's 0.7433 top-1, 0.4010 macro-F1, 0.6967
correct-and-confident rate, and 0.9425 foreign-object safe rejection, with at
least 0.85 coverage. Every check is mandatory.

`src/check_candidate_gate.py` recomputes both manifest hashes, verifies report
identity and threshold, requires the same checkpoint in both reports, and exits
nonzero on any failure. The factory test split and final 11-object Isaac
scenario remain locked until a candidate passes every validation check.

## Reproducible candidate data

The balanced P1.2 manifest intentionally has six visual classes and no `empty`
rows. Direct initialization from the seven-output production checkpoint was
therefore rejected by the trainer before epoch 1. This was repaired without
changing validation: `src/build_p14_candidate_manifest.py` deterministically
adds 78 independent, training-only `empty` rows from the existing adaptation
manifest.

The resulting ignored candidate manifest has SHA-256
`9f3b60...15b4a1`, 636 rows, no test rows, and:

| split | rows per class | classes | total |
|---|---:|---:|---:|
| train | 78 | 7, including `empty` | 546 |
| validation | 15 | 6 visual classes | 90 |

Rebuild it with:

```bash
.venv/bin/python src/build_p14_candidate_manifest.py
```

## Bounded candidates

Both candidates initialized from the production checkpoint, used the identical
candidate manifest, trained for 12 epochs with `--skip-test`, and were selected
only by the balanced validation macro-F1. V1 used learning rate `2e-5` and
label smoothing `0.1`; v2 used `5e-5` and label smoothing `0.05`.

| validation metric | baseline | v1 | v2 | frozen requirement |
|---|---:|---:|---:|---:|
| balanced top-1 | 0.3667 | 0.5111 | 0.5000 | >= 0.4467 |
| balanced macro-F1 | 0.3036 | 0.4451 | 0.4484 | >= 0.3836 |
| balanced correct + confident | 0.3222 | 0.4778 | 0.4889 | >= 0.4022 |
| balanced blue recall | 0.2000 | 0.8000 | 0.8000 | >= 0.2000 |
| balanced fresh recall | 0.0667 | 0.0667 | 0.0667 | >= 0.2000 |
| balanced semi-hard recall | 0.2667 | 0.4667 | 0.4000 | >= 0.3333 |
| balanced soft recall | 0.0000 | 0.0000 | 0.0000 | >= 0.2000 |
| operational top-1 | 0.7433 | 0.7567 | 0.7367 | >= 0.7433 |
| operational macro-F1 | 0.4010 | 0.4252 | 0.3838 | >= 0.4010 |
| operational coverage | 0.8867 | 0.8567 | 0.9033 | >= 0.8500 |
| operational correct + confident | 0.6967 | 0.7033 | 0.7133 | >= 0.6967 |
| operational safe rejection | 0.9425 | 0.9483 | 0.9425 | >= 0.9425 |

V1 passed every aggregate and operational guardrail but failed fresh and soft
recall and confident-correct requirements. V2 failed those same rare-bin checks
and regressed operational top-1 and macro-F1. Both were rejected; an aggregate
gain is not allowed to hide an unsolved destination.

The complete gate decisions are committed at
[`docs/evidence/p14-candidate-v1-gate.json`](../evidence/p14-candidate-v1-gate.json)
and
[`docs/evidence/p14-candidate-v2-gate.json`](../evidence/p14-candidate-v2-gate.json).

## Evidence and artifact boundary

No test image was loaded, no test metric was computed, and the final Isaac
scenario was not run. Candidate
checkpoints, generated manifests, training logs, and full evaluation reports
remain ignored on the workstation; no weights or generated datasets are
committed. The production checkpoint path, threshold, and running perception
service were not changed. This stage improves the promotion process and blue
recognition evidence, but makes no claim that production accuracy improved.
