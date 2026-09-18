# Stage 8 — rare-bin perception

Status: completed without promotion on the RTX PRO 6000 workstation on
2026-09-18.

## Decision

No candidate passed the validation-only acceptance gate. Production therefore
continues to use `runs/sim_bin_adapt_v2/best.pt` at threshold `0.55`. The target
test split and deterministic 11-object scenario were not run during Stage 8,
because those locked evaluations are allowed only after selecting a validation
winner.

This is a negative experimental result, not an accuracy improvement. It keeps
the working factory stable while making future training and evaluation safer.

## Audit and immutable protocol

The work began at commit `69619c97f102c805e3b059915953ef0267b09e48` on
`codex/hackathon-integration`. Preflight passed on Isaac Sim 6.1 and the NVIDIA
RTX PRO 6000. The production trained-model service and WebRTC stack were healthy
before training.

The factory-camera manifest contains 2,007 unique crops from 669 source groups:

| split | images | groups |
|---|---:|---:|
| train | 1,407 | 469 |
| validation | 300 | 100 |
| test | 300 | 100 |

There are no missing images, duplicate capture IDs, or groups shared across
splits. The combined manifest contains 5,652 unique rows from 969 groups and
also has zero cross-split groups. Candidate selection always used the same 300
factory-camera validation images. Stage 8 did not evaluate a test split.

The validation class support is highly uneven: 3 blue, 15 fresh, 60 hard, 24
semi-hard, 24 soft, and 174 foreign-object images. In particular, the three
blue views come from one source group, so the blue estimate has high variance.

## Selection criterion

The criterion was fixed before candidate results were examined:

1. Primary: correct-and-confident rate at threshold `0.55` at least `0.7167`
   (a +0.0200 absolute gain), coverage at least `0.85`, and foreign-object safe
   rejection at least `0.9425`.
2. Alternate: macro-F1 at least `0.4510` (a +0.0500 absolute gain),
   correct-and-confident at least `0.6967`, and the same coverage and safety
   floors.
3. The fail-closed output contract and rejection/empty tests must continue to
   pass.

Foreign safe rejection counts a foreign crop as unsafe only when a covered
prediction selects an actionable cheese bin. A covered `empty` or `not_cheese`
prediction remains non-actionable.

## Validation experiments

All seven-class rows below use the identical 300-image target-camera validation
slice and threshold `0.55`.

| experiment | isolated change | top-1 | macro-F1 | coverage | covered accuracy | correct + confident | foreign safe rejection | decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| accepted baseline | none | 0.7433 | 0.4010 | 0.8867 | 0.7857 | 0.6967 | 0.9425 | retained |
| original direct-bin | no camera adaptation | 0.6333 | 0.3341 | 0.4267 | 0.8359 | 0.3567 | 0.9425 | rejected |
| E2 | hierarchical class/domain sampler | 0.7300 | 0.4479 | 0.8000 | 0.8125 | 0.6500 | 0.9310 | rejected |
| E3 | focal loss, gamma 2 | 0.6833 | 0.3832 | 0.8000 | 0.7625 | 0.6100 | 0.8793 | rejected |
| E5 | initialize from original direct-bin checkpoint | 0.7133 | 0.4022 | 0.8533 | 0.7930 | 0.6767 | 0.9368 | rejected |

E2 showed a real but unusable rare-bin tradeoff: fresh recall rose from 0.0667
to 0.2667 and semi-hard recall from 0.2917 to 0.5417, while safety, coverage,
and operational yield regressed. E3 was worse and was not combined with E2.

The five-bin cheese-only E4a diagnostic used only the 126 cheese validation
images, so it is not directly comparable to the table. It measured 0.5476
top-1, 0.3235 macro-F1, 0.7619 coverage, and 0.5000 correct-and-confident. The
accepted router's conditional five-bin top-1 on those images was 0.5952. Since
E4a was worse and blue remained 0/3, a separate gate was not trained or
integrated.

Convex validation-only fusion of the accepted direct-bin probabilities and the
fine-type model's bin-aggregated probabilities was also rejected. No weight in
the predeclared 0.05 grid met the gate; the accepted direct model remained the
best operational endpoint.

## Accepted model limitations

The retained baseline's unthresholded per-class validation report is:

| class | precision | recall | F1 | support | coverage | correct + confident |
|---|---:|---:|---:|---:|---:|---:|
| bin_blue | 0.0000 | 0.0000 | 0.0000 | 3 | 1.0000 | 0.0000 |
| bin_fresh | 0.1000 | 0.0667 | 0.0800 | 15 | 0.6667 | 0.0667 |
| bin_hard | 0.6575 | 0.8000 | 0.7218 | 60 | 0.9333 | 0.7667 |
| bin_semi_hard | 0.7778 | 0.2917 | 0.4242 | 24 | 0.7500 | 0.2917 |
| bin_soft | 0.3750 | 0.2500 | 0.3000 | 24 | 0.7500 | 0.1667 |
| not_cheese | 0.8385 | 0.9253 | 0.8798 | 174 | 0.9253 | 0.8678 |

All three blue views are confidently predicted as `not_cheese` (median
confidence 0.9077). This is the sharpest unresolved limitation. The validation
set contains no `empty` crop; empty-frame safety is enforced before
classification by the foreground detector and remains covered by unit and
integration tests.

## Reproduction

Baseline validation:

```bash
env LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib" \
  .venv/bin/python src/evaluate_checkpoint.py \
  runs/sim_bin_adapt_v2/best.pt \
  --manifest data/processed/manifest_factory_only.csv \
  --task sim_bin --split val --min-confidence 0.55 \
  --output runs/stage8_baselines/sim_bin_adapt_v2_val.json
```

E2 domain-balanced training:

```bash
env LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib" \
  .venv/bin/python src/train.py --task sim_bin \
  --manifest data/processed/manifest_factory_adapt.csv \
  --model convnext_base.fb_in22k_ft_in1k_384 --img-size 384 \
  --epochs 20 --batch-size 64 --lr 1e-4 --workers 8 \
  --domain-balanced-sampler --skip-test \
  --out runs/sim_bin_stage8_domain_v1
```

For E3, replace `--domain-balanced-sampler` with
`--balanced-sampler --loss focal --focal-gamma 2.0`. For E5, use
`--balanced-sampler --init-checkpoint runs/sim_bin/best.pt`. Candidate
evaluation uses the baseline command with the candidate checkpoint path.

`--skip-test` is mandatory during candidate development. The evaluator now
records confidence quantiles, correct-and-confident rate, per-class covered
metrics, confusion, and foreign-object safety in one machine-readable report.

## Artifact boundary

Checkpoints, training logs, validation reports, manifests, and images remain
ignored and local to the workstation. No `.pt`/`.onnx` file, generated image,
raw dataset, runtime output, or secret is committed. The rejected artifacts are
retained locally under `runs/sim_bin_stage8_*`; no dataset or checkpoint was
deleted. Production configuration and its checkpoint path were not changed.
