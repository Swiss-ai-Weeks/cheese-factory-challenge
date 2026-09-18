# Stage 9 — targeted rare-bin data

Status: completed without model promotion on the RTX PRO 6000 workstation on
2026-09-18.

## Decision

Neither data intervention passed the immutable Stage 8 validation gate.
Production remains on `runs/sim_bin_adapt_v2/best.pt` at threshold `0.55`.
The target test split and deterministic 11-object Isaac scenario were not run,
because no validation winner existed.

The accepted deliverable from this stage is safer capture tooling: factory
camera generation can now be restricted to named splits and bins, invalid
values fail before the running Isaac service is touched, and the wrapper
restores Isaac only after a capture job actually stops it.

## Why target data rather than tune again

Stage 8 showed that sampler, focal-loss, initialization, two-stage, and fusion
changes did not pass the operational gate. Its audit also found only seven
blue training source groups and one blue validation source group. Stage 9
therefore tested additional evidence rather than another optimizer variation.

The unchanged gate required either:

1. correct-and-confident at least `0.7167`, coverage at least `0.85`, and
   foreign safe rejection at least `0.9425`; or
2. macro-F1 at least `0.4510`, correct-and-confident at least `0.6967`, and
   the same coverage and safety floors.

## Training-only targeted capture

The capture runner gained `CHEESE_CAPTURE_SPLITS` and
`CHEESE_CAPTURE_BINS`, both comma-separated and fail-closed. This command
captured views `v3` through `v11` only for training groups in the four weak
cheese bins:

```bash
CHEESE_CAPTURE_SPLITS=train \
CHEESE_CAPTURE_BINS=bin_blue,bin_fresh,bin_semi_hard,bin_soft \
CHEESE_CAPTURE_VIEWS=9 CHEESE_CAPTURE_VIEW_START=3 \
CHEESE_CAPTURE_CHUNK_SIZE=50 \
  infra/isaac-sim/capture-all-domain.sh
```

It produced 927/927 crops with no misses. The factory-only manifest grew from
2,007 to 2,934 unique rows. Training counts became 84 blue, 312 fresh, 288
hard, 432 semi-hard, 408 soft, and 810 foreign-object images.

Validation and test each remained exactly 300 rows. Their canonical
UID/group/label/path fingerprints remained, respectively:

- validation: `404f7aa616c00464c9fab79fb3c777cf1d3c351e3b1510d3d57672b11b8c06fa`
- test: `e0c035a6aa5fef9cd354b079eb18a9f66abe862f2ae77792b7df68bb1c06b631`

There were zero missing files, duplicate UIDs, or groups crossing splits.
Generated images and manifests remain ignored and local.

## Validation-only experiments

All rows below use the same 300-image factory-camera validation slice and
threshold `0.55`.

| experiment | isolated change | top-1 | macro-F1 | coverage | covered accuracy | correct + confident | foreign safe rejection | decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| accepted baseline | none | 0.7433 | 0.4010 | 0.8867 | 0.7857 | 0.6967 | 0.9425 | retained |
| E9a | nine extra training-only camera views for four weak bins | 0.6900 | 0.4035 | 0.8333 | 0.7720 | 0.6433 | 0.8736 | rejected |
| E9b | 36 curated web-blue training images on original three-view data | 0.7133 | 0.4123 | 0.8300 | 0.7992 | 0.6633 | 0.9253 | rejected |

E9a improved fresh recall from 0.0667 to 0.2667 and soft recall from 0.2500
to 0.2917, but materially reduced yield and safety. E9b improved semi-hard
recall from 0.2917 to 0.6250, but reduced soft, foreign-object, and operational
performance. Neither changed blue recall from 0/3.

For E9b, 178 training-split images whose variety names indicated blue cheese
were screened with the existing fine-type checkpoint. Only 36 with
`blue_mould_cheese` probability at least `0.50` were added. Visual inspection
confirmed genuine Roquefort, Stilton, Gorgonzola, Cabrales, and related cheese
among the selected examples. This screening was necessary: the source
dataset's generic `blue` label also contains unrelated blue objects, including
a dragonfly. The experiment used the original accepted three-view manifest,
not E9a's rejected additional views, so the intervention was isolated.

## Diagnosis and boundary

The three blue validation views are rotations of one pale, small, visually
ambiguous source cutout. More views of the same seven training objects and
additional web-domain semantics did not transfer to that source. This is a
data-quality and independent-group coverage limit, not evidence that a larger
model or longer training will solve the problem.

A defensible next perception experiment requires new independently sourced,
segmented blue-cheese objects captured through the factory camera and assigned
to train and validation before model selection. Until then, the fail-closed
production baseline is safer than promoting either Stage 9 candidate.

Checkpoints, generated manifests, crops, logs, and validation JSON remain under
ignored workstation paths. No generated data, model weight, raw dataset,
runtime output, or secret is committed.
