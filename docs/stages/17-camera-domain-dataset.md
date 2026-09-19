# Stage 17 — balanced canonical-camera dataset

Status: completed on the RTX PRO 6000 workstation on 2026-09-19. This stage
prepares data only; it does not train or promote a model.

## Problem closed

The previous factory-camera manifest was group-safe but not a useful rare-bin
gate. Its training split had only seven independent blue-cheese objects and its
validation split had one. Additional rotations of those same objects could not
create independent evidence, and Stage 9 showed that web photos pasted directly
into training did not transfer.

P1.2 now supplies independently split blue-cheese objects, removes their web
backgrounds, captures them through the final `CheeseFactorySample` inspection
camera, and builds an exactly group-balanced train/validation dataset. The
existing test split was neither read nor modified.

## Source and mask gate

The normalized `cheese-images` manifest contained 131 named blue-cheese sources
in the existing train/validation splits after excluding the ambiguous generic
`blue` label. That generic label is contaminated with unrelated blue objects.

Six generated contact sheets were inspected. Packaging, menus, side dishes,
failed/fragmented masks, and images without an isolated cheese subject were
rejected. The committed review file approves 24 independent objects:

- 19 original training-split sources;
- 5 original validation-split sources;
- 0 test sources.

`src/prepare_camera_sources.py` reproduces only those approvals with pinned
`rembg==2.0.84` and `onnxruntime-gpu==1.23.2`. It verifies the U2Net model SHA-256
(`8d10d2f...12b491`), source identity, foreground fill, connected-component
dominance, centering and minimum extent before writing an RGBA cutout. Every
generated row records the original source hash, output hash, model hash, mask
measurements and human-review reason.

## Canonical Isaac capture

The approved objects were rendered through the same final camera, scene and
foreground detector used by the integrated factory:

```bash
CHEESE_CAPTURE_SOURCE_MANIFEST=data/processed/camera_sources_v2/manifest.csv \
CHEESE_CAPTURE_DATASET=factory_blue_v2 \
CHEESE_CAPTURE_SPLITS=train,val CHEESE_CAPTURE_BINS=bin_blue \
CHEESE_CAPTURE_VIEWS=3 CHEESE_CAPTURE_CHUNK_SIZE=50 \
  infra/isaac-sim/capture-all-domain.sh
```

Isaac produced 72/72 requested crops with no misses. Visual review of all three
contact sheets confirmed isolated blue-cheese objects on the canonical
inspection surface across the three deterministic orientations. The captures
are 77 x 102 pixels before the training transform, matching the existing target
camera pipeline rather than pretending web-photo pixels are target-domain data.

## Balanced manifest and audit

`src/build_camera_dataset.py` combines the new captures with existing canonical
captures, retains views 0–2 only, and selects complete independent source groups.
`src/audit_camera_dataset.py` then verifies each file and content hash, source
hash, camera-configuration hash, group boundary and view set.

The accepted generated manifest contains 558 rows:

| split | groups per class | rows per class | classes | total rows |
|---|---:|---:|---:|---:|
| train | 26 | 78 | 6 | 468 |
| validation | 5 | 15 | 6 | 90 |

The six classes are `bin_blue`, `bin_fresh`, `bin_hard`, `bin_semi_hard`,
`bin_soft`, and `not_cheese`. The audit found zero test rows, duplicate UIDs,
cross-split groups, missing files or hash mismatches. All 558 rows carry the
same canonical camera configuration hash
`08998514510d994bb5c9ce5be0f054ba21fa473f98518d9ce6b8d86449086e92`.
The machine-readable accepted audit is committed at
[`docs/evidence/p12-camera-dataset-audit.json`](../evidence/p12-camera-dataset-audit.json).

## Evidence boundary

This is a prospective train/validation dataset, not a claim of improved model
accuracy. P1.3 will add seeded Replicator variation; P1.4 will freeze a gate and
train candidates. Until a candidate passes that gate, production remains on
`runs/sim_bin_adapt_v2/best.pt` and all earlier reported metrics remain unchanged.

Raw sources, RGBA cutouts, camera crops, generated manifests, contact sheets and
model weights remain ignored and local. No dataset or checkpoint is committed.
