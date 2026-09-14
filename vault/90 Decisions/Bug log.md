---
tags: [bug, gotcha]
priority: high
---

# Bug log

Four failures that produced **wrong numbers rather than error messages**. Every one was
caught by a count. → [[Measurement pitfalls]] for the pattern.

## 1. Wheel leakage in CHEESE-HIDB
**Symptom:** 100% macro-F1 by epoch 13.
**Cause:** 9 wheels × 42 turntable views, split per image.
**Fix:** `group` = the wheel, reconstructed from consecutive DSC blocks.
**Residue:** the product×ripeness task is simply not measurable. → [[CHEESE-HIDB]]

## 2. A quarter of the renders never produced
**Symptom:** 7,175 images where 9,545 were expected.
**Cause:** two shards seeded differently, and the seed also drove the shuffle *before*
sharding, so each sliced a different ordering. A quarter rendered twice (colliding
filenames), a quarter never.
**Fix:** fixed shuffle seed; `--seed` drives only domain randomisation; `--skip-existing`;
a guard that refuses to train on a bad count. → [[Isaac Sim rendering]]

## 3. 1,190 images silently dropped from training
**Symptom:** manifest said 12,355, trainer loaded 11,165.
**Cause:** a later render pass rewrote `manifest_0.json`, erasing 238 pieces' type
metadata. Rows with an empty label are filtered out silently.
**Fix:** the type is now joined from the cutouts manifest, which is authoritative; a
warning counts typeless renders. → [[Render manifest]]

## 4. A phantom class faked a regression
**Symptom:** bin macro-F1 apparently 0.724 → 0.611 when reject classes were added.
**Cause:** rejected cheeses labelled `REJET` created a sixth class with F1 0 inside a
macro average over five bins. 0.733 × 5/6 = 0.611 exactly.
**Fix:** always pass an explicit `labels=` to `f1_score`.
**Near-miss:** nearly led to recommending a two-head architecture. → [[Model comparison]]

## Smaller ones

| issue | fix |
|---|---|
| system Python had no `ensurepip` | `uv`, standalone → [[Environment]] |
| Isaac container is not root, writes fail after 40 s startup | `chmod 777` the output dir |
| first render frames fail while the engine warms | 8 throwaway steps + 3 retries |
| `UsdGeom.Cylinder` renders as a decagon | hand-built 96-segment mesh |
| DatasetNinja signed link returns 404 | `dataset-tools` regenerates it |
| `sync_jupyter.sh` copied 3.6 GB of unused PNGs | excluded `sim/out` |
