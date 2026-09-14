---
tags: [results, gotcha]
priority: high
---

# Measurement pitfalls

> [!danger] Four times, a number was wrong in a way no error message would show.

## 1 — A split that leaked (100% macro-F1)

[[CHEESE-HIDB]] was split per image, so the same wheel appeared in train and test.
The model memorised the wheel. **Symptom:** a suspiciously perfect score by epoch 13.
**Detection:** looking at the images, then at the filename numbering.
→ [[Splits and data leakage]]

## 2 — A quarter of the dataset that never rendered

Two render shards were seeded differently, and the seed also drove the shuffle before
sharding. A quarter of pieces rendered twice, a quarter never.
**Symptom:** 7,175 images where 9,545 were expected.
**Detection:** counting files. Nothing in the metrics would have shown it.
→ [[Isaac Sim rendering]]

## 3 — 1,190 images silently dropped from training

A later render pass overwrote a shard manifest, erasing the type metadata of 238 pieces.
Rows without a label are filtered out by `build_label_space` — silently.
**Symptom:** the manifest said 12,355, the trainer loaded 11,165.
**Detection:** comparing the two numbers.
→ [[Render manifest]]

## 4 — A phantom class that faked a regression

Rejected cheeses were labelled `REJET` in an evaluation array, creating a sixth class
with F1 0 inside a macro average over five real bins.
**Symptom:** bin macro-F1 apparently dropping 0.724 → 0.611 when reject classes were
added, nearly leading to a wrong architectural recommendation.
**Detection:** per-bin F1 showed no class had degraded; the arithmetic then confirmed
0.733 × 5/6 = 0.611 exactly.
→ [[Model comparison]]

## What the four have in common

> [!tip] Every one was caught by a **count**, never by a metric
> A number of files, a number of rows, a number of classes. Metrics are computed on
> whatever survives — they cannot report what is missing.
>
> Guards now in place:
> - the render chain refuses to train if the piece count or per-piece view count is off
> - `render_manifest.py` warns when a render has no type
> - splits are verified for group straddling after every rebuild
> - macro-F1 is always computed with an explicit `labels=` list
