---
tags: [pipeline, gotcha]
priority: high
aliases: [Leakage, Splits]
---

# Splits and data leakage

> [!danger] This bit us three times. Read before touching any split.
> **The split key is the physical object, never the file.** The `group` column carries it.

## Trap 1 — CHEESE-HIDB wheels

[[CHEESE-HIDB]] holds 9 wheels shot 42 times each. A per-image split put the same wheel
in train *and* test: **100% macro-F1 by epoch 13**, meaningless. Fixed by reconstructing
the wheel from consecutive DSC blocks. With an honest per-wheel split, [[hidb_product]]
drops to 0.727 macro-F1 — *that gap is what the leak was hiding*.

## Trap 2 — renders of the same piece

The 5 renders of one piece are one object under five angles. `group` = the cutout uid.

| split | images | pieces |
|---|---|---|
| train | 8655 | 2235 |
| val | 1850 | 478 |
| test | 1850 | 478 |

2235 + 478 + 478 = 3191 pieces. No piece straddles a boundary — verified.

## Trap 3 — comparing models across a changed split

> [!bug] Caught late, and it nearly produced a false conclusion
> Adding the reject classes introduced two new bins into the split algorithm, which
> **moved the cheese piece assignments**. Evaluating the older 11-class model on the new
> test set gave 0.844 — but **56.3% of that test set had been in its training data**.
>
> Fix: the 11-class model was retrained on the current splits ([[sim_type11]]) so both
> models share one test set. Only then was the comparison valid — [[Model comparison]].

> [!tip] The lesson
> A split is part of a model's identity. Changing the data changes the split, and every
> previously trained model becomes incomparable until retrained.

## Also worth knowing

- Food Recognition's official `test` split has **no annotations** — splits are rebuilt
  locally, the original kept in `extra.official_split`.
- Multiple cheese instances in one meal photo share `group = img_path`.
- Each `empty` render is its own group — no recycling between train and test.

## The algorithm

See [[Algorithms#Group-aware stratified split]].
