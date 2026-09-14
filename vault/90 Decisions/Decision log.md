---
tags: [decision]
---

# Decision log

Each entry: what was decided, why, and what it cost.

## Keep the three label spaces separate

The datasets label different things and no image carries two kinds. Merging would also
be harmful: HIDB is always a wheel on cardboard, Food Recognition always a plate — the
background alone separates them. **Cost:** several heads instead of one, and no head can
name a variety *and* a paste type. → [[Datasets MOC]]

## Split by physical object, always

**Cost:** a rebuild and a retrain each time the data changes, because the split moves.
**Benefit:** the first measurement was a meaningless 100%. → [[Splits and data leakage]]

## One model, bin derived by marginalisation

The bins are a strict partition of the types, so one network answers both.
**Cost:** none measured — the derived bin matches a dedicated 5-class model.
→ [[Probability marginalisation]]

## Rejection as classes, not as a threshold

A threshold cannot distinguish an empty belt from an ambiguous cheese, and on an empty
belt a 5-class model *must* pick a bin. **Cost:** none measured on the cheese task.
→ [[Reject classes]] · [[Model comparison]]

## Render in Isaac rather than composite in 2D

> [!note] Chosen with a known trade-off, and never validated
> A flat decal render is close to a 2D homography plus a background plus a colour tint —
> all reproducible in the dataloader at zero GPU cost, with unlimited variation instead
> of 5 fixed views. Only cast shadow and plate specular truly need a renderer.
>
> Kept because `render_belt.py` becomes directly useful once real 3D assets exist — only
> the quad needs to become a mesh. **A 2D baseline was never built**, so the claim that
> Isaac was worth the hour remains untested. → [[Open questions]]

## Convert renders to 512 px JPEG

Not for speed — training is GPU-bound. For **comparability**: renders were 768 px PNG
while the baseline was 512 px JPEG, so any gap would have mixed domain with resolution.
Side benefit: 3.6 GB → 311 MB. → [[Render manifest]]

## Keep hyperparameters identical across compared models

No balanced sampler anywhere, despite an 18:1 imbalance. Changing the setting and the
domain at once would make every gap uninterpretable. **Cost:** `processed_cheese` is
never learned. → [[Class taxonomy]]

## 20 epochs, not 60

Corrected mid-project. Every head peaks before epoch 20; 60 wasted over an hour of GPU
across five models. → [[Training recipe]]

## Never touch the shared LaunchPad stack

Isaac runs in a disposable container, Jupyter gets a file copy rather than a new mount.
**Cost:** the copy is a snapshot needing manual resync. → [[Isaac Sim setup]] · [[Jupyter web app]]
