---
tags: [pipeline]
script: src/render_manifest.py
---

# Render manifest

Turns rendered PNGs into a training manifest with leak-free splits.

```bash
.venv/bin/python src/render_manifest.py
```

## What it does

1. **Enumerates the files on disk** — not the shard JSONs, which overwrite each other
2. **Recovers the fine type** by joining `group` (the cutout uid) against the cutouts
   manifest, which is authoritative
3. **Converts** 768 px PNG → 512 px JPEG q92 into `data/processed/images/sim_belt/`
4. **Assigns splits** grouped by piece — [[Algorithms#Group-aware stratified split]]
5. **Warns** if any render has no type, instead of letting it disappear

> [!bug] Both of those safeguards exist because of real failures
> A later render pass rewrote `manifest_0.json` and erased the previous pass's metadata.
> 238 pieces × 5 views = **1,190 cheese renders lost their type** and were silently
> excluded from training — the manifest said 12,355, the trainer loaded 11,165, and
> nothing flagged it. See [[Bug log]].

## Why convert to JPEG

Not for speed — training is GPU-bound and the conversion changed nothing there.
For **comparability**: renders were 768 px PNG while the real-photo baseline was 512 px
JPEG, so any gap would have mixed domain with resolution and compression.
Side benefit: 3.6 GB → 311 MB.
