---
tags: [pipeline]
script: src/normalize.py
---

# Normalisation

```bash
.venv/bin/python src/normalize.py --workers 64
```

Brings [[CHEESE-HIDB]], [[Cheese Images]] and [[Food Recognition 2022]] into one format.

## Output

- `data/processed/images/<source>/<class>/<uid>.jpg` — RGB JPEG, long side 512, q92
- `data/processed/manifest.csv` — one row per image, 16 columns
- `label_maps.json`, `stats.json`

## What it does beyond resizing

| step | detail |
|---|---|
| EXIF rotation | `ImageOps.exif_transpose` |
| colour space | alpha flattened on white; palette/CMYK → RGB |
| instance extraction | Food Recognition polygons → bbox + 8% margin |
| de-duplication | MD5 of decoded pixels — 19 removed |
| splits | group-aware, see [[Splits and data leakage]] |
| statistics | channel mean/std on the **train split only** |

## Current state

6486 images, 0 failures.

| source | images | classes | train/val/test |
|---|---|---|---|
 | `cheese_hidb` | 378 | 6 | 128/126/124 |
| `cheese_images` | 3211 | 288 | 2257/477/477 |
| `food_recognition` | 2897 | 12 | 2041/433/423 |

Constants: mean `[0.57893, 0.5269, 0.45318]`, std `[0.27042, 0.26677, 0.27936]`.

> [!note] Known imperfection
> The renders reuse these constants, computed on real photos, although their colour
> distribution differs (grey belt, white plate). It is an affine recentring and the
> network adapts — and keeping them identical makes [[Model comparison]] valid.
