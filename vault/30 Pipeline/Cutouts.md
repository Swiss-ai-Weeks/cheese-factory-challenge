---
tags: [pipeline]
script: src/cutouts.py
---

# Cutouts

RGBA pieces with a real silhouette, extracted from [[Food Recognition 2022]] polygons.

![[cutouts.png]]

## Why

Normalised images are rectangles: the cheese **plus** the bread, plate or hand around it.
Dropping one into a 3D plate renders "a photograph lying in a plate".

## How — six lines, no model

> [!example] The contours were already in the data
> ```python
> points = obj["points"]["exterior"]                 # the annotator's polygon
> mask = Image.new("L", img.size, 0)
> ImageDraw.Draw(mask).polygon(points, fill=255)     # rasterise
> for hole in obj["points"].get("interior", []):     # subtract holes
>     ImageDraw.Draw(mask).polygon(hole, fill=0)
> mask = mask.filter(ImageFilter.GaussianBlur(1.2))  # feather, avoid scissor edges
> img.putalpha(mask)
> ```
> No SAM, no segmentation network. The real work was **finding the right polygon**:
> `object_index` lets you walk back from a normalised crop to the exact object.

## Filters

| filter | value | what it catches |
|---|---|---|
| `MIN_SIDE` | 64 px | instances too small to render |
| `MIN_FILL` | 0.12 | polygons covering under 12% of their box |
| `--min-fill` (render) | 0.35 | sliver silhouettes that read as paper |
| `--max-ar` (render) | 3.0 | extreme aspect ratios |

> [!warning] `fill_ratio` does not detect dirty cutouts
> It measures how much of the bounding box the polygon covers, so it catches *slivers*.
> A cutout that wrongly includes bread has a **high** fill ratio — it is a solid blob.
> Annotation errors are not detectable geometrically.

## Output

`data/processed/cutouts/<bin>/<uid>.png` — RGBA, long side ≤ 512.

 - `bin_blue` — 92
- `bin_fresh` — 417
- `bin_hard` — 678
- `bin_semi_hard` — 338
- `bin_soft` — 475
- `not_cheese` — 400

Total 2400, 88 rejected. The `not_cheese` entries come from [[Reject classes]].
