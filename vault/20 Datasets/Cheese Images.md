---
tags: [dataset]
source: https://huggingface.co/datasets/NoeFlandre/cheese-images
status: secondary
---

# Cheese Images (HuggingFace)

~3,200 web photos across **290 cheese varieties**, one folder per variety. After
de-duplication: 288 varieties, 3,211 images.

Heavily long-tailed — 1 to 92 images per variety, ~11 on average.

> [!warning] 125 of 288 varieties have no test image at all
> They have too few examples to hold any out. Any variety-level score is therefore
> measured on less than half the label space.

## Why it cannot name cheese on the belt

[[variety]] reaches 0.524 top-1 on its own domain — well above the 0.3% of chance, so it
does learn something. But on Isaac Sim renders its confidence **collapses**:

| | own domain (web photos) | belt renders |
|---|---|---|
| median confidence | 0.224 | **0.030** |
| images above 0.5 | 32% | **0%** |

On renders it concentrates on a handful of arbitrary varieties — `trou_du_cru` on 60 of
240 images. It is not recognising, it is guessing.

> [!failure] So the model cannot tell you "this is a Comté"
> And no data in the project could teach it to: Food Recognition carries no variety
> names, and cheese-images carries no paste type. The two label spaces never intersect.
> The shipped model answers `emmental_cheese`-level names instead — see [[sim_type13]].
