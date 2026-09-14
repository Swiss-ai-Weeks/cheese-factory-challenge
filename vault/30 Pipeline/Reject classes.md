---
tags: [pipeline, ship-it]
---

# Reject classes

Two classes that do not name a bin. They are **valid answers**, and the only way the
pipeline can know it must not act.

![[reject-classes.png]]

## `not_cheese` — 400 cutouts, 1,910 renders

Food Recognition has 498 classes; only 11 are cheese. The other **482 are free
negatives**, with real human-drawn silhouettes.

400 objects were sampled across **400 distinct classes** — bread, tomato, egg, salmon,
avocado — by cycling through the class list rather than taking the most frequent, so the
model sees variety rather than 400 slices of bread. Anything whose class name contains
`cheese` was excluded, dishes included.

## `empty` — 900 renders

The belt with nothing on it. Half the images have **no plate at all**, half an **empty
plate**, so the model distinguishes "nothing to pick" from "a container is there".

Rendered by `render_belt.py --empty 900`, which hides the piece via
`UsdGeom.Imageable.MakeInvisible()` and randomises the scene as usual.

## What they buy

| | value |
|---|---|
| cheese vs reject accuracy | **0.953** |
| `empty` recall | **1.000** |
| `not_cheese` recall | 0.828 |
| cheese wrongly rejected | 2.4% (34 of 1,430) |

> [!success] And they cost nothing
> Per-bin F1 is unchanged or slightly better than the model without them —
> see [[Model comparison]]. No two-head architecture is needed.

> [!warning] 17% of foreign objects still pass as cheese
> `not_cheese` recall is 0.828. An unknown object has roughly a one-in-six chance of
> being sorted into a bin. More negatives, or a confidence gate, would help.

## Why classes and not a confidence threshold

A threshold can only say *"I am unsure"*. It cannot distinguish an empty belt from an
ambiguous cheese, and on an empty belt a 5-class model **must** pick a bin — sometimes
confidently. Making rejection a class lets the model answer honestly, and lets the
pipeline react differently to each cause. See [[Output contract]].
