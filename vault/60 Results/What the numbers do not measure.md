---
tags: [results, caveat]
priority: high
---

# What the numbers do not measure

> [!danger] Read before quoting any score from this project.

## What is measured

The ability to classify these source pieces under varied angles, lighting and framing,
with clean splits. That is a genuine measure of **viewpoint invariance** and of the
**reject behaviour**.

## What is not

**Performance on the actual demo scene.** Train and test share the same source photos,
the same renderer, the same plate, the same belt, and the same approximation — flat
decals, not volumes. A high score here says nothing more about the real belt than the
photo-trained [[bin]] does.

## Four known biases, most serious first

1. **Lighting is baked into the photo.** Each [[Cutouts|cutout]] carries the lighting of
   its original shot — flash, window, fluorescent — layered on the scene's own. That
   photographic style **correlates with the class**, so the model can learn JPEG noise or
   white balance instead of cheese.
2. **The piece is flat.** No thickness, no self-shadowing, no specular of its own.
3. **The background is always the same.** One plate, one belt — the model can lean on
   them invisibly.
4. **`empty` is trivially separable.** Its perfect recall inflates any blended metric;
   that is why [[sim_type13]] reports four separate numbers.

## What would settle it

See [[Open questions]]. Short version: **a hundred labelled frames from the actual Isaac
scene** would measure the gap instead of assuming it.
