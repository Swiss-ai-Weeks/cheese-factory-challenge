---
tags: [bug, gotcha]
priority: high
---

# Bug log

Failures that produced **wrong numbers rather than error messages**. Every one was caught
by a count. → [[Measurement pitfalls]] for the pattern.

Bugs 1–4 are perception, 5–10 are the [[Pick and place cell]]. The robotics ones share a
family resemblance: a physics engine that degrades silently instead of raising, and reads
that lag one step behind writes.

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

## 5. PhysX dropped contacts instead of failing

**Symptom:** at 1024 cells the cheese fell through the bottom of its own plate;
`part_tenue` stuck at 0.4%. The same scene at 256 cells reached 16% in five iterations.
**Cause:** `PxGpuDynamicsMemoryConfig::totalAggregatePairsCapacity` is sized for *a*
scene, not a thousand. Too small, PhysX does not fail — it **misses interactions** and
says so in a flood of errors drowned in a 10 MB log.
**Fix:** the GPU buffers are now sized from the cell count. 4,657 errors → 0.
**Pattern:** the failure was invisible in every metric except the one physical quantity
nobody was plotting. → [[Measurement pitfalls]]

## 6. A non-uniform scale on a collider makes the body drift

**Symptom:** the cheese accelerated upward at 8.7 g, left its plate in five steps, and
the episode was declared spilled.
**Cause:** the piece was a unit cube **scaled** to 26×22×10 mm. `usd_kit.py` already
documented this for a rigid body's own transform; it holds one level further down, for
its colliders. The plate's sixteen wall pavers had the same fault, and made the plate
slide out of the jaws.
**Fix:** primitives at native size, only rotated and translated.

## 7. Imposing a velocity does not make a conveyor

**Symptom:** 0.16 m/s commanded, 0.000 m travelled.
**Cause:** PhysX has no conveyor. The belt's own friction (1.3) cancels an imposed
velocity within one substep, and a body resting on a static slab simply stops.
**Fix:** impose the **advance** — position and velocity together. That is what a belt
does: it moves things, it does not push them.
**Aside:** `set_velocities(..., indices=...)` silently does nothing on a one-cell view,
which sent the diagnosis down the wrong path for an hour.

## 8. The arm does not stop where you write it

**Symptom:** the rest pose read 18 mm away from where the arm settled; a plate placed
into the jaws at that pose fell straight between them.
**Cause:** body transforms lag one step behind a joint write, *and* the arm sags under
its own weight below its setpoint.
**Fix:** gravity compensation on the arm links — which every real FR3 controller has —
and the rest pose measured after letting it settle.

## 9. Teleporting an arm catapults what is near it

**Symptom:** a "already gripped" reset sent the cheese off at 7 m/s.
**Cause:** writing the arm to a new pose makes its fingers sweep, in one step, the whole
volume between the old pose and the new one. Whatever was just written there is ejected.
**Fix:** a reset that never teleports a gripping arm — the plate is slid into the jaws
one step *after* the arm has been placed, when its transforms have caught up.

## 10. A dense reward that argues against succeeding

Filed at length in [[Pick and place cell]], because it is a design fault rather than a
bug: carrying the plate paid ~10 per step indefinitely, setting it down paid 80 once and
**ended the episode**. Loitering was worth ten times succeeding. 29 M steps went into
learning exactly that, and every metric looked healthy except the only one that mattered.

## Smaller ones

| issue | fix |
|---|---|
| system Python had no `ensurepip` | `uv`, standalone → [[Environment]] |
| Isaac container is not root, writes fail after 40 s startup | `chmod 777` the output dir |
| first render frames fail while the engine warms | 8 throwaway steps + 3 retries |
| `UsdGeom.Cylinder` renders as a decagon | hand-built 96-segment mesh |
| DatasetNinja signed link returns 404 | `dataset-tools` regenerates it |
| `sync_jupyter.sh` copied 3.6 GB of unused PNGs | excluded `sim/out` |
