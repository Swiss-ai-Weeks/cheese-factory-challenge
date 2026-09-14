---
tags: [methodology, reference]
---

# Algorithms

The four pieces of logic that are not standard library calls.

## Group-aware stratified split

Standard stratified splitting assigns *rows*. Here it must assign **whole objects**, and
still balance classes. The algorithm:

```python
for source in sorted(by_source):
    rng = random.Random(f"{SEED}:{source}")       # per-source, independent
    groups_of_label = {label: {group, ...}}

    # rarest class first: it has the least room to reach its quota
    for label in sorted(groups_of_label, key=lambda l: (len(groups_of_label[l]), l)):
        gs = sorted(groups_of_label[label]); rng.shuffle(gs)
        free = [g for g in gs if g not in assigned]   # a group may already be placed
        if len(gs) < 3:                                # cannot hold anything out
            assign all to train; continue
        already = Counter(assigned[g] for g in gs if g in assigned)
        need_val  = quota_val  - already["val"]        # account for prior placements
        need_test = quota_test - already["test"]
        budget = max(0, len(free) - 1)                 # never empty the train set
        ...
```

Three properties that matter:
- **a group is assigned once** — a photo with two cheese types is placed by whichever
  class claims it first, and the second class accounts for it
- **rarest first** — otherwise a rare class finds all its groups already taken
- **per-source RNG** — so changing one source does not reshuffle the others

> [!warning] It is still order-dependent across *labels*
> Adding the reject classes changed the label set and moved cheese assignments. That is
> [[Splits and data leakage|trap 3]].

## Polygon → alpha channel

See [[Cutouts]]. Rasterise the exterior ring, subtract interior rings, feather 1.2 px,
put into the alpha channel.

## Probability marginalisation

See [[Probability marginalisation]].

## Camera look-at in USD

`Gf.Matrix4d().SetLookAt(eye, target, up)` returns a **view** matrix. A camera prim needs
the **world** transform, so the result must be inverted:

```python
op.Set(Gf.Matrix4d().SetLookAt(eye, target, up).GetInverse())
```

Spherical placement: azimuth × elevation × distance → cartesian, Z-up stage.
