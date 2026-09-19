# Stage 18 — seeded Replicator domain randomization

Status: completed on the RTX PRO 6000 workstation on 2026-09-19. This stage
adds and validates a data-generation path only; it does not train or promote a
model.

## Problem closed

The canonical-camera pipeline from P1.2 rendered reviewed source cutouts, but it
did not exercise Omniverse Replicator as a domain-randomization engine. P1.3 now
uses the installed `omni.replicator.core-1.13.36` extension inside Isaac Sim
6.1 to produce bounded, seeded variation around the actual integrated factory
scene rather than a disconnected preview stage.

Each render varies:

- carrier position, rotation, and cheese size;
- inspection-camera position and focal calibration;
- key, work, and dome illumination, including key-light colour;
- floor and inspection-plate material colour/roughness;
- three clutter objects with independently sampled pose, scale, and colour;
- a bounded inspection-area occluder that can be enabled or disabled.

The existing RTX CameraSensor remains the image source, so generated crops use
the same resolution and foreground detector as the running factory. Camera
translation uses Isaac's camera-axis-aware `ViewportManager`; Replicator's
generic pose helper was rejected during validation because it resets this
existing sensor to a different axis convention. Replicator still applies the
object, material, illumination, clutter, occlusion, and focal randomization.

## Reproduce and audit

Run the accepted A/B smoke gate from the repository root:

```bash
infra/isaac-sim/replicator-smoke.sh
```

The wrapper safely stops the streamed Isaac container, captures two independent
runs with seed `2026`, audits both, and restores production model mode through
an exit trap. Generated images and manifests remain ignored.

Every manifest row carries its source group and split, source SHA-256, crop box,
crop and full-frame hashes, camera-config hash, Replicator extension version,
seed, view index, complete randomization plan, and a SHA-256 of the canonical
plan JSON. `src/audit_replicator_dataset.py` recomputes these hashes, rejects
test rows, incomplete view sets, duplicate IDs, missing files, insufficient
variation, and seeded plan drift.

## Accepted smoke evidence

The accepted run generated 18 images: one independent source for each of the
six routing labels and three views per source. The audit reported:

| gate | result |
|---|---:|
| source groups / rows | 6 / 18 |
| rows per label | 3 for all 6 labels |
| test rows | 0 |
| duplicate IDs | 0 |
| missing or hash-mismatched files | 0 |
| seeded plan matches | 18 / 18 |
| object rotation span | 335.071 degrees |
| object size span | 0.204 |
| camera height span | 0.116 m |
| focal-length span | 0.323 mm in the public RtxCamera scale |
| key / dome intensity spans | 3345.829 / 256.444 |
| occluded / unoccluded frames | 16 / 2 |

The full-frame contact sheet was inspected for all 18 renders. It demonstrates
meaningful camera, pose, lighting, material, clutter, and occlusion variation;
all 18 detector crops also contain the intended inspection object. The reviewed
sheet is committed at
[`docs/evidence/p13-replicator-visual-qa.jpg`](../evidence/p13-replicator-visual-qa.jpg),
and the complete machine-readable gate is
[`docs/evidence/p13-replicator-smoke-audit.json`](../evidence/p13-replicator-smoke-audit.json).

## Evidence boundary

The same seed reproduced all 18 source/view keys and all 18 parameter-plan
hashes exactly. It did **not** reproduce crop or full-frame pixels byte for
byte: both exact-pixel counts were 0/18. RTX rendering and sensor capture are
not claimed to be bit deterministic. Reproducibility here means the selected
sources, labels, transforms, randomization parameters, and provenance are
exact and machine-verifiable; each produced image is independently hashed.

This smoke set is visual and pipeline evidence, not training data volume and
not a model-accuracy claim. P1.4 owns the frozen evaluation gate and any future
candidate training. Production remains on `runs/sim_bin_adapt_v2/best.pt` until
a candidate actually passes that gate.
