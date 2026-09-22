# Stage 26 — collision-aware inspection gantry

## Problem

The original inspection portal occupied the Franka's visible swept region but
was authored as presentation-only geometry. It had neither PhysX collision nor
an explicit repository-owned contract requiring cuMotion to track it. The
scripted arm could therefore visibly pass through a post.

## Safety change

- widened the post centres from x=0.31/0.69 m to x=-1.05/1.80 m, beyond
  Franka's nominal 0.855 m reach from its world origin;
- raised the crossbeam from z=1.13 m to z=1.75 m, with its lower face at
  z=1.7125 m;
- created identically aligned, invisible collision proxies for both posts and
  the crossbeam and applied `UsdPhysics.CollisionAPI` to those proxies;
- marked those exact proxy paths as `planning_obstacle` declarations;
- made controller initialization fail if any declared root is absent from
  cuMotion's collision world;
- kept the decorative guards, work lights and camera housing aligned with the
  redesigned gantry;
- reframed only the operator overview camera so both posts and the raised beam
  remain visible; the calibrated inspection camera and perception ROI are
  unchanged.

The visual/collision split is deliberate. Applying the world binding directly
to the rendered boxes caused the Isaac 6.1 RTX inspection camera to return an
occluded black frame. The proxies preserve one-to-one geometry and physical
contact while keeping planner-owned primitives out of the rendered image.

## Acceptance evidence

- `114 passed` in the pure-Python suite; compilation and `git diff --check`
  passed.
- Isaac reported all three `FACTORY_CUMOTION_OBSTACLES` proxy paths and their
  world bounds: posts outside x=-1.0225/1.7725 m and beam lower face z=1.7125 m.
- The complete showcase sequence finished 11/11: ten successful physical
  pick/place cycles covering every receiver twice, followed by one correctly
  inhibited non-cheese item. Detection, pick, correct-bin and end-to-end rates
  were all 1.0 in this scripted-routing integration run.
- The final 15-second operator capture was visually inspected during arm
  retreat. Both posts and the raised beam are separated from the robot; no link
  intersects the gantry. Generated evidence remains ignored under
  `outputs/gantry-reframe-2.mp4` (SHA-256
  `b8b7db3a496363b86034ca166c7446ca3cba9e349341d31e37c962be062726eb`).

These are physical-integration results for showcase mode, not trained-model
accuracy.
