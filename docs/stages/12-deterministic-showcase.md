# Stage 12 — Deterministic physical showcase

## Problem

The arm appeared idle in production because a weak trained prediction correctly blocks
unsafe motion. The older pixel-color development classifier was also coupled to scene
lighting, so a visual redesign could change its route. Neither behavior makes a reliable
judge demonstration, and silently treating scripted routing as AI would be misleading.

## Resolution

- Added a first-class `showcase` mode to GUI, headless evaluation and the allowlisted
  control gateway.
- Kept rendered foreground detection and calibrated camera-to-world localization in the
  loop; only the route is supplied from the known scripted scenario.
- Kept `model` and `development` behavior unchanged.
- Added unambiguous machine-readable evidence fields:
  `showcase_ground_truth_routing: true` and `trained_model: false`.
- Added strict tests for all five routes, rejection, unknown labels and safe control
  allowlisting.

## Full Isaac acceptance

The 11-object sequence ran in the redesigned canonical cell:

- 11/11 detections and scripted end-to-end outcomes;
- 10/10 physical cheese pick/place cycles across all five receivers;
- 1/1 foreign object rejected without a pick;
- 1/1 empty interval produced no action;
- no controller failure or timeout;
- mean cycle time 15.61 seconds.

The complete ignored machine-readable artifact is
`outputs/factory/showcase-acceptance.json` on the evaluation workstation.

## Verification

- Full repository tests: **40 passed**.
- Python compilation, Bash syntax and `git diff --check`: passed.
- The evaluation launcher restored the complete healthy WebRTC stack after the run.

This stage proves repeatable physical behavior. It deliberately does not improve or
restate the production classifier's measured accuracy.
