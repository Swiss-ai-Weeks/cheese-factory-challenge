# Stage 13 — Live operator HUD

## Problem

The streamed viewport gave no explanation of whether the factory was running, why the
arm was idle, which decision had been made, or whether a result came from the trained
model. Terminal logs held that information, but judges watching WebRTC could not see it.

## Resolution

Added an Isaac Kit operator window that remains in the streamed viewport and follows the
camera-to-control loop. It displays:

- a permanent mode/evidence banner;
- scenario and current conveyor/robot phase;
- current item and evaluation ground truth;
- status, predicted type, confidence/evidence type and destination;
- processed, successful, safely rejected and failed totals.

The three mode banners are deliberately unambiguous:

- `MODEL · TRAINED PERCEPTION`;
- `DEVELOPMENT · PIXEL PROXY`;
- `SHOWCASE · SCRIPTED ROUTING (NOT MODEL ACCURACY)`.

Controller phase changes update the panel during approach, descent, grasp, lift, place,
release and retreat. Detection faults and fatal errors are also visible instead of
leaving a mysteriously idle arm.

## Verification

- Full repository tests: **43 passed**.
- Python compilation and `git diff --check`: passed.
- A live WebRTC-equivalent 1920×1080 screenshot was visually inspected after a physical
  showcase pick/place. The panel showed `RUN COMPLETE`, the item, `bin_hard`, `SCRIPTED`,
  and `1/1 processed · 1 successful · 0 safe rejects · 0 faults`.
- Pure tests require every runtime mode to have an evidence banner and prevent an unknown
  mode from being silently presented.

The screenshot is retained only in ignored evaluation outputs; it is not a classifier
metric or a committed binary artifact.
