# Stage 11 — Industrial live factory cell

## Problem

The canonical WebRTC scene was still the minimal Isaac-factory layout: a wide gray slab,
five open colored boxes and a flat reject pad. Two receivers physically overlapped the
rendered conveyor, the scene had no composed overview camera, and the stronger visual
assets from the perception branch were not represented in the live robot application.

## Resolution

- Replaced the wide slab with a compact detailed conveyor whose visible and collision
  footprints agree.
- Rebuilt all five destinations and rejection as open, labelled receiver stations with
  dark totes, stainless frames, replaceable feet and restrained color accents.
- Added an inspection portal, guarded pick zone, camera housing, work lights, conveyor
  slats and supports, safety beacon, bounded factory deck and floor markings.
- Added a dedicated overview camera and made it the streamed viewport while retaining a
  separate calibrated inspection camera for perception.
- Added a pure-Python footprint contract and tests proving 4 mm minimum clearance among
  the conveyor and all receiver footprints.

The implementation reuses the perception branch's label textures and USD material/
primitive toolkit inside the reliable modular factory runtime. It does not fork another
standalone scene.

## Verification

- Full repository tests: **37 passed**.
- Python compilation and `git diff --check`: passed.
- Layout tests: conveyor, five cheese receivers and reject receiver have no overlapping
  footprint and retain at least 4 mm of modeled air gap.
- Live WebRTC-equivalent 1920×1080 capture was inspected from the active
  `/World/OverviewCamera`; all stations, the conveyor, inspection portal and robot are
  visible and the old conveyor/bin intersections are gone.
- The inspection camera continued to detect and localize the test object. A physical
  pick/place also completed during scene iteration, demonstrating that the narrowed
  conveyor and receiver geometry preserve the maintained cuMotion path.

## Evidence boundary

This stage accepts scene composition and clearance only. The legacy pixel-color
development classifier changes under the improved lighting and is not used as an
accuracy claim. The next checklist item introduces an explicitly labelled deterministic
showcase path and validates every destination; trained-model mode remains separate.
