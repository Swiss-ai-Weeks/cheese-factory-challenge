# Stage 15 — live visual and physics acceptance

## Problem

The prior stream devoted roughly half of the browser image to Isaac editor panels.
Automated checks also did not prove that the composed camera, HUD, receiver clearances,
and robot motion remained understandable together for an entire showcase.

## Resolution

The factory now enters a bounded presentation workspace after reset. It hides only the
known editor panes (`Stage`, `Layer`, render settings, robot inspector, properties,
console, and content browser); it leaves the viewport, simulator controls, performance
overlay, and factory HUD visible. Missing panes are ignored, so this remains safe across
minor Kit layout differences.

The expanded overview presents the physical arm, guarded inspection portal, conveyor,
five labelled cheese receivers, red reject receiver, safety boundary, and live decision
story in one view. This is materially more complete as a live factory presentation than
the perception branch's standalone render/training scripts: it adds a persistent
operator HUD, runtime identity, physical action states, receiver labels, and safe-reject
evidence.

## Saved evidence

- [active physical pick](../evidence/p06-showcase-active.png) — item 2, state
  `ROBOT · APPROACH PICK`, scripted-routing evidence boundary visible
- [completed run](../evidence/p06-showcase-complete.png) — `11/11`, safe reject and
  zero faults visible
- [machine-readable records](../evidence/p06-showcase-results.json) — per-item state
  histories, destinations, timings, localization, and outcomes

SHA-256:

- active image: `8e2e1ad4fc7ce8054b352de71978aab322c755d80a84b6d2f18cefcf3b693e18`
- completion image: `84584573adb600a59a64b3fdcc4eb6606750214d178790e0a0c93ef899139c45`
- results JSON: `69baf2c3a7589b4579277d6b2f9bb4d3a22f1d0c03d9eecc35dcbab0f29d1bd3`

## Acceptance result

The canonical streamed `judge-showcase` ran on repository base commit
`1a21e47996aac47ef841588a89126851792f3497` with the exact presentation changes in this
stage's diff. It completed 11 scenario items: ten physical cheese picks across all five
receivers, one fail-safe non-cheese rejection, one empty-belt interval, and no reported
faults or timeouts. Camera detection/localization and attempted manipulation both
reported 100% success in this bounded simulated scenario; mean cycle time was 18.41 s.

These are scripted-routing physical-integration results, not trained classifier accuracy
and not real-world performance. The HUD and JSON explicitly record
`showcase_ground_truth_routing=true` and `trained_model=false`.

Automated verification also passed `git diff --check`, Python compilation, the full test
suite, and a post-commit exact-SHA recovery smoke.
