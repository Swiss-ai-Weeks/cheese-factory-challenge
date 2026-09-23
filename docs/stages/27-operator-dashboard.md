# Stage 27 — live operator dashboard

The canonical streamed factory now has a viewport-attached dashboard inspired
by the perception branch's video HUD. Unlike the video compositor in
`sim/make_video.py`, these widgets update inside the running Isaac application.

## What the panels mean

- **Inspection / capture:** the exact detected crop sent to the classifier,
  resized for display only. This is a captured observation, not a live video
  feed. It clears when the next item enters the line.
- **Decision:** predicted type, destination and learned bin confidence. Model
  mode also shows the returned top three fine-type probabilities and inference
  latency. Fine-type probabilities and bin confidence come from different
  heads; they should not be interpreted as the same score.
- **Showcase / development:** permanently labeled as scripted routing or a
  pixel proxy. Neither mode displays fabricated learned confidence bars or
  model inference timing. The observation-age value is labeled separately.
- **Completed actions:** five output counts increment only when the controller
  reports a completed placement. The reject count includes simulated reject
  and quarantine diversions. These are action counts, not classification
  accuracy, bin occupancy measurements, or independent placement verification.
- **Robot:** the actual controller phase, item-bound authorization state and
  accumulated failure count. Model disagreement remains a visible safe hold.
- **Session:** processed / total count, current item and separately labeled
  scenario truth. Scenario truth is not used as the model prediction.

The central manipulation area stays open. The operator overview camera is
reframed to keep the cell above the lower information panels. Only the
overview camera and viewport grid presentation change; the inspection camera,
scene collision geometry, cuMotion obstacles and controller remain unchanged.

## Editing and running

`sim/factory/hud.py` owns the palette, panel layout, display model and bounded
image upload. `sim/factory/run_factory.py` supplies runtime events; keep motion
and decision policy out of the HUD. Colors are packed ABGR for `omni.ui`.
The dashboard targets the existing desktop stream (1280×720 or larger).

Use `infra/isaac-sim/factory-demo.sh showcase` for scripted physical routing or
`infra/isaac-sim/factory-demo.sh model` for the trained classifier. The dashboard
is included in the streamed Isaac viewport automatically. Restart the runtime
after editing its Python module. In the GUI, the viewport's visibility menu
can restore the editor grid when needed.

## Validation

- All 121 pure-Python tests pass, including separation of learned/scripted
  information, cross-model disagreement, completed-action accounting and
  bounded progress. Compilation and `git diff --check` pass.
- A three-object live showcase run completed all three placements. The HUD
  showed two hard-bin placements and one semi-hard-bin placement, matching the
  controller records, with scripted scores suppressed.
- A three-object live trained-model run completed two placements and diverted
  one item after type/route disagreement. The HUD showed actual crop pixels,
  top-three fine-type scores, route confidence, inference time and a safe hold.
  This is UI/integration smoke coverage, not a new classifier evaluation.
- The running 1920×1080 remote desktop view was visually inspected for readable
  text, unobstructed robot motion, camera framing and completed-action counts.
  Generated screenshots and smoke results remain under ignored `outputs/` or
  in the operator's local artifacts directory. No datasets or weights changed.
- The existing browser sometimes needs a fresh connection after an Isaac
  restart; recreating only the `remote-desktop` service restored the stream
  during the check. The scene's blue gridded ground asset is unchanged.
