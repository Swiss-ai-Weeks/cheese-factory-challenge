# Known issues and technical debt

This list is intentionally blunt. It prevents a new developer from interpreting
an attractive streamed demo as stronger evidence than it is.

## P0 — blocks a credible robotics claim

- [x] **Inspection portal is outside the validated robot routes.** The portal
  posts were widened to x=-1.05/1.80 m and its beam raised to z=1.75 m. All three
  structural members have PhysX collision and are named safety obstacles in
  `scene_layout.json`. Aligned invisible collision proxies prevent cuMotion's
  tracked-world binding from disturbing RTX presentation geometry. Startup now
  fails unless cuMotion discovers and enables every declared proxy. The full
  11-item showcase route suite is the acceptance gate; see
  `docs/stages/26-gantry-safety.md`.
- [ ] **Production perception is not hackathon-ready.** The accepted model run
  achieved 4/11 complete outcomes and 3/10 correct cheese-bin placements.
  Showcase's 11/11 result uses scenario-provided routing and cannot be used as
  evidence of model accuracy.
- [ ] **The visible receiver walls are not colliders.** Only the reject floor is
  explicitly collidable. This was done to keep the current Franka descent path
  feasible, but it means the visible bins are not physically faithful. Bin
  geometry and motion paths must be redesigned together.

## P1 — important engineering limitations

- [ ] **The rendered cheese and manipulated collision object differ in model
  mode.** The photographic cutout/plate is a visual carrier while the hidden
  task cube remains the controller's dynamic object. This is valid for testing
  perception-to-target wiring, but it is not a faithful cheese grasp.
- [ ] **Belt motion is scripted pose interpolation.** The active object is moved
  along the belt by code; this is not a motorized conveyor with contact-driven
  transport.
- [x] **Repository-owned cuMotion safety-obstacle registry.** Declarative
  `planning_obstacle` entries are validated against static collision and checked
  against NVIDIA's tracked collision world during controller initialization.
- [ ] **Static scene edits made only in the GUI are temporary.** The new
  declarative layout makes the source obvious, but a referenced `.usda` plus an
  editable override layer would provide a more native DCC-style authoring flow.
- [ ] **Model weights are workstation-local.** Git cannot reproduce a production
  run without restoring or rebuilding both ignored checkpoints. Add an artifact
  registry with immutable hashes and documented access policy.

## P2 — maintainability and presentation

- [ ] `config.yaml` contains JSON, not YAML. Rename it to `.json` in a deliberate
  compatibility change or introduce a real YAML dependency and migration.
- [ ] Receiver construction remains procedural in `_create_bin`. Convert it to
  a referenced USD asset or declarative module once its physics design is fixed.
- [ ] Several older experiment scripts and French identifiers remain. They are
  documented for provenance but increase cognitive load.
- [ ] The repository has extensive stage reports but previously lacked one
  human takeover path. `DEVELOPER_HANDOFF.md` is now that entry point; stale or
  conflicting operational notes should be consolidated over time.

## Resolved for handoff

- [x] Static presentation objects are no longer buried as numeric literals in
  `scene.py`; they are declared in validated `scene_layout.json`.
- [x] Static prims have stable, meaningful Stage paths and purpose/source/
  collision metadata.
- [x] Unity-to-Isaac concepts, workstation operations, Git workflow and the
  evidence boundary are documented.
