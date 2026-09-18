# Hackathon readiness checklist

This is the acceptance plan for `codex/hackathon-integration`. Items are ordered by
delivery risk, not convenience. A checked item must have its own coherent commit,
reviewed diff, automated checks, and (where applicable) live Isaac Sim evidence.

The production branch must remain truthful: deterministic showcase routing and trained
perception are separate, visibly labelled modes. A presentation-friendly showcase is
not evidence of model accuracy.

## P0 — demo blockers

- [x] **P0.1 — Prove which application WebRTC is showing.** The current health check
  can report healthy when plain Isaac Sim is running without the factory entry point.
  Add a factory-owned runtime status/heartbeat, make service readiness depend on it,
  and ensure every restart path restores the canonical factory command.
  **Accept when:** the status identifies branch/commit, mode, scenario and phase; a
  plain Isaac process fails readiness; restart and recovery tests pass.
  **Completed:** commit documented in
  [`docs/stages/10-runtime-identity.md`](stages/10-runtime-identity.md).

- [x] **P0.2 — Replace the overlapping canonical layout.** The live `sim/factory`
  scene places two receiving bins inside the conveyor footprint and looks materially
  less complete than `sim/sorting_line.py`. Rebuild the canonical scene with a clear
  inspection zone, non-overlapping transfer/output lanes, industrial receiving
  stations, guards, safety markings, signs, lighting and a composed overview camera.
  Reuse the strongest assets from both parent branches rather than maintaining a third
  disconnected design.
  **Accept when:** geometry tests prove clearance; overview and inspection renders show
  every station clearly; no bin, robot or cheese intersects the conveyor at rest.
  **Completed:** commit documented in
  [`docs/stages/11-industrial-live-cell.md`](stages/11-industrial-live-cell.md).

- [ ] **P0.3 — Make robot work unmistakable.** In trained-model mode, rejected or
  uncertain decisions suppress most picks, so the arm appears idle even when control is
  functioning. Add an explicitly labelled deterministic `showcase` mode that exercises
  every destination plus rejection, while preserving the honest `model` mode.
  **Accept when:** a repeatable showcase visibly completes all intended pick/place
  cycles with no timeouts, and model mode remains fail-closed.

- [ ] **P0.4 — Add an operator-visible story.** The live view does not explain the
  current item, prediction, confidence, destination, robot state or totals. Add a clear
  factory HUD/scoreboard and legible station labels so judges can understand the full
  perception-to-action loop without reading terminal logs.
  **Accept when:** a WebRTC screenshot alone makes the current decision and accumulated
  outcome understandable, with the active mode visibly labelled.

- [ ] **P0.5 — Deliver reliable demo controls.** Starting, stopping, resetting and
  selecting scenarios currently depend on shell knowledge and can leave a healthy but
  wrong Isaac process behind. Provide safe idempotent commands/API controls, one-click
  scenario reset, and a recovery path that does not corrupt metrics.
  **Accept when:** a fresh operator can launch either mode, replay the showcase and
  recover the service using the runbook only.

- [ ] **P0.6 — Pass a live visual and physics acceptance run.** Automated tests did not
  catch the poor camera composition, overlapping bins or apparently idle robot. Add
  saved evidence from the canonical WebRTC-equivalent camera and inspect the full run.
  **Accept when:** the factory is visually stronger than the perception-branch demo,
  the arm completes its showcase, and the evidence records commit and configuration.

## P1 — integrated intelligence

- [ ] **P1.1 — Unify the perception contract.** Fine-type classification and direct-bin
  routing can disagree (for example, a blue-cheese type paired with `not_cheese`). Define
  one authoritative decision, preserve both raw outputs for diagnosis, and make policy
  precedence explicit and tested.
  **Accept when:** contradictory outputs cannot silently produce an unexplained action;
  unit and scenario tests cover every status and bin.

- [ ] **P1.2 — Match training data to the canonical camera.** The production classifier
  was trained on a different rendered domain; the tiny target validation set is heavily
  imbalanced and contains almost no blue cheese diversity. Use the final inspection
  camera and geometry to generate group-safe, independently varied data with balanced
  coverage and provenance.
  **Accept when:** manifests prove source/group separation, per-class coverage and no
  test leakage; generated data is not committed unless repository/LFS policy permits.

- [ ] **P1.3 — Use Omniverse Replicator as a real data engine.** Current Isaac rendering
  varies some parameters but underuses Replicator. Add reproducible randomization for
  pose, material, illumination, clutter, occlusion and camera calibration around the
  canonical scene, plus metadata for every render.
  **Accept when:** a seeded smoke batch reproduces; visual QA demonstrates meaningful
  variation; labels and transforms are machine-verifiable.

- [ ] **P1.4 — Retrain only behind fixed gates.** Earlier candidates consumed compute
  but did not beat the production checkpoint. Freeze validation before training and
  require improvements in both aggregate and rare-bin metrics without regressing reject
  safety or live-scenario behavior.
  **Accept when:** a candidate passes documented gates before promotion; otherwise the
  baseline stays in production and rejection is recorded honestly.

- [ ] **P1.5 — Harden camera-to-control timing.** Couple observations, decisions and
  actions with item IDs and timestamps so stale responses cannot move the arm for the
  wrong cheese. Expose timeout/fault states in telemetry and the HUD.
  **Accept when:** delayed, missing and malformed inference tests fail safely and a
  normal run has traceable end-to-end timing.

- [ ] **P1.6 — Expose safe agent/factory control.** Keep the documentation MCP's scope
  distinct from runtime control. Provide a narrow authenticated/read-only-by-default
  interface for status, reset, mode/scenario selection and approved actions, with an
  audit trail and emergency stop behavior.
  **Accept when:** unsafe or unknown commands are rejected and every accepted mutation
  is attributable and reversible by reset.

## P2 — hackathon delivery

- [ ] **P2.1 — One-command reproducibility.** Pin dependencies and model provenance,
  validate prerequisites, and supply one documented command for the live demo plus a
  fast preflight.
  **Accept when:** a clean workstation reaches a healthy canonical demo using the
  runbook without undocumented manual repair.

- [ ] **P2.2 — Publish honest end-to-end evaluation.** Separate classifier, routing,
  manipulation and full-loop metrics; include failure counts, sample sizes, seeds and
  confidence boundaries. Do not present synthetic or showcase results as real-world
  accuracy.
  **Accept when:** all reported numbers are regenerated from saved machine-readable
  artifacts tied to the tested commit.

- [ ] **P2.3 — Produce judge-ready evidence.** Create a short continuous demo capture,
  architecture diagram, feature comparison, model card and concise pitch explaining the
  industrial value and limitations.
  **Accept when:** the evidence shows camera input, decision, arm motion and destination
  in one understandable sequence and links to reproducible results.

- [ ] **P2.4 — Final regression and branch audit.** Confirm both parent histories remain
  contained, no generated datasets/checkpoints leaked into Git, no source branch was
  modified, and `codex/hackathon-integration` is visibly and functionally superior to
  `work/perception` for the live factory experience.
  **Accept when:** full tests, compile checks, shell checks, diff checks, service health,
  screenshot review and the feature matrix all pass on the remote commit.

## Required gate for every checklist commit

1. Inspect the complete diff and confirm it touches only the intended concern.
2. Run `git diff --check`, compilation and the relevant automated tests.
3. For scene, physics or UI changes, inspect rendered/live evidence—not logs alone.
4. State evidence limits; never invent metrics or mark an unobserved behavior complete.
5. Commit one coherent change, push it, and verify the remote SHA before starting the
   next item.
