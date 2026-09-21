# P2.3 — Judge-ready evidence

## Deliverables

- `docs/JUDGE_GUIDE.md`: 90-second pitch, live walkthrough, parent-branch comparison,
  differentiators and honest evidence boundaries.
- `docs/ARCHITECTURE.md`: perception-to-action architecture, control/MCP separation,
  repository ownership and mode semantics.
- `docs/MODEL_CARD.md`: model provenance, intended use, data, accepted metrics,
  limitations, controls and promotion policy.
- `docs/evidence/p23-live-showcase.mp4`: continuous 75-second H.264 capture of the live
  WebRTC/browser experience at 1280×720 and 20 fps.
- `docs/evidence/p23-live-showcase-preview.jpg`: five evenly spaced frames used for
  visual QA and a GitHub-readable preview.

## Capture provenance

The capture was produced from executable commit
`7a5da41928ab2434b71e1a785c95eab3eac7fb77` with scenario
`judge-showcase-replay`. `infra/isaac-sim/capture-demo.sh` rebuilds the browser desktop,
replays the factory and records its X11 display directly with FFmpeg. It does not splice
still images or fabricate robot motion.

Artifacts:

- MP4 SHA-256: `ddf4239bdd50985f1c5ddb7ae2dfd75001033f56d0895261d720b062b95922a4`
- preview SHA-256: `52aa3e7194d542d864bf7bd9346d4e655fa189412dd9f77876ee76ba4bf659bd`

The five-frame preview was visually inspected. It shows the same overview camera and
HUD across the continuous sequence, with the Franka moving through approach, grasp/
carry and receiver-side poses. The video retains the on-screen scripted-routing evidence
label so it cannot be mistaken for trained-model accuracy.

## Branch comparison basis

Git ancestry checks confirmed that both `work/perception` at `fe70eab` and
`work/isaac-factory` at `0f144ae` are ancestors of the integration line. The comparison
in the judge guide describes durable repository capabilities rather than contributor
credit or subjective code quality.

## Reproduce

```bash
infra/isaac-sim/capture-demo.sh 75 docs/evidence/p23-live-showcase.mp4
```

This requires the same configured RTX/Docker baseline as the one-command live demo. The
capture tool bounds duration to 15–180 seconds and writes only to an explicit
repository-relative MP4 path.
