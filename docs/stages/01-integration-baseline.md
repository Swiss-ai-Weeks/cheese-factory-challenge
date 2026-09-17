# Stage 1 — integration baseline

## Inputs

- Visual/perception base: `origin/work/perception` at `fe70eab`.
- Autonomous factory branch: `origin/work/isaac-factory` at `0f144ae`.
- Merge base: `6b482e4148f57784e6011149e7a84b1d1508e215`.
- Integration branch: `codex/hackathon-integration`.
- Installed target runtime: NVIDIA Isaac Sim 6.1 on the RTX PRO 6000 workstation.

Both contributor branches remain unchanged. The integration keeps their full
histories through a non-fast-forward merge.

## Capabilities retained from `work/perception`

- Trained 13-class `CheeseSorter` pipeline, grouped dataset split discipline,
  recorded offline metrics and model-serving API.
- Polished main conveyor, inspection station, six alternating output conveyors,
  diverters, signs, HUD and committed sorting-line video.
- Franka FR3 pick-and-place cell, deterministic reference state machine, PPO
  training environment, recorded training curves and committed arm video.
- Extensive reproducibility, results, failure and integration documentation.

## Capabilities retained from `work/isaac-factory`

- Isaac Sim 6.1 Docker/remote-GUI launch integration.
- Modular `sim/factory/` camera, perception, state-machine, controller, scene
  and evaluation implementation.
- Explicit development-versus-production classifier behavior and fail-closed
  production checkpoint handling.
- Machine-readable lifecycle/timing metrics and twelve pure-Python regression
  tests.

## Evidence boundaries

The robot implementations serve different purposes and must not be conflated:

1. The committed `sim/cheese_picking.mp4` is driven by the scripted reference
   state machine. This proves the cell is mechanically feasible.
2. The PPO policy is unfinished. Recorded results show gripping/carrying, but no
   completed deposit; this is experimental work, not the primary demo.
3. The `sim/factory/` development run uses rendered proxy-colour
   classification. It validates camera-to-controller integration and metrics,
   not trained cheese-model accuracy.
4. Real perception requires the `sim_type13` checkpoint and must report its
   own offline and scene-level evidence.

## Current local artifact state

At integration time, the RTX and H200 workstations do not contain:

- `runs/sim_type13/best.pt`;
- the approximately 14 GB local `data/` tree;
- the approximately 4.6 GB raw `sim/out/` renders.

The repository contains small result JSON/sidecar files and the following
committed demonstration evidence:

- `sim/cheese_sorting.mp4`;
- `sim/cheese_picking.mp4`;
- `vault/attachments/sorting-line.png`;
- `runs/pick_fr3/results.json`;
- `runs/pick_fr3_echec_bord/results.json`;
- `runs/sim_type13_plateau/results.json`.

Previously generated `sim/factory/` evaluation outputs were preserved outside
the Git worktree at
`/home/ubuntu/cheese-factory-artifacts/pre-integration-20260918`; they are not
committed evidence for this merge.

## Stage 1 acceptance

Stage 1 changes no Python implementation manually. It combines the two histories,
resolves the README by preserving both capability descriptions, documents the
baseline and runs the existing regression and compilation gates. Real checkpoint
recovery, runtime smoke testing, scene redesign and end-to-end validation are
separate later stages.
