# Cheese Factory developer handoff

This document is the authoritative starting point for a human developer taking
over the project. It describes what is implemented, what is generated, where
to make changes, and which claims are supported by evidence.

## Start here

1. Read [Known issues](KNOWN_ISSUES.md). The current demo is not a finished
   autonomous product.
2. Read [Isaac Sim for Unity developers](ISAAC_SIM_FOR_UNITY_DEVELOPERS.md).
3. Connect to the workstation and run the preflight in
   [Operations and Git](OPERATIONS_AND_GIT.md).
4. Run `infra/isaac-sim/factory-demo.sh showcase`, inspect the Stage tree and
   then run `infra/isaac-sim/factory-demo.sh model` to see the honest model path.
5. Create a feature branch before changing anything.

## What the repository contains

The repository contains three related but distinct bodies of work:

1. **Perception pipeline** — dataset ingestion, normalization, synthetic
   rendering, ConvNeXt training, inference and candidate evaluation.
2. **Earlier manipulation experiments** — a plate-carrying FR3 task, PPO
   experiments and prerecorded videos under `sim/pick_*`, `sim/train_pick.py`
   and `sim/cheese_picking.mp4`.
3. **Maintained live factory harness** — the current camera-to-control demo
   under `sim/factory/`, operated through `infra/isaac-sim/`.

Do not treat those three implementations as one unified controller. The live
factory uses NVIDIA's experimental `PickPlaceTask`; it does not execute the PPO
checkpoint from the earlier manipulation experiments.

## Current evidence boundary

| Runtime mode | Route source | What it proves | Important limitation |
|---|---|---|---|
| `showcase` | known scenario label | rendered-camera foreground localization and repeatable manipulation | routing is scripted; it is not model accuracy |
| `model` | fine-type and direct-bin PyTorch models served by `sim/sort_server.py` | actual current perception behavior and fail-closed disagreement handling | accepted end-to-end result is only 4/11; see `docs/evaluation_results.md` |
| `development` | rendered proxy color | software plumbing and diagnostics | no trained perception claim |

The HUD deliberately labels the active mode. Never remove or obscure that
label. Machine-readable results are written under `outputs/factory/` and are
ignored by Git.

## Runtime data flow

```text
config.yaml + scene_layout.json
            |
            v
Isaac Sim Stage -----------------------------------------------+
  /World/InspectionCamera -> RGB/depth frame                    |
            |                                                   |
            v                                                   |
ForegroundDetector -> crop + pixel centroid                    |
            |                                                   |
            +-> showcase: scenario route                        |
            +-> model: HTTP -> sort_server.py -> two models     |
            +-> development: color proxy                        |
            |                                                   |
            v                                                   |
pixel_to_plane() -> world-space pick target                     |
            |                                                   |
            v                                                   |
PerceptionPickPlaceTask -> cuMotion/Isaac pick-place phases     |
            |                                                   |
            v                                                   |
physics updates + HUD + results.json + annotated camera frames -+
```

The pick XY is calculated from camera pixels. The controller intentionally
does not query the active object's ground-truth XY when capturing its setpoint.
The test for that boundary is in `tests/test_factory_controller.py`.

## Scene ownership: what a human should edit where

Isaac's **Stage** window is the equivalent of Unity's **Hierarchy**. The
**Property** window is the closest equivalent to Unity's **Inspector**.

| Scene content | Source of truth | Why |
|---|---|---|
| floor, safety markings, conveyor decoration, inspection portal, work lights, beacon, overview camera | `sim/factory/scene_layout.json` | static presentation geometry, now declarative and human-readable |
| belt speed, spawn/pick positions, calibrated inspection camera, bin centres, robot path, test sequence | `sim/factory/config.yaml` | runtime and evaluation contract |
| conveyor collision lane | `sim/factory/scene.py` | physics-critical runtime primitive |
| Franka and pick-place task | `sim/factory/controller.py` plus NVIDIA asset/task | NVIDIA articulation and motion-generation API |
| receivers and label cards | `sim/factory/scene.py::_create_bin` plus `config.yaml` | repeated runtime geometry tied to routing names |
| active proxy or textured cheese carrier | `sim/factory/scene.py` | replaced for each evaluation object |
| HUD | `sim/factory/hud.py` | runtime UI, not a USD scene asset |

The static layout file creates stable paths such as
`/World/Factory/Inspection/left_post`. Each prim carries custom metadata:

- `factory_role` explains why it exists;
- `factory_source` identifies the layout file;
- `factory_collision` says whether the declaration requests static collision.

To test a separate layout without editing the checked-in file:

```bash
cp sim/factory/scene_layout.json /tmp/my-layout.json
export CHEESE_FACTORY_SCENE_LAYOUT=/tmp/my-layout.json
infra/isaac-sim/factory-demo.sh recover showcase
```

The override must be visible inside the container. A path under the repository
is safest because the repository is mounted at `/workspace`.

### Why not save arbitrary live Stage edits?

The application creates a fresh runtime stage when it launches. A transform
changed only in the Stage/Property UI changes that in-memory stage; it does not
update `scene_layout.json` and disappears after restart. Use the UI for visual
experimentation, then copy the accepted position/rotation/scale values into the
layout or runtime config and rerun tests. A future improvement could replace
the JSON layout with a referenced `.usda` asset and a separate runtime override
layer, but that conversion has not been validated yet.

## Code map

### Maintained factory (`sim/factory/`)

| File | Responsibility |
|---|---|
| `config.yaml` | machine-independent runtime values; despite the extension it is strict JSON |
| `config.py` | load and validate runtime configuration |
| `scene_layout.json` | human-editable static scene declarations |
| `scene_layout.py` | pure-Python layout loader and validation |
| `scene.py` | turn config/layout into USD prims; own camera and active-object access |
| `controller.py` | wrap NVIDIA `PickPlaceTask`; accept camera-derived targets |
| `geometry.py` | project image pixels to a world plane |
| `perception.py` | foreground detector, sorter adapters, showcase/development paths and annotations |
| `timing.py` | observation identity, HTTP correlation headers and fail-closed decision validation |
| `state_machine.py` | legal lifecycle and timeout transitions |
| `run_factory.py` | orchestrate the full scenario and write evidence |
| `kit_entry.py` | entry point executed inside Isaac Sim after extensions load |
| `hud.py` | operator window and evidence labels |
| `runtime_status.py` | atomic health/status identity used by Docker checks |
| `control_server.py` | loopback-only status/audit API and token-gated allowlisted control plane |
| `capture_domain.py` | canonical-camera dataset capture |
| `capture_replicator.py` | seeded Replicator capture with domain randomization |
| `replicator_plan.py` | pure randomization-plan generation |
| `capture_filters.py` | fail-closed split/bin selection for captures |
| `layout.py` | 2-D belt/receiver footprints and overlap checks |

The camera-to-control boundary is explicit. `run_factory.py` creates an
`ObservationContext` only after a camera crop exists, then passes it to the sorter. In
model mode, `perception.py` sends the item ID, sequence number, UUID request ID, capture
time and frame hash to `sort_server.py`; the server echoes that identity with its receive
and decision timestamps. `timing.py` validates the response once after inference and
again immediately before actuation. A missing, malformed, mismatched, future-dated or
expired response enters recovery, leaves the arm unauthorized and diverts the item to
the simulated reject path. The contract prevents accidental stale-response actuation;
it is not authentication against a malicious local service.

### Perception (`src/`)

| File | Responsibility |
|---|---|
| `download_datasets.py` | idempotent public-dataset download |
| `extract_food_recognition_hf.py` | convert Food Recognition shards |
| `normalize.py` | canonical records, deduplication and group-aware splits |
| `cutouts.py` | create polygon/alpha object cutouts |
| `render_manifest.py` | build simulation render work list and group splits |
| `dataset.py` | PyTorch dataset, transforms and label spaces |
| `train.py` | ConvNeXt training, class/domain balancing and validation |
| `predict.py` | checkpoint loading, result contracts and hybrid safety policy |
| `export.py` | ONNX export |
| `evaluate_checkpoint.py` | checkpoint metrics and confidence coverage |
| `prepare_camera_sources.py` | reviewed target-camera source selection |
| `build_camera_dataset.py` | group-balanced camera-domain manifest |
| `audit_camera_dataset.py` | hashes, leakage and coverage checks |
| `audit_replicator_dataset.py` | Replicator provenance and replay audit |
| `build_p14_candidate_manifest.py` | construct training-only candidate manifest |
| `check_candidate_gate.py` | frozen multi-suite model-promotion gate |

### Runtime and infrastructure (`infra/isaac-sim/`)

| File | Responsibility |
|---|---|
| `factory-demo.sh` | supported human operator command: start, replay, recover, status, stop |
| `run-gui.sh` | launch streamed GUI and optional model service |
| `run-headless.sh` | bounded headless integration run |
| `run-evaluation.sh` | reproducible evaluation wrapper |
| `check-environment.sh` | GPU, Docker, containers, model and disk preflight |
| `docker-compose.yml` | Isaac, web viewer and browser desktop services |
| `capture-domain.sh` / `capture-all-domain.sh` | canonical camera captures |
| `capture-replicator.sh` / `replicator-smoke.sh` | randomized synthetic captures and audit |
| `remote-desktop/` | Chromium plus noVNC desktop wrapper |
| `web-viewer/` | NVIDIA WebRTC viewer served locally |

### Older experiment code

`sim/render_belt.py`, `sim/sorting_line.py`, `sim/pick_cell.py`,
`sim/train_pick.py`, `sim/pick_line.py`, `sim/make_video.py`, and the French
bench scripts are retained for provenance and earlier results. Do not quietly
modify them assuming the maintained `sim/factory/` runtime uses them.

## Generated and external files

The repository intentionally does not contain the large raw datasets, rendered
images or 350 MB checkpoints. On the workstation they live under:

- `data/` — downloaded and processed datasets;
- `sim/out/` — older synthetic renders;
- `outputs/factory/` — current runtime evidence and logs;
- `runs/*/best.pt` — ignored PyTorch checkpoints;
- Docker cache directories under `/home/ubuntu/docker/isaac-sim/`.

Never commit a checkpoint or dataset unless the repository has an explicit Git
LFS policy and the license/provenance has been reviewed.

## Safe change recipes

### Change the visible factory

1. Create a feature branch.
2. Copy `scene_layout.json` and use `CHEESE_FACTORY_SCENE_LAYOUT` for iteration.
3. Start showcase, select the corresponding `/World/Factory/...` prim in Stage,
   and experiment with its transform in Property.
4. Copy accepted values into the checked-in JSON.
5. Run `pytest`, `git diff --check`, and a complete 11-object showcase.
6. Inspect the viewport, annotated frames and `results.json` before committing.

### Change a bin or runtime camera

Edit `config.yaml`, not `scene_layout.json`. Run the footprint tests after any
bin movement. A camera change invalidates the ROI, pixel projection and
camera-domain training assumptions; it requires capture/evaluation again.

### Add collision to visible equipment

Setting `collision` to `static` authors a PhysX collider, but that alone does
not guarantee cuMotion will avoid it. Use `collision_path` for an aligned proxy
and `planning_obstacle: true` for a startup registration check. Stage 26's
gantry is the maintained example; any geometry change still requires route
revalidation.

### Change perception

Do not select a model from one attractive aggregate metric. Preserve group
splits and use the frozen gate in `config/p14_candidate_gate.json`. The locked
test split is only for a candidate that passes validation criteria.

## Minimum acceptance before merging

```bash
python -m pytest -q
python -m compileall -q sim src tests
git diff --check
infra/isaac-sim/factory-demo.sh preflight
infra/isaac-sim/run-evaluation.sh showcase 11
```

For perception changes, also run the relevant dataset audits, candidate gate
and honest model evaluation. For visual or physics changes, screenshots and
machine metrics are both required; unit tests alone are insufficient.

## Definition of a credible autonomous factory

The project should not be called complete until the live system demonstrates:

- camera pixels, not scenario labels, determine the route;
- collision-aware robot motion respects all visible equipment;
- grasp and placement use physical contacts rather than teleportation;
- model, robot and scene identities are recorded with each run;
- an unseen, group-separated evaluation set produces acceptable per-bin
  performance, not just aggregate accuracy;
- every public metric can be regenerated from committed code and documented
  external artifacts.
