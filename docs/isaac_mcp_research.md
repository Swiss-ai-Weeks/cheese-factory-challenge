# Isaac Sim MCP research

Research date: 2026-09-16. Installed simulator:
`6.1.0-rc.26+release.49347.2d230af4.glunknown` (container tag `6.1.0`).

## MCP setup and health

The server was installed from NVIDIA's official
[`isaacsim_mcp` README](https://github.com/NVIDIA-Omniverse/kit-usd-agents/blob/main/source/mcp/isaacsim_mcp/README.md).
The source checkout is external to this repository. The local endpoint is
`http://127.0.0.1:9904/mcp`; it answers MCP `initialize` using protocol
`2025-11-25` and identifies itself as `NeMo Agent Toolkit MCP 1.30.0`.

The server is registered for Codex as:

```bash
codex mcp add isaac-sim-mcp --url http://localhost:9904/mcp
codex mcp list
```

Restart the Codex session after first registration so native MCP tools appear.
No API key is stored in this repository. The server's local index searches work
without a key; an optional `NVIDIA_API_KEY` must be supplied only through the
runtime environment if NVIDIA-hosted intelligence is desired.

The required tools were listed and exercised:

- `get_isaac_sim_instructions`
- `search_isaac_sim_extensions`
- `get_isaac_sim_extension_details`
- `search_isaac_sim_code_examples`
- `search_isaac_sim_settings`

`get_isaac_sim_instructions` returned 43 instruction groups, including robot
simulation, sensors, physics, Python scripting, and synthetic data.

## Useful queries and findings

| Area | MCP query | Result used |
|---|---|---|
| manipulator | `Franka manipulator parallel gripper articulation` | Franka is configured in maintained `isaacsim.robot_motion.examples`; its joint gripper uses `panda_finger_joint1/2`. |
| motion | `RMPflow Lula inverse kinematics collision aware motion generation` | Selected `isaacsim.robot_motion.cumotion.RmpFlowController`; Lula IK alone reports no collision-avoidance support. |
| pick/place | `pick place controller Franka gripper grasp lift release timeout` | NVIDIA's callback-free `PickPlaceTask` supplies bounded phases and grasp/lift/place validation. |
| camera | `RGB depth camera intrinsics pixel world projection` | Selected `isaacsim.sensors.experimental.rtx.CameraSensor`, with `rgb` and `distance_to_image_plane`. Resolution ordering is `(height, width)` and buffers are Warp arrays. |
| projection | `camera intrinsics pixel to world depth` | The experimental camera exposes authoring optics but no high-level pixel-to-world helper; this project uses calibrated pinhole ray/plane intersection and records depth for diagnostics. |
| conveyor | `conveyor belt create conveyor physics` | `isaacsim.asset.gen.conveyor.create_conveyor_belt` exists. The deterministic demo instead kinematically advances one rigid object and stops at the inspection line, reducing intercept timing risk. |
| spawning | `replicator scripted spawning domain randomization` | Replicator extensions are available; deterministic scripted spawning is used so evaluation IDs, seed, and ordering are reproducible. |
| stepping | `physics stepping timeline SimulationApp headless` | Physics/app updates use experimental app utilities; the reproducible container command injects the coroutine with Kit `--exec` so the full-streaming app owns its event loop. |
| settings | `RTX sensor Hydra time render loop physics stepping` | `/rtx/rtxsensor/useHydraTimeAlways=True` is relevant to RTX timestamps; render-loop rate limit settings were found. |
| exact error | `ModuleNotFoundError isaacsim.robot_motion.examples enable extension Python import` | The extension exists but is not enabled by the base Python experience. It must be enabled through Kit's extension manager before import. |

`search_isaac_sim_code_examples` returned no indexed examples for the tested
Franka, ParallelGripper, Camera RGB, RMPflow, Lula, conveyor, Replicator, and
SimulationApp queries. Consequently, the exact 6.1 source and tests installed in
the NVIDIA container were inspected. In particular, NVIDIA's
`test_pick_place_interactive.py`, `pick_place_task.py`, camera sensor tests, and
manipulation robot configuration were used as the compatibility authority.

## Extension details retrieved and selected

Full details were retrieved for all of the following. Deprecated entries were
researched but not selected for new code.

| Extension | Version | Decision |
|---|---:|---|
| `isaacsim.sensors.experimental.rtx` | 1.9.0 | selected for RGB/depth |
| `isaacsim.robot_motion.experimental.motion_generation` | 7.1.2 | selected state/setpoint types |
| `isaacsim.robot_motion.cumotion` | 1.4.0 | selected collision-aware RMPflow |
| `isaacsim.robot_motion.examples` | 0.2.6 | selected maintained Franka task/controller |
| `isaacsim.asset.gen.conveyor` | 1.2.4 | researched; deterministic kinematic belt chosen |
| `isaacsim.robot_motion.motion_generation` | 8.2.9 | legacy, not selected for new control |
| `isaacsim.robot.manipulators` | 3.4.5 | deprecated in 6.x, not selected |
| `isaacsim.sensors.camera` | 1.7.13 | deprecated in 6.x, not selected |
| `isaacsim.core.api` | 5.3.2 | deprecated, avoided in factory modules |

The Franka asset selected by the maintained example is
`/Isaac/Robots_Multiphysics/FrankaRobotics/FrankaPanda/franka/franka.usda`.
The configured tool frame is `panda_hand`, grasp orientation is WXYZ
`(0, 1, 0, 0)`, and the controller-to-grasp offset is 0.1034 m along Z.

## Compatibility notes

- Enable `isaacsim.robot_motion.examples` before importing it in the Python app.
- RTX sensors need bounded warm-up; `CameraSensor.has_data()` is checked.
- `CameraSensor.get_data()` may return CUDA Warp arrays; convert with `.numpy()`.
- The selected controller disables the active object's planning collision only
  during close-contact phases, keeps the remaining scene collision-aware, checks
  that the cube was grasped/lifted, and enforces per-phase timeouts.
- Factory pick setpoints are overridden with the calibrated camera estimate. The
  rigid object's ground-truth pose is read only to report localization error and
  validate evaluation—not to command the pick or classify it.
- The 6.1 full-streaming app is required on this workstation for reliable RTX
  render-product updates. A disposable headless run exits at the process
  boundary after synchronously writing results because live stage teardown can
  hang in the streaming/debug-draw extension.
