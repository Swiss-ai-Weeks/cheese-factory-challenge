# Isaac Sim for a Unity developer

This guide assumes you understand Unity scenes, GameObjects, Transforms,
colliders, rigidbodies, prefabs and play mode, but are new to OpenUSD,
Omniverse Kit and Isaac Sim.

## The shortest mental model

Isaac Sim is an Omniverse Kit application. OpenUSD holds the scene, RTX renders
it, PhysX/Newton simulate it, and Isaac extensions add robot descriptions,
sensors, ROS interfaces, motion generation and synthetic-data tools.

| Unity concept | Isaac/OpenUSD concept | Important difference |
|---|---|---|
| Scene | USD Stage | a stage can compose many layers and referenced assets |
| Hierarchy | Stage window | entries are called prims |
| GameObject | prim | a prim has a path such as `/World/Franka` |
| Transform component | Xform/XformOps | transform operations are ordered and layerable |
| Inspector | Property window | properties may come from several USD layers |
| Prefab | reference/payload/variant | composition is a core USD feature |
| Component | applied API schema | collision and rigid-body behavior are separate APIs |
| MeshRenderer/material | UsdGeom + UsdShade | appearance does not imply collision |
| Rigidbody | RigidBodyAPI | must be authored explicitly |
| Collider | CollisionAPI | a visible mesh can have no collider |
| ArticulationBody | articulation + joints/drives | optimized reduced-coordinate robot chain |
| Play mode | timeline Play | simulation and many ROS/graph nodes update only while playing |
| C# MonoBehaviour | Python/C++ extension, OmniGraph node | logic often runs through Kit extensions or standalone Python |
| Editor tooling | Kit extensions | Isaac itself is assembled from extensions |

The most important rule is the same as Unity: **what you see is not necessarily
what collides**. Isaac adds a second distinction: an object may collide in PhysX
but still be absent from a motion planner's obstacle world.

## Coordinates and units

The project uses metres and a right-handed, Z-up world:

- +X: across the cell;
- +Y: along the conveyor toward the pick line;
- +Z: upward;
- rotations in the layout file are XYZ degrees;
- robot quaternions in config use `[w, x, y, z]`.

Unity normally uses metres, Y up and a left-handed convention. Do not copy Unity
positions or quaternions without converting them.

## OpenUSD essentials

### Prim paths

Every object has a stable path:

```text
/World
  /Factory
    /Conveyor
    /Inspection
  /Franka
  /Bins
  /InspectionCamera
```

Paths are identity, not just display names. Code frequently looks up a prim by
its path, so renaming `/World/InspectionCamera` without updating configuration
breaks the runtime.

### Layers and composition

A USD stage is the composed result of one or more layers. A layer can override
values from a weaker layer without modifying the original asset. References
and payloads are conceptually similar to nested prefabs, but USD composition is
more powerful and therefore easier to misunderstand.

The current factory creates an in-memory stage from Python. Static presentation
declarations live in `scene_layout.json`; runtime-critical objects are still
authored by code. UI edits to that anonymous stage are temporary.

### Schemas and APIs

A cube prim is only geometry until behavior schemas are applied. Typical APIs:

- `UsdPhysics.CollisionAPI` — participate as collision geometry;
- `UsdPhysics.RigidBodyAPI` — become a simulated rigid body;
- mass/material APIs — mass, density, friction and restitution;
- articulation root and joint schemas — robot kinematic chain;
- semantic labels — ground truth for synthetic data.

Visible geometry, PhysX collision and planner obstacles are separate concerns.
The inspection gantry demonstrates the safe pattern: human-readable rendered
prims plus aligned invisible collision proxies that are verified in cuMotion's
world during controller initialization.

## First tour of the running project

1. Connect and run `infra/isaac-sim/factory-demo.sh showcase`.
2. Open the noVNC URL from `OPERATIONS_AND_GIT.md`.
3. In Isaac, locate the **Stage** panel. Expand `/World`.
4. Select `/World/Factory/Inspection/left_post`.
5. In **Property**, inspect its transform and custom data. The custom data points
   back to `scene_layout.json`; its aligned collision proxy is under
   `/World/Factory/PlanningObstacles/Inspection`.
6. Select `/World/InspectionCamera`. This is the actual RTX perception camera.
   `/World/Factory/Inspection/camera_body` is only its decorative housing.
7. Select `/World/OverviewCamera`. This drives the operator viewport; changing
   it does not change model input.
8. Expand `/World/Franka` and inspect links/joints. Avoid editing articulation
   transforms while the timeline is running.
9. Press Stop before experimenting. Resume with Play. Remember that restarting
   the factory recreates the stage and discards unsaved live edits.

## How this project builds the Stage

Launch sequence:

1. `factory-demo.sh` chooses a bounded runtime mode.
2. `run-gui.sh` prepares identities, services and Docker environment.
3. Isaac executes `sim/factory/kit_entry.py` after Kit starts.
4. `run_factory.py` creates `CheeseFactorySample`.
5. `scene.py` asks NVIDIA's task to add the Franka, then authors the factory.
6. `scene_layout.py` validates `scene_layout.json` before any static prim is made.
7. `scene.py` creates belt collision, receivers, inspection camera and model
   carrier, then wraps the camera in `CameraSensor`.
8. The timeline plays and `run_factory.py` advances the scenario.

## Editing the static environment

Use the Stage and Property panels to discover and prototype values, but commit
changes through `scene_layout.json`.

Each box declaration contains:

```json
{
  "path": "/World/Factory/Inspection/left_post",
  "collision_path": "/World/Factory/PlanningObstacles/Inspection/left_post",
  "size": [0.055, 0.075, 1.71],
  "position": [-1.05, 0.08, 0.845],
  "material": "steel",
  "collision": "static",
  "planning_obstacle": true,
  "role": "wide inspection gantry post beyond Franka nominal reach"
}
```

- `path` determines its Stage hierarchy location;
- `size` is full XYZ size in metres;
- `position` is the centre in world metres;
- optional `rotation_xyz_deg` defaults to zero;
- `material` refers to the material table in the same file;
- `collision` can be `none` or `static`;
- optional `collision_path` creates an aligned invisible collider instead of
  applying collision directly to the rendered prim;
- `planning_obstacle: true` makes startup verify that cuMotion tracks the
  collision root;
- `role` is required documentation exposed on the prim.

Test the file without launching Isaac:

```bash
.venv/bin/python -m pytest tests/test_factory_scene_layout.py -q
```

### When JSON is not enough

Use a proper USD asset when geometry becomes artist-authored, instanced or has
variants. A good future structure is:

```text
assets/factory_shell.usda       # referenced static asset
assets/receiver.usda            # reusable receiver module
scenes/cheese_factory.usda      # composition root
runtime_override.usda           # generated/session data, not committed
```

Do this as a tested migration. Do not export the anonymous live stage and call
it the new source of truth without checking asset paths, physics schemas,
meters-per-unit, up axis, references and runtime-owned prims.

## Physics: five independent questions

For every object ask:

1. Does it have visible geometry?
2. Does it have collision geometry?
3. Is it static or a rigid body?
4. Are mass, friction, contact offsets and solver settings reasonable?
5. Does the robot motion planner know it is an obstacle?

A yes to one question says nothing about the others.

### Static equipment

A wall or machine frame usually needs collision but not `RigidBodyAPI`. Keep
collision meshes simple. Convex approximations are faster and more stable than
arbitrary triangle meshes.

### Dynamic product

A physically transported cheese needs collision, a rigid body, mass/inertia,
friction and a graspable shape. The current model-mode visual cutout does not
satisfy this requirement; the hidden task cube carries the physical state.

### Robot planning

The Franka controller uses a NVIDIA manipulation task backed by motion
generation. Planner collision representations and PhysX collision are not
automatically identical. After adding equipment:

- represent it in the planner world;
- visualize robot collision spheres/geometry;
- test every approach, descent, lift, transfer, release and retreat;
- test failure recovery, not only successful trajectories.

## Cameras and perception

There are two different cameras:

- `/World/InspectionCamera` — calibrated RTX sensor used by perception;
- `/World/OverviewCamera` — human presentation viewport.

The inspection camera produces RGB and distance-to-image-plane data. A
background-subtraction detector finds a foreground component in a configured
ROI. Its centroid is projected onto the known belt plane. The crop is routed by
one of the three classifier modes.

Changing camera pose, field of view, resolution or ROI is a dataset change, not
just an artistic camera adjustment. It can invalidate:

- foreground detection thresholds;
- pixel-to-world projection;
- crop scale and distribution;
- model accuracy;
- previous evaluation evidence.

## Replicator and training

Replicator is Isaac's synthetic-data system. In this repository it randomizes
camera calibration, lighting, material appearance, product pose/scale, clutter
and occlusion while retaining exact provenance. It creates training images and
metadata; PyTorch performs model training.

The correct loop is:

```text
reviewed source cutout
 -> Isaac canonical camera / Replicator render
 -> group-preserving train/validation manifest
 -> PyTorch training
 -> frozen validation gate
 -> locked test only for a passing candidate
 -> final live Isaac evaluation
```

Avoid learning from the same physical source object in both training and
validation. Multiple camera views of one cheese are still one source group.

## ROS and MCP

Isaac can publish sensors and robot state through ROS 2 and consume robot
commands. This project does not require ROS for its maintained runtime; the
loop currently runs in-process plus an HTTP model service.

The installed `isaacsim_mcp` searches Isaac documentation and extension
information. It does not press Play, modify the Stage or control the robot.
The repository's `control_server.py` is a separate, deliberately limited
localhost control/read interface. Its unauthenticated surface is read-only.
Mutations require an operator token, actor label and unique request UUID, and
are restricted to bounded evaluation, canonical reset and emergency stop. It
cannot execute arbitrary Python, shell commands, paths or USD edits.

## Debugging checklist

### Object is visible but arm passes through it

- Inspect whether CollisionAPI is applied.
- Inspect whether the object is in the planner world.
- Confirm the planner's robot collision model covers the relevant link.
- Check that animation/teleport code is not bypassing physics.

### Robot does not move

- Is the timeline playing?
- Did the Franka asset finish loading?
- Is the task in a fault/recovery state?
- Did perception return `ok`, or did safety correctly reject it?
- Inspect `outputs/factory/gui.log` and runtime status.

### Camera sees the wrong thing

- Confirm active sensor path, not overview camera.
- Check pose, FOV, resolution, clipping and ROI in `config.yaml`.
- Check decorative housing is not on the optical axis.
- Inspect saved annotated frames, not only the overview viewport.

### Stream is blank or stale

```bash
infra/isaac-sim/factory-demo.sh recover showcase
```

Then refresh noVNC and ensure only one viewer is connected.

## Recommended learning order

1. OpenUSD prims, paths, transforms, layers and references.
2. Isaac timeline and Python/Kit execution model.
3. Collision, rigid bodies, articulations and joints.
4. Camera sensors and coordinate projection.
5. Robot kinematics, cuMotion/RMPflow and obstacle worlds.
6. Replicator and dataset provenance.
7. ROS 2 only if the next milestone connects external robot software.

Learn by changing one observable variable at a time and recording the Git SHA,
configuration, screenshot and result. Isaac's complexity becomes manageable
when scene appearance, physics, planning, perception and evidence are treated
as separate systems.
