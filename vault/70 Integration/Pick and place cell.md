---
tags: [integration, simulation, robotics, rl]
---

# Pick and place cell

The [[Sorting line demo]] one block further down: the camera's verdict no longer drives
a diverter, it drives a **Franka FR3**. A plate arrives, the belt indexes it under the
inspection station, `sim_type13` names the cheese, and the arm carries the plate to the
output conveyor for that class and sets it down flat, without spilling the piece.

```
 camera ──▶ sim_type13 (host, HTTP) ──▶ lane number ──▶ FR3 policy ──▶ motion
```

Two models, two trades, neither knows the other exists. The classifier returns a bin and
has never seen an arm. The arm receives **an integer** — never an image, never a class —
and knows only how to grip a plate, carry it and put it down.

## The pieces

| file | runs | role |
|---|---|---|
| `sim/pick_cell.py` | container | the cell: belt, FR3, six output conveyors, reward |
| `sim/train_pick.py` | container | PPO, 1024 cells in parallel on one H100 |
| `sim/pick_line.py` | container | the two models end to end, and the rendering |
| `sim/banc_automate.py` | container | bench for the reference state machine |
| `sim/banc_politique.py` | container | bench for a checkpoint: what it actually does |
| `sim/make_video.py` | host | HUD overlay + H.264, shared with the other demo |

> [!warning] What drives the arm in the video
> `sim/cheese_picking.mp4` is driven by the **reference state machine**, not by the
> learned policy. The HUD says so, top right. See [[#Where the policy stands]].

## Three things the cell had to be taught

Each of these alone was enough to make training impossible, and each cost a full run.

### A plate a parallel gripper can hold

A 180 mm plate gripped by its rim hangs 90 mm from its centre of mass: 0.12 N·m on two
parallel jaws. It tips, and the cheese it carries falls out. No reward tuning fixes a
task that is mechanically infeasible.

At **68 mm** the gripper straddles the whole plate. The grip line passes through the
centre of mass, the plate hangs flat, and the yaw becomes mechanically irrelevant — a
free degree of freedom removed from the problem. Measured: held motionless to within
0.7 mm for 400 steps.

### A reward that pays progress, not state

The first reward paid ~10 per step for carrying the plate, for as long as you liked, and
80 once for putting it down — **which ended the episode**. With `gamma = 0.99` the
effective horizon is 100 steps, so loitering was worth ~1000 and succeeding was worth 80.
The policy learned exactly that: 29 M steps parked above the grip point, reward plateaued
at 4.1, success 0.0. → `runs/pick_fr3_echec_bord/`

The reward is now the **change in a potential** (Ng, Harada & Russell, 1999): standing
still pays nothing, and the terminal bonus has no competitor. The potential has to be
zeroed on real terminations, otherwise the shaping itself argues against ending the
episode.

> [!note] The potential must describe the gesture, not the straight line
> Aimed at the drop point in 3D, the policy discovered that descending is cheaper than
> going round the arm's own pedestal: it parked the plate mid-height above the belt and
> stopped. Carrying height first, descent only above the lane, and the first deposits
> appeared within forty iterations.

### An arm that can reach its own lanes

Joint 1 of an FR3 stops at ±157°, so there is a **46° sector behind the arm where
nothing is reachable**. Mounted facing the belt, that sector fell in the middle of the
output arc: `bin_soft` and `bin_fresh` were unreachable, and the state machine sat in
front of them until the cycle expired. The base is now turned so the dead sector lands in
the gap between the last lane and the belt — everything is within 119°, with 38° spare.

## Where the policy stands

| | |
|---|---|
| steps | 12.3 M (501 iterations × 1024 cells × 24) |
| plate held | up to **33%** of cells |
| plate set down | **0** |

The policy grips, lifts and carries. It has never completed a deposit, and when the
"already gripped" curriculum fades out at iteration 500 it does not pick off the belt
either — holding falls to zero. The curve is in `runs/pick_fr3/results.json` and in
section 9.1 of the notebook.

What is left is an **exploration** problem, not a physics one: lowering the plate onto
the lane and opening the jaws. The jaw command is a sign decision, the policy learns to
close early and its mean sits around −1.2, so the probability of ever sampling an opening
falls to 1e-5. The floor on the action noise raises it; it is not enough on its own.

> [!tip] The next thing to try
> A second curriculum symmetric with the first: episodes that start with the plate
> already resting on its lane, jaws still closed, so that *opening* becomes the only
> remaining action and therefore explorable. The state machine deposits in the same cell,
> so the task is reachable — it is the policy that has not found it.

## Running it

```bash
# train
.../python.sh /workspace/sim/train_pick.py --envs 1024 --iters 900

# the line end to end — model on the host, scene in the container
.venv/bin/python sim/sort_server.py
.../python.sh /workspace/sim/pick_line.py --out /workspace/sim/rendu_bras
.venv/bin/python sim/make_video.py --render sim/rendu_bras --out sim/cheese_picking.mp4
```

`--scripte` swaps the policy for the state machine. `sim/resume_partiel.py` rebuilds the
summary of a render stopped part-way, so the video can still be cut.

## Staging notes

The feed belt is 5 m long and runs off both edges of the frame: the plate is **born
outside the camera field** and rides in, instead of appearing in the middle of the shot.
The six output conveyors carry what is set down on them out of frame, which is both what
a real cell does and what keeps the next piece from replacing a plate still visible on
its lane — there is only one plate body in the scene.

→ [[Sorting line demo]] · [[Robot pipeline]] · [[Bug log]]
