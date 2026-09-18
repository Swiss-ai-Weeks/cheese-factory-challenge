# Cheese Factory — Perception

Perception block for the **Physical AI cheese factory** challenge: a belt camera frame
goes in, a sorting decision comes out, and the downstream pipeline drives the robot arm.

```
Isaac Sim camera ──▶ CheeseSorter.predict(frame) ──▶ SortResult ──▶ arm control
```

The model answers three things at once — the fine cheese **type**, the **bin** it belongs
in, and a **status** telling the arm whether to act at all.

```python
from predict import CheeseSorter

sorter = CheeseSorter("runs/sim_type13/best.pt", min_confidence=0.55)
sorter.warmup()

r = sorter.predict(rgb_frame)
if r.actionable:              # status == "ok"
    place_in_bin(r.bin)       # "bin_hard"
else:
    signal(r.status)          # "empty" | "not_cheese" | "uncertain"
```

---

## 📖 Documentation

**The full documentation is an Obsidian vault in [`vault/`](vault/).**
Open that folder in Obsidian (*Open folder as vault*) — 52 linked notes, 11 figures and a
visual canvas. It also reads fine as plain Markdown on GitHub.

Start at **[`vault/00 Index/Home.md`](vault/00%20Index/Home.md)**, or jump straight to:

| I want to… | read |
|---|---|
| see the model drive a sorting line | [Sorting line demo](vault/70%20Integration/Sorting%20line%20demo.md) |
| understand where the data comes from | [Data journey](vault/20%20Datasets/Data%20provenance.md) |
| integrate the robot | [Output contract](vault/70%20Integration/Output%20contract.md) · [Inference API](vault/70%20Integration/Inference%20API.md) |
| see what the models are worth | [Model comparison](vault/60%20Results/Model%20comparison.md) |
| know what is **not** proven | [What the numbers do not measure](vault/60%20Results/What%20the%20numbers%20do%20not%20measure.md) |
| rebuild everything | [Reproduce everything](vault/80%20Ops/Reproduce%20everything.md) |
| learn from our mistakes | [Bug log](vault/90%20Decisions/Bug%20log.md) · [Measurement pitfalls](vault/60%20Results/Measurement%20pitfalls.md) |

**[`hpe.ipynb`](hpe.ipynb)** is a runnable notebook covering the same ground with live
inference and dropdown selectors. `hpe_fr.ipynb` is the French version.

---

## 🧀 What it recognises

11 cheese types, grouped into 5 sorting bins, plus 2 reject classes:

| bin | types |
|---|---|
| `bin_hard` | hard_cheese, emmental_cheese |
| `bin_semi_hard` | semi_hard_cheese, raclette_cheese |
| `bin_soft` | soft_cheese, goat_cheese_soft, processed_cheese |
| `bin_fresh` | fresh_cheese, cottage_cheese, cream_cheese |
| `bin_blue` | blue_mould_cheese |
| *(no bin)* | `not_cheese`, `empty` — the arm must not act |

> It does **not** name commercial varieties ("Comté"). No dataset in the project carries
> both a variety name and a paste type. See [variety](vault/50%20Models/variety.md).

## 🎩 How we multiplied a small dataset

Food Recognition only yields ~2,900 usable cheese crops. Training set is **12,355
images** — and not one of them was annotated by hand. Four tricks, in order of how much
they actually bought:

**1. The labels were already there, and so were the outlines.**
Food Recognition is a *segmentation* dataset: every instance carries a human-drawn
polygon. That polygon becomes the alpha channel, giving a piece with a **real silhouette**
instead of a rectangle of photo — and the class label rides along for free. No SAM, no
manual labelling, six lines of PIL.

**2. The 482 classes we did not want became the class we needed.**
Only 11 of the 498 food classes are cheese. The other 482 — bread, tomato, egg, salmon —
are exactly what a `not_cheese` class needs: real objects, real silhouettes, already
annotated. We sampled 400 objects across **400 distinct classes** (cycling the class list
rather than taking the most frequent, so the model sees variety rather than 400 slices of
bread) → **1,910 renders**, for zero data collection.

**3. An empty belt costs one line.**
Hide the quad, render the scene: `UsdGeom.Imageable.MakeInvisible()`. 900 images of
`empty`, half with an empty plate and half with nothing, so the model separates "nothing
to pick" from "a container is there". The model now has **perfect recall on empty belts**.

**4. Five viewpoints per piece.**
Each cutout is placed in the plate and rendered 5 times with randomised camera azimuth,
elevation, distance and focal length, lamp position/intensity/tint, and plate and belt
shades. 1,909 pieces → **9,545 images**.

> [!IMPORTANT]
> **Trick 4 multiplies images, not cheeses.** There are still 1,909 distinct pieces seen
> from more angles. It buys viewpoint invariance — genuinely useful for a belt camera —
> but it teaches no new cheese.
>
> It also creates a trap: five renders of one piece are **one object**, not five samples.
> Split them across train and test and you measure memorisation. Every manifest carries a
> `group` column holding the source piece, and that is what drives the split — never the
> filename. We learned this the hard way on CHEESE-HIDB, where 9 wheels photographed 42
> times each produced a meaningless **100% macro-F1**. See
> [Splits and data leakage](vault/30%20Pipeline/Splits%20and%20data%20leakage.md).

| | pieces | images |
|---|---|---|
| cheese, 11 types | 1,909 | 9,545 |
| `not_cheese`, 400 classes | 382 | 1,910 |
| `empty` | 900 | 900 |
| **total** | **3,191** | **12,355** |

## 📊 Results

Shipped model **`sim_type13`** — ConvNeXt-Base, 384 px, trained on 12355 Isaac Sim
belt renders from 3191 distinct pieces.

| measurement | top-1 | macro-F1 |
|---|---|---|
| 13 classes | 0.728 | 0.600 |
| bin decision | 0.754 | 0.733 |
| cheese vs reject | 0.953 | 0.932 |

`empty` recall **1.000** · `not_cheese` recall 0.828 · ~14 ms per piece on an H100.

> [!WARNING]
> **These numbers do not measure performance on the real demo scene.** Training data is
> renders built from photographs; nothing has been validated against the actual Isaac
> scene. This is the largest remaining risk —
> see [Open questions](vault/90%20Decisions/Open%20questions.md).

## 🔧 How it was built

```
3 public datasets  ──▶  normalise  ──▶  polygon cutouts  ──▶  Isaac Sim render  ──▶  train
   (already annotated)                  (real silhouettes)     (plate on a belt)
```

No image was hand-labelled. Every label rides for free from annotations that already
existed — see [Data journey](vault/20%20Datasets/Data%20provenance.md).

| source | role |
|---|---|
| [Food Recognition 2022](https://datasetninja.com/food-recognition) | everything shipped — polygons make the cutouts possible |
| [CHEESE-HIDB](https://github.com/andrealoddo/CHEESE-HIDB) | a cautionary tale about data leakage |
| [NoeFlandre/cheese-images](https://huggingface.co/datasets/NoeFlandre/cheese-images) | variety names, unusable on renders |

## 📁 Repository layout

```
src/                 normalize · cutouts · render_manifest · dataset · train · predict · export
sim/render_belt.py   the USD belt scene and domain randomisation
sim/sorting_line.py  the demo: a full sorting line, routed by the model
sim/pick_cell.py     the FR3 cell: belt, arm, six output lanes, reward
sim/train_pick.py    PPO for the arm, 1024 cells in parallel
sim/pick_line.py     camera + arm end to end: the model decides, the arm executes
sim/make_video.py    the HUD and the H.264 encode, for both demos
sim/banc_automate.py bench for the reference state machine — the feasibility witness
sim/banc_politique.py bench for a checkpoint: what the policy does, step by step
sim/factory/         Isaac 6.1 camera-to-control evaluation harness and metrics
infra/isaac-sim/     remote Isaac Sim GUI stack for SSH and Brev
docs/                demo runbook, evaluation protocol and MCP compatibility research
runs/<head>/         results.json and ONNX sidecars for 8 trained heads
vault/               the documentation (Obsidian)
hpe.ipynb            runnable notebook
sync_jupyter.sh      copy to a Jupyter workspace
```

> **Not in this repository:** `data/` (14 GB of downloads + 947 MB processed),
> `sim/out/` (4.6 GB of raw renders) and the `.pt` / `.onnx` weights (350 MB each, above
> GitHub's 100 MB file limit). Everything regenerates from
> [Reproduce everything](vault/80%20Ops/Reproduce%20everything.md) in about two hours,
> most of it rendering.

## 🦾 The arm

The camera's verdict drives a Franka FR3: the plate is indexed under the inspection
station, `sim_type13` names the cheese, and the arm carries the plate to the lane for
that class. Two models, two trades — the classifier has never seen an arm, and the arm's
policy receives an integer, never an image.

Three things the cell had to be taught, each of which cost a full training run:

* **A plate a parallel gripper can actually hold.** 180 mm gripped by the rim hangs
  90 mm from its centre of mass — 0.12 N.m on two jaws, it tips and the cheese falls. At
  68 mm the gripper straddles it and the grip line runs through the centre of mass.
* **A reward that pays progress, not state.** Carrying paid ~10 per step for as long as
  you liked; putting the plate down paid 80 once and ended the episode. Loitering was
  worth ten times succeeding, and 29 M steps went into learning exactly that. The reward
  is now the change in a potential, which is zero when nothing happens.
* **An arm that can reach its own lanes.** Joint 1 of an FR3 stops at ±157°, so there is
  a 46° sector behind it where nothing is reachable. Mounted facing the belt, that sector
  sat in the middle of the lane arc: `bin_soft` and `bin_fresh` were unreachable.

```bash
# train (in the container, one H100)
.../python.sh /workspace/sim/train_pick.py --envs 1024 --iters 900

# the line, end to end — model on the host, scene in the container
.venv/bin/python sim/sort_server.py
.../python.sh /workspace/sim/pick_line.py --out /workspace/sim/rendu_bras
.venv/bin/python sim/make_video.py --render sim/rendu_bras --out sim/cheese_picking.mp4
```

`--scripte` swaps the policy for the reference state machine, which is the witness that
the task is feasible at all — and it is what drives the arm in `sim/cheese_picking.mp4`
today. **The learned policy is not there yet**: it grips, lifts and carries the plate
(up to 33% of cells holding), but it has never completed a deposit, and once the
"already gripped" curriculum fades out it no longer picks off the belt either. The curve
is in `runs/pick_fr3/results.json` and in section 9.1 of the notebook. `runs/pick_fr3_echec_bord/`
keeps the 29 M-step run that failed before the cell was fixed, as the reference point.

What is left to solve is an exploration problem, not a physics one: lowering the plate
onto the lane and opening the jaws. The state machine does it in the same cell, so the
task is reachable; the policy has not found it.

### Staging

The feed belt is 5 m long and runs off both edges of the frame, and the plate is **born
outside the camera field**: it rides into shot instead of appearing in the middle of it.
The six output conveyors carry away what is set down on them — that is what a real cell
does, and it is also what stops the next piece from replacing a plate still sitting
visibly on its lane, since the scene holds only one plate body.

Both videos use the same HUD (`sim/make_video.py`): inspection thumbnail, the model's
decision with per-type probabilities and latency, per-output counters, and — for the arm
— its phase, the plate height, and the placed/missed tally.

The full write-up, including every failure that got in the way, is in the vault:
[Pick and place cell](vault/70%20Integration/Pick%20and%20place%20cell.md) and
[Bug log](vault/90%20Decisions/Bug%20log.md).

## Isaac Sim 6.1 evaluation harness

The integrated repository also keeps the modular harness under `sim/factory/`.
It uses the Isaac Sim 6.1 experimental RTX camera API, separates perception,
state-machine and controller logic, and writes machine-readable detection,
classification, pick, placement and timing metrics.

```bash
# deterministic camera-to-controller integration check
infra/isaac-sim/run-headless.sh development

# streamed GUI on the remote workstation
infra/isaac-sim/run-gui.sh development
```

The explicit `development` mode classifies rendered proxy colours. It is useful
for testing the complete software and robot-control path, but its results are
**not trained-model accuracy**. Production mode expects the ignored
`runs/sim_type13/best.pt` fine-type checkpoint plus
`runs/sim_bin_adapt_v2/best.pt` for direct routing, and fails closed when either
model or the host inference service is unavailable. See
[`docs/stages/05-production-perception.md`](docs/stages/05-production-perception.md)
for target-camera capture, training, and honest trained-model results.

For agent/UI integration, `sim/factory/control_server.py` exposes localhost-only
health and latest-result reads. A fixed, bounded evaluation start is available
only with an operator-provided token; NVIDIA's separate `isaacsim_mcp` remains
a documentation and extension-search service, not a simulator control plane.
See [`docs/stages/06-agent-ready-control.md`](docs/stages/06-agent-ready-control.md).

The three robot paths have different evidence:

- `sim/pick_line.py --scripte` is the reliable scripted reference state machine
  used by the committed arm video.
- The learned PPO arm policy remains experimental: it grips and carries but has
  not completed a deposit.
- `sim/factory/` is a deterministic Isaac 6.1 integration/evaluation harness;
  its development-classifier metrics validate plumbing, not cheese recognition.

See the [demo runbook](docs/demo_runbook.md), [evaluation protocol](docs/evaluation_results.md)
and [Isaac MCP research](docs/isaac_mcp_research.md).

## ⚡ Quick start

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch torchvision \
    --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv/bin/python timm pillow numpy pandas scikit-learn \
    matplotlib huggingface_hub onnx onnxruntime jupyterlab ipywidgets
```

Then follow [Reproduce everything](vault/80%20Ops/Reproduce%20everything.md).
