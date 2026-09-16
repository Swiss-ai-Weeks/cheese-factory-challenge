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
infra/isaac-sim/     remote Isaac Sim GUI stack for SSH and Brev
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

## ⚡ Quick start

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch torchvision \
    --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv/bin/python timm pillow numpy pandas scikit-learn \
    matplotlib huggingface_hub onnx onnxruntime jupyterlab ipywidgets
```

Then follow [Reproduce everything](vault/80%20Ops/Reproduce%20everything.md).
