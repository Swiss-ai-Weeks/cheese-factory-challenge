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
