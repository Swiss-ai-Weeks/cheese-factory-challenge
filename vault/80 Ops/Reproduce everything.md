---
tags: [ops, reference]
---

# Reproduce everything

From an empty machine to the shipped model.

## 1 — Environment

See [[Environment]].

## 2 — Download the three datasets

```bash
.venv/bin/pip install dataset-tools huggingface-hub pyarrow opencv-python-headless
.venv/bin/python src/download_datasets.py

# Dataset Tools currently points at a disabled Dropbox archive. When the
# downloader reports a Hugging Face mirror, reconstruct the exact directory
# contract consumed by normalize.py. Multi-object mirror rows are excluded:
# that mirror does not preserve a reliable label-to-box association for them.
.venv/bin/python src/extract_food_recognition_hf.py \
  --clean --single-object-only
```

The downloader is idempotent. Use `--force` only when intentionally replacing
an existing raw dataset.

## 3 — Normalise and cut out

```bash
.venv/bin/python src/normalize.py --workers 64
.venv/bin/python src/cutouts.py --negatives 400 --workers 48
```

## 4 — Render the belt

```bash
mkdir -p outputs/render-belt && chmod 777 outputs/render-belt
docker run --rm --gpus all -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -e PYTHONPATH=/workspace -v "$PWD:/workspace" \
  --entrypoint /isaac-sim/python.sh \
  ${ISAAC_SIM_IMAGE:-nvcr.io/nvidia/isaac-sim:6.1.0} \
  /workspace/sim/render_belt.py --all --views 5 --min-fill 0.35 \
    --max-ar 3.0 --skip-existing --out /workspace/outputs/render-belt

# Then render the empty-belt reject class with the same mounts/image.
docker run --rm --gpus all -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -e PYTHONPATH=/workspace -v "$PWD:/workspace" \
  --entrypoint /isaac-sim/python.sh \
  ${ISAAC_SIM_IMAGE:-nvcr.io/nvidia/isaac-sim:6.1.0} \
  /workspace/sim/render_belt.py --empty 300 --views 1 --skip-existing \
    --out /workspace/outputs/render-belt
```

> [!important] Verify the counts before training
> ```bash
> ls outputs/render-belt/*.png | sed 's/.*__\(.*\)__.*/\1/' | sort | uniq -c | awk '$1 != 5'
> ```
> Should print nothing except the `empty` pieces. See [[Measurement pitfalls]].

## 5 — Manifest and training

```bash
.venv/bin/python src/render_manifest.py --renders outputs/render-belt \
    --out data/processed/manifest_sim.csv
.venv/bin/python src/train.py --task sim_type \
    --manifest data/processed/manifest_sim.csv \
    --model convnext_base.fb_in22k_ft_in1k_384 --img-size 384 \
    --epochs 20 --batch-size 64 --lr 1e-4 --workers 8 \
    --balanced-sampler --out runs/sim_type13

# Direct seven-way sorting head (five bins plus empty/not-cheese)
.venv/bin/python src/train.py --task sim_bin \
    --manifest data/processed/manifest_sim.csv \
    --model convnext_base.fb_in22k_ft_in1k_384 --img-size 384 \
    --epochs 20 --batch-size 64 --lr 1e-4 --workers 8 \
    --balanced-sampler --out runs/sim_bin
```

## 6 — Export and publish

```bash
.venv/bin/pip install onnx onnxscript onnxruntime-gpu
.venv/bin/python src/export.py runs/sim_type13/best.pt \
    --out runs/sim_type13/model.onnx
./sync_jupyter.sh
```

The ONNX exporter prints the maximum numerical difference against PyTorch.
Treat the export as accepted only when that parity check succeeds.

## 7 — The sorting line demo (optional)

```bash
.venv/bin/python sim/make_labels.py
.venv/bin/python sim/pick_demo_pieces.py --per-bin 2 --negatives 1
.venv/bin/python sim/sort_server.py &
# then the container, see [[Sorting line demo]]
.venv/bin/python sim/make_video.py
```

~15 min: 10 of rendering, the rest model and encoding.

Total: ~2 h wall clock, most of it rendering.
