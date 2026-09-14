---
tags: [ops]
---

# Environment

## Hardware

2× **NVIDIA H100 NVL** (94 GB each), 128 cores, 1 TB RAM, Ubuntu 24.04, 1.8 TB disk.

## Python

> [!bug] The pre-existing `.venv` had no pip
> `ensurepip` is missing system-wide, so `python3 -m venv` produced a venv with no
> package manager and no `activate`. Rebuilt with `uv`, installed standalone:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch torchvision \
    --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv/bin/python huggingface_hub datasets pillow numpy \
    pandas tqdm pyarrow timm scikit-learn matplotlib onnx onnxruntime \
    ipykernel nbconvert jupyterlab ipywidgets
```

## Observed performance characteristics

> [!tip] Training is GPU-bound
> 99% GPU utilisation with a CPU load of 12 on 128 cores. Raising dataloader workers to
> 32 and converting PNG→JPEG changed **nothing** on speed. The JPEG conversion earned
> its place for comparability and disk, not throughput — see [[Render manifest]].

| job | throughput |
|---|---|
| normalisation | 6,505 images in ~90 s on 64 processes |
| cutouts | 2,488 in ~40 s on 48 processes |
| rendering | ~0.5 s/image, ~220 images/min on 2 GPUs |
| training | 44-55 s/epoch at 384 px, batch 96 |
