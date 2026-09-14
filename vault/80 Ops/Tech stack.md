---
tags: [ops, reference]
---

# Tech stack

## Core

| library | version | role |
|---|---|---|
| `torch` | 2.6.0+cu124 | training and inference |
| `torchvision` | matching | `transforms.v2` augmentation pipeline |
| `timm` | 1.0.29 | ConvNeXt backbones and pretrained weights |
| `numpy` | 2.5.2 | arrays throughout |
| `pillow` | 12.3.0 | all image I/O, polygon rasterisation, alpha |
| `scikit-learn` | 1.9.1 | F1, confusion matrix, classification report |
| `matplotlib` | 3.11.2 | every figure in this vault |

## Data acquisition

| library | role |
|---|---|
| `huggingface_hub` | `snapshot_download` for [[Cheese Images]] |
| `dataset-tools` | regenerates the dead DatasetNinja link |
| `pyarrow` | parquet reading |

## Simulation

| | |
|---|---|
| Isaac Sim | `nvcr.io/nvidia/isaac-sim:6.0.1` |
| USD API | `pxr` — `UsdGeom`, `UsdShade`, `UsdLux`, `Sdf`, `Gf` |
| Replicator | `omni.replicator.core` — render products, RGB annotator, orchestrator |
| renderer | RaytracedLighting, `rt_subframes=12` |

## Export and serving

| | |
|---|---|
| `onnx` / `onnxruntime` | export and numerical verification |
| `jupyterlab`, `ipykernel`, `ipywidgets` 8.1.9 | the notebook and its dropdowns |
| `nbconvert` | headless execution to fill outputs |

## Tooling

| | |
|---|---|
| `uv` 0.12.13 | the venv — the system Python had no `ensurepip` |
| Docker | Isaac Sim, Jupyter |
| `rsync` | `sync_jupyter.sh` |

## Notable non-choices

> [!note] What was deliberately not used
> - **No SAM or segmentation model** — the polygons were already in the data
> - **No albumentations** — `torchvision.transforms.v2` covers it
> - **No Lightning / Hydra** — a single readable training loop was clearer at this scale
> - **No DDP** — one GPU per run; two runs in parallel instead
> - **No MLflow / W&B** — `results.json` per run, read directly by the notebook and this vault
