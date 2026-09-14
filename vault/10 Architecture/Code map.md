---
tags: [architecture, reference]
---

# Code map

```
cheese/
├── src/
│   ├── normalize.py          3 raw datasets  -> one manifest      [[Normalisation]]
│   ├── cutouts.py            polygons        -> RGBA silhouettes  [[Cutouts]]
│   ├── render_manifest.py    renders         -> manifest + splits [[Render manifest]]
│   ├── dataset.py            PyTorch Dataset, task table, transforms
│   ├── train.py              training loop and evaluation         [[Training recipe]]
│   ├── predict.py            CheeseClassifier + CheeseSorter      [[Inference API]]
│   └── export.py             ONNX + sidecar                       [[ONNX export]]
├── sim/
│   └── render_belt.py        USD scene, domain randomisation      [[Isaac Sim rendering]]
├── data/
│   ├── raw/                  untouched downloads (~10 GB)
│   └── processed/            images, cutouts, manifests
├── runs/<head>/              best.pt, results.json, best.onnx
├── vault/                    this documentation
├── sync_jupyter.sh           copy to the Jupyter workspace        [[Jupyter web app]]
└── hpe.ipynb                 the notebook (at the Jupyter root)
```

## Key entry points

| I want to… | run |
|---|---|
| rebuild everything from raw | see [[Reproduce everything]] |
| re-render the belt | `sim/render_belt.py --all --views 5 --skip-existing` |
| retrain the shipped model | `src/train.py --task sim_type --manifest data/processed/manifest_sim.csv` |
| classify one image | `src/predict.py runs/sim_type13/best.pt <image>` |
| export for TensorRT | `src/export.py runs/sim_type13/best.pt` |

## The task table

`src/dataset.py` holds a `TASKS` dict mapping a task name to *(sources, label column)*.
Adding a head means adding one entry there — no other change. Current entries:
`bin`, `sim_bin`, `sim_type`, `sim_type_cheese`, `fr_cheese_type`, `hidb_product`,
`hidb_ripeness`, `hidb_product_ripeness`, `variety`, `texture`, `all`.
