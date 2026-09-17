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
│   ├── render_belt.py        USD scene, domain randomisation      [[Isaac Sim rendering]]
│   ├── usd_kit.py            belt, plate, camera helpers, shared
│   ├── sort_server.py        the model behind an HTTP service     [[Sorting line demo]]
│   ├── sorting_line.py       the diverter line demo               [[Sorting line demo]]
│   ├── pick_cell.py          FR3 cell: belt, arm, lanes, reward   [[Pick and place cell]]
│   ├── train_pick.py         PPO for the arm, 1024 cells          [[Pick and place cell]]
│   ├── pick_line.py          camera + arm end to end              [[Pick and place cell]]
│   ├── banc_automate.py      bench: the reference state machine
│   ├── banc_politique.py     bench: what a checkpoint really does
│   └── make_video.py         HUD overlay + H.264, both demos
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
| train the arm | `sim/train_pick.py --envs 1024 --iters 900` → [[Pick and place cell]] |
| film the arm cell | `sim/pick_line.py` then `sim/make_video.py` |
| check the cell is sane | `sim/banc_automate.py` — the feasibility witness |
| retrain the shipped model | `src/train.py --task sim_type --manifest data/processed/manifest_sim.csv` |
| classify one image | `src/predict.py runs/sim_type13/best.pt <image>` |
| export for TensorRT | `src/export.py runs/sim_type13/best.pt` |

## The task table

`src/dataset.py` holds a `TASKS` dict mapping a task name to *(sources, label column)*.
Adding a head means adding one entry there — no other change. Current entries:
`bin`, `sim_bin`, `sim_type`, `sim_type_cheese`, `fr_cheese_type`, `hidb_product`,
`hidb_ripeness`, `hidb_product_ripeness`, `variety`, `texture`, `all`.
