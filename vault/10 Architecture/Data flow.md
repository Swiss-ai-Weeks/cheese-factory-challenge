---
tags: [architecture]
---

# Data flow

Every artefact, and what transforms it into the next.

```mermaid
flowchart TD
    classDef raw fill:#f1f5f9,stroke:#94a3b8
    classDef proc fill:#fef3c7,stroke:#b8894a
    classDef model fill:#dcfce7,stroke:#16a34a

    FR["Food Recognition 2022<br/>44k images + polygons<br/>3.2 GB"]:::raw
    HI["CHEESE-HIDB<br/>378 images 6016x4016<br/>6.0 GB"]:::raw
    CI["cheese-images<br/>3.2k photos, 290 varieties<br/>928 MB"]:::raw

    FR --> NORM["normalize.py<br/>--workers 64"]
    HI --> NORM
    CI --> NORM
    NORM --> MAN["manifest.csv<br/>6486 rows"]:::proc
    NORM --> IMG["images/&lt;source&gt;/&lt;class&gt;/<br/>JPEG 512 q92"]:::proc
    NORM --> ST["stats.json<br/>train-split mean/std"]:::proc

    MAN --> CUT["cutouts.py<br/>--negatives 400"]
    FR -.polygons.-> CUT
    CUT --> CO["cutouts/&lt;bin&gt;/<br/>2400 RGBA PNG"]:::proc

    CO --> REND["render_belt.py<br/>Isaac Sim 6.0.1"]
    REND --> RAW768["sim/out/<br/>PNG 768, archive"]:::proc
    REND -.--empty 900.-> RAW768

    RAW768 --> RM["render_manifest.py<br/>convert + split"]
    CO -.type lookup.-> RM
    RM --> SB["images/sim_belt/&lt;bin&gt;/<br/>12355 JPEG 512"]:::proc
    RM --> MS["manifest_sim.csv<br/>3191 pieces"]:::proc

    MS --> TR["train.py<br/>--task sim_type"]
    ST -.mean/std.-> TR
    TR --> CK["runs/sim_type13/<br/>best.pt + results.json"]:::model
    CK --> EXP["export.py"] --> ONNX["best.onnx + sidecar"]:::model
    CK --> PRED["predict.py<br/>CheeseSorter"]:::model
```

## Sizes at each stage

| stage | on disk | count |
|---|---|---|
| `data/raw/` | ~10 GB | 3 datasets |
| `data/processed/images/` | 315 MB | 6486 |
| `data/processed/cutouts/` | ~120 MB | 2400 |
| `sim/out/` (archive) | 3.6 GB | 12355 PNG |
| `data/processed/images/sim_belt/` | 311 MB | 12355 JPEG |
| `runs/` | ~2.5 GB | 8 checkpoints + 2 ONNX |

## What is authoritative for what

> [!important] When two sources disagree, this is the order
> 1. **files on disk** — for which renders exist ([[Render manifest]])
> 2. **cutouts manifest** — for the fine type of a piece
> 3. **shard JSONs** — only for the camera angle, they overwrite each other
>
> Every one of those rules exists because of a bug — see [[Bug log]].
