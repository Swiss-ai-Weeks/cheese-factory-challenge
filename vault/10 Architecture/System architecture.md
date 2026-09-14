---
tags: [architecture]
---

# System architecture

## Runtime — what happens per frame

```mermaid
sequenceDiagram
    participant C as Belt camera
    participant D as Detector (to build)
    participant S as CheeseSorter
    participant R as Arm control
    C->>D: RGB frame 1280x720
    D->>D: background subtraction on a uniform belt
    D->>S: crop boxes [(x0,y0,x1,y1), ...]
    S->>S: resize 438 -> centre-crop 384 -> normalise
    S->>S: ConvNeXt-Base -> 13 logits -> softmax
    S->>S: sum probabilities per bin
    R-->>S: SortResult(status, bin, type, confidences)
    alt status == ok
        R->>R: place in bin
    else
        R->>R: let it pass, raise status
    end
```

> [!tip] The detector is not built yet
> It is the single highest-value addition. The classifier is trained on framed pieces;
> feeding it a whole belt frame loses the pixels that separate `bin_semi_hard` from
> `bin_hard`. See [[Robot pipeline]].

## Training — what produced the model

```mermaid
flowchart LR
    subgraph raw[data/raw]
        FR[Food Recognition 2022<br/>3.2 GB]
        HI[CHEESE-HIDB<br/>6.0 GB]
        CI[cheese-images<br/>928 MB]
    end
    subgraph proc[data/processed]
        M[manifest.csv]
        CO[cutouts/]
        SB[images/sim_belt/]
        MS[manifest_sim.csv]
    end
    FR --> N[normalize.py] --> M
    HI --> N
    CI --> N
    M --> CU[cutouts.py] --> CO
    CO --> RB[render_belt.py<br/>Isaac Sim] --> SB
    SB --> RM[render_manifest.py] --> MS
    MS --> T[train.py] --> CK[runs/sim_type13/best.pt]
    CK --> P[predict.py<br/>CheeseSorter]
    CK --> E[export.py<br/>ONNX]
```

## Design choices worth knowing

| choice | why | detail |
|---|---|---|
| one model, not two | the bins are a strict grouping of the types | [[Probability marginalisation]] |
| reject as classes, not a threshold | a threshold cannot say *why* it is unsure | [[Reject classes]] |
| split by object, not by file | the same wheel in train and test gives a fake 100% | [[Splits and data leakage]] |
| macro-F1 for checkpoints | classes are imbalanced 18:1 | [[Evaluation protocol]] |
| render in Isaac, not 2D compositing | reusable once real 3D assets exist | [[Decision log]] |
