---
tags: [model, baseline]
classes: 12
checkpoint: runs/fr_cheese_type/best.pt
---

# fr_cheese_type

The 12 fine cheese types, trained on **real photos** (before rendering). The real-photo
counterpart of [[sim_type13]]'s type head.

| | |
|---|---|
| test top-1 | 0.591 |
| macro-F1 | 0.499 |
| best epoch | 14/60 |

> [!note] Not an independent second opinion
> Its 12 classes map exactly onto [[bin]]'s 5. Same images, same information — their
> errors are correlated, so agreement between them adds little. Disagreement, however,
> flags a doubtful case. See [[Decision log]].
