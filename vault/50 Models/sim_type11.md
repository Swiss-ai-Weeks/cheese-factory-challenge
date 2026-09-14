---
tags: [model, ablation]
checkpoint: runs/sim_type11/best.pt
classes: 11
---

# sim_type11 — the ablation

Identical to [[sim_type13]] but **without** the reject classes, trained on the **same
splits** so the two share one test set.

> [!info] Why it exists
> To answer one question: *what does adding `empty` and `not_cheese` cost on the cheese
> task?* An earlier attempt compared against a model trained on different splits and
> produced a false answer — see [[Splits and data leakage|trap 3]].

| measurement | sim_type11 | [[sim_type13]] | delta |
|---|---|---|---|
| type top-1 | 0.679 | 0.684 | +0.005 |
| type macro-F1 | 0.561 | 0.554 | -0.007 |
| bin top-1 | 0.757 | 0.754 | -0.003 |
| bin macro-F1 | 0.724 | 0.733 | +0.009 |

![[per-bin-f1.png]]
![[per-type-f1.png]]

> [!success] Conclusion: rejection is free
> All four differences are within ±0.01. Four of five bins are slightly **better** with
> the reject classes. No two-head architecture is needed.

See [[Model comparison]] for the full analysis, including a correction to an earlier
wrong conclusion.
