---
tags: [model, ablation, superseded]
checkpoint: runs/sim_bin/best.pt
classes: 5
---

# sim_bin — superseded

 5 sorting bins directly, trained on belt renders. **Was** the model to ship, until
 [[sim_type13]] proved one model could give the type as well.

 | | |
 |---|---|
 | test top-1 | 0.763 |
 | macro-F1 | 0.709 |
 | best epoch | 17/30 |

 | bin | precision | recall | F1 | support |
|---|---|---|---|---|
| `bin_hard` | 0.769 | 0.915 | 0.836 | 470 |
| `bin_soft` | 0.753 | 0.766 | 0.759 | 350 |
| `bin_fresh` | 0.811 | 0.817 | 0.814 | 300 |
| `bin_semi_hard` | 0.692 | 0.478 | 0.565 | 245 |
| `bin_blue` | 0.705 | 0.477 | 0.569 | 65 |

 > [!note] Why it was replaced
 > Deriving the bin from the 11-type model by [[Probability marginalisation|summing
 > probabilities]] scores **0.771 / 0.716** against this model's
 > 0.763 / 0.709 — no worse, and it returns the fine type as a bonus.

 Kept as a reference and because its ONNX export exists.
