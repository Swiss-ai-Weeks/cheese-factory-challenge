---
tags: [model, baseline]
checkpoint: runs/bin/best.pt
classes: 5
---

# bin — the real-photo baseline

 The same 5 bins, trained on [[Food Recognition 2022]] crops **before** any rendering.
 The reference the whole render pipeline is measured against.

 | | |
 |---|---|
 | test top-1 | 0.789 |
 | macro-F1 | 0.767 |
 | best epoch | 9/60 |
 | train set | 1,473 images |

 | bin | precision | recall | F1 | support |
|---|---|---|---|---|
| `bin_hard` | 0.837 | 0.897 | 0.866 | 97 |
| `bin_soft` | 0.778 | 0.671 | 0.721 | 73 |
| `bin_fresh` | 0.855 | 0.768 | 0.809 | 69 |
| `bin_semi_hard` | 0.696 | 0.780 | 0.736 | 50 |
| `bin_blue` | 0.632 | 0.800 | 0.706 | 15 |

 > [!warning] Do not compare this to [[sim_type13]] directly
 > It is measured on **real photographs**, a domain the robot will never see. Higher here
 > does not mean better on the belt.

 ## Where it visibly fails on renders

 On one render of `bin_fresh`, `bin` answered `bin_semi_hard` at **0.59** — above a 0.55
 threshold, so the arm would have confidently used the wrong bin. The render-trained model
 answered `bin_fresh` at 0.86. That single example is the argument for the whole render
 pipeline.

 ## Confidence threshold behaviour

 | threshold | pieces sorted | of which correct |
 |---|---|---|
 | 0.0 | 100% | 79.3% |
 | 0.5 | 87.8% | 84.6% |
 | 0.7 | 71.7% | 87.2% |
 | 0.9 | 32.9% | 93.0% |
