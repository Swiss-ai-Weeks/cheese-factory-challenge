---
tags: [results]
priority: high
---

# Model comparison

> [!important] Only matched pairs are compared here
> Same splits, same test set, one variable at a time. The table in [[Models MOC]] is
> **not** a ranking.

## Question 1 — one model or two?

Does deriving the bin from the type head lose anything against a dedicated 5-class model?

| derivation | bin top-1 | bin macro-F1 |
|---|---|---|
| dedicated 5-class ([[sim_bin]]) | 0.763 | 0.709 |
| **from 11 types, by sum** | **0.771** | **0.716** |
| from 11 types, by argmax | 0.765 | 0.712 |

> [!success] One model suffices
> The margin is within noise, but nothing is lost — and the fine type comes free.
> See [[Probability marginalisation]].

## Question 2 — what does rejection cost?

[[sim_type11]] vs [[sim_type13]], same splits, 1,430 cheese test images.

| measurement | without reject | with reject | delta |
|---|---|---|---|
| type top-1 | 0.679 | 0.684 | +0.005 |
| type macro-F1 | 0.561 | 0.554 | -0.007 |
| bin top-1 | 0.757 | 0.754 | -0.003 |
| bin macro-F1 | 0.724 | 0.733 | +0.009 |

![[per-bin-f1.png]]

> [!success] Rejection is free
> All deltas within ±0.01; four of five bins slightly better with reject.

> [!bug] This conclusion was wrong the first time
> The first computation reported bin macro-F1 dropping **0.724 → 0.611** and nearly led
> to recommending a two-head architecture. The cause was in the *metric*, not the model:
> rejected cheeses were labelled `REJET`, creating a sixth phantom class with F1 0 that
> dragged the macro average.
>
> Arithmetic check: (0.660+0.824+0.825+0.615+0.740)/5 = 0.733, and 0.733 × 5/6 = **0.611**.
> Restricting the label set to the five real bins gives 0.733. See [[Measurement pitfalls]].

## Question 3 — did moving to the render domain help?

Not answerable with a number: [[bin]] is measured on real photos, [[sim_type13]] on
renders. Different test sets, different domains.

What *is* evidence: on a render of `bin_fresh`, [[bin]] answered `bin_semi_hard` at 0.59
— above threshold, so the arm would have acted wrongly — while the render-trained model
answered `bin_fresh` at 0.86.

## Question 4 — 60 epochs or 20?

Best epochs across all runs: **3, 9, 14, 17, 18, 19, 19, 38**. Never a late peak except
[[variety]] at 38/60. 20 epochs is enough; 60 wasted over an hour of GPU across five
models.
