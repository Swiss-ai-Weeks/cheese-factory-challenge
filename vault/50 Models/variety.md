---
tags: [model]
classes: 288
checkpoint: runs/variety/best.pt
---

# variety

288 commercial cheese varieties from [[Cheese Images]]. The head that would answer
"which cheese is this?" — and cannot.

| | |
|---|---|
| test top-1 | 0.524 |
| top-5 | 0.704 |
| macro-F1 | 0.353 |
| best epoch | 38/60 |
| train | 2,257 images ≈ **8 per variety** |

Chance is 0.3%, so it does learn. But 125 of 288 classes have no test image at all.

> [!failure] It collapses on belt renders
> | | own domain | renders |
> |---|---|---|
> | median confidence | 0.224 | **0.030** |
> | above 0.5 | 32% | **0%** |
>
> On 240 renders it answered `trou_du_cru` 60 times and `ardi_gasna` 30 times. It is not
> recognising, it is guessing. Displaying that name next to a bin would be worse than
> showing nothing.

> [!tip] What replaced it
> [[sim_type13]] answers `emmental_cheese`-level names — not "Comté", but measurable and
> in the right domain.
