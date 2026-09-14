---
tags: [model]
classes: 3
checkpoint: runs/hidb_product/best.pt
---

# hidb_product

 Semi-hard / hard / extra-hard on [[CHEESE-HIDB]] wheels, **split per wheel**.
 The only HIDB task an honest protocol supports.

 | | |
 |---|---|
 | test top-1 | 0.758 |
 | macro-F1 | 0.727 |
 | best epoch | 3/40 |
 | train | 128 images = **one wheel per product** |

 | product | precision | recall | F1 | support |
|---|---|---|---|---|
| `extra_hard` | 0.583 | 1.000 | 0.737 | 42 |
| `hard` | 1.000 | 0.286 | 0.444 | 42 |
| `semi_hard` | 1.000 | 1.000 | 1.000 | 40 |

 > [!danger] The global number is misleading
 > 0.758 top-1 reads respectably. Per class: `semi_hard` is **perfect**, while `hard` has
 > recall **0.286** — the model misses 71% of hard wheels, sending them to `extra_hard`.
 >
 > With **one training wheel per class** it learns "semi-hard or not" and guesses the
 > rest. This is exactly why [[Evaluation protocol]] insists on per-class tables.

 > [!info] Compare with the leaked version
 > The same task, split per image instead of per wheel, reported **1.000** macro-F1.
 > The gap between 1.000 and 0.727 is what the leak was hiding.
