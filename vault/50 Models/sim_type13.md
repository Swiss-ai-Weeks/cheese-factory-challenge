---
tags: [model, ship-it]
checkpoint: runs/sim_type13/best.pt
classes: 13
trained_on: Isaac Sim belt renders
status: shipped
---

# sim_type13 — the model to ship

 > [!success] One network, three answers
 > the fine **type** (`emmental_cheese`), the **bin** it maps to (`bin_hard`), and a
 > **status** telling the arm whether to act at all.

 ## Identity

 | | |
 |---|---|
 | backbone | `convnext_base.fb_in22k_ft_in1k_384` |
 | classes | 13 — 11 cheese types + `not_cheese` + `empty` |
 | train / val / test | 8,655 / 1,850 / 1,850 images |
 | pieces | 2,235 / 478 / 478 |
 | epochs | 20, best at **19** |
 | checkpoint | `runs/sim_type13/best.pt` |

 ## Results

 | measurement | top-1 | macro-F1 |
 |---|---|---|
 | 13 classes (raw task) | 0.728 | 0.600 |
 | 11 cheese types only | 0.684 | 0.554 |
 | **bin decision** | **0.754** | **0.733** |
 | cheese vs reject | 0.953 | 0.932 |

 > [!note] Why four numbers and not one
 > The reject classes are easy — `empty` has recall 1.000 — so a blended figure would be
 > flattered by them. See [[Evaluation protocol]].

 ### Per class

 | class | precision | recall | F1 | support |
|---|---|---|---|---|
| `hard_cheese` | 0.745 | 0.881 | 0.807 | 430 |
| `not_cheese` | 0.858 | 0.828 | 0.843 | 285 |
| `soft_cheese` | 0.661 | 0.765 | 0.709 | 260 |
| `semi_hard_cheese` | 0.525 | 0.503 | 0.514 | 145 |
| `empty` | 1.000 | 1.000 | 1.000 | 135 |
| `cottage_cheese` | 0.790 | 0.983 | 0.876 | 115 |
| `fresh_cheese` | 0.583 | 0.445 | 0.505 | 110 |
| `raclette_cheese` | 0.654 | 0.700 | 0.676 | 100 |
| `goat_cheese_soft` | 0.478 | 0.259 | 0.336 | 85 |
| `cream_cheese` | 0.455 | 0.267 | 0.336 | 75 |
| `blue_mould_cheese` | 0.917 | 0.508 | 0.653 | 65 |
| `emmental_cheese` | 0.773 | 0.425 | 0.548 | 40 |
| `processed_cheese` | 0.000 | 0.000 | 0.000 | 5 |

 ![[confusion-sim_type13.png]]

 ## Strengths

 - never misses an empty belt (recall **1.000**)
 - rejects only 2.4% of real cheese
 - 21% of its type errors stay **inside the correct bin** — invisible to sorting
 - ~14 ms per piece

 ## Weaknesses

 > [!failure] `processed_cheese` is never recognised
 > 0.000 F1. 160 training images, 5 in test — the class barely exists for the model.
 > All 5 test instances are classified as `not_cheese`.

 > [!warning] 17% of foreign objects pass as cheese
 > `not_cheese` recall 0.828. See [[Reject performance]].

 > [!warning] `semi_hard` ↔ `hard` is the dominant real error
 > 41% of `semi_hard_cheese` goes to `hard_cheese` — and those are **different bins**.
 > Paste grain is what separates them, and the piece is small in frame.
 > The fix is architectural: [[Robot pipeline]].

 ## Usage

 [[Inference API]] · [[Output contract]] · [[ONNX export]]
