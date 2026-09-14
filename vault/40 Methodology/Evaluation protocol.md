---
tags: [methodology, results]
---

# Evaluation protocol

## Which metric, and why

| metric | used for | why |
|---|---|---|
| **macro-F1** | checkpoint selection, headline | classes are imbalanced 18:1; accuracy would ride on `hard_cheese` alone |
| top-1 | reported alongside | intuitive, and what a per-piece demo shows |
| top-5 | reported | mostly saturated at 13 classes, kept for the 288-class head |
| per-class P/R/F1 | **the one that matters** | a flat global number hides a collapsed class |

> [!important] Always read the per-class table
> [[hidb_product]] shows 0.758 top-1 and looks respectable — until you see `hard` has
> recall **0.286**. The global number was carried by one easy class.

## How the shipped model is measured

Three separate measurements, never one blended number:

1. **13-class type** — the raw task
2. **11 cheese types only** — the reject classes are easy and would flatter the average
3. **cheese vs reject** — a binary view of the safety behaviour
4. **bin decision** — after [[Probability marginalisation]]

## Test set

1,850 images, grouped by piece, no overlap with train — [[Splits and data leakage]].

## Confidence threshold

The `min_confidence` gate is evaluated as a **coverage / precision trade-off**, not as
an accuracy number:

| threshold | pieces sorted | of which correct |
|---|---|---|
| 0.0 | 100% | 79.3% |
| 0.5 | 87.8% | 84.6% |
| 0.7 | 71.7% | 87.2% |
| 0.9 | 32.9% | 93.0% |

*(measured on [[bin]]; the shape holds for the shipped model)*

The right setting depends on the cost of a wrong bin versus a missed cycle — an
operations decision. See [[Robot pipeline]].
