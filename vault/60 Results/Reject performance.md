---
tags: [results, ship-it]
---

# Reject performance

Can the model say "do not pick this up"? Measured on 1,850 test renders, 420 of which
are reject cases.

## Binary view — cheese or not

```
             precision    recall  f1-score   support
    cheese      0.966     0.973     0.970      1430
     reject     0.907     0.883     0.895       420
   accuracy                         0.953      1850
```

## Per reject class

| class | recall | test images | reading |
|---|---|---|---|
| `empty` | **1.000** | 135 | never misses an empty belt |
| `not_cheese` | 0.828 | 285 | 17% of foreign objects pass as cheese |

## The two error directions

| error | count | consequence |
|---|---|---|
| cheese → reject | 34 (2.4%) | a cheese is not sorted. Cheap. |
| reject → cheese | 49 (11.7%) | **a foreign object goes into a food bin.** Expensive. |

Wrongly rejected cheeses concentrate in `bin_soft` (18 of 34).

> [!warning] The asymmetry is the wrong way round for safety
> The costly direction is the weaker one. Two mitigations, neither built:
> - **more negatives** — only 400 objects were used; Food Recognition has 482 classes
> - **a confidence gate on the reject head** — treat low-confidence `cheese` as reject

> [!tip] `empty` being perfect is not surprising
> An empty belt looks like nothing else in the dataset. Do not read it as evidence that
> the model is generally excellent — see [[Measurement pitfalls]].
