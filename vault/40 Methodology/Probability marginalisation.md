---
tags: [methodology, ship-it]
---

# Probability marginalisation

> [!abstract] Why one network answers two questions

The 5 bins are a **strict partition** of the 11 cheese types. So a model that outputs a
distribution over types already contains a distribution over bins:

$$P(\text{bin}=b \mid x) = \sum_{t \,\in\, b} P(\text{type}=t \mid x)$$

```python
per_target = {b: 0.0 for b in bins}
for i, t in enumerate(types):
    target = t if t in REJECT_STATUS else BIN_OF_TYPE[t]
    per_target[target] += probs[i]
bin = max(per_target, key=per_target.get)
```

## Why not argmax-then-map

A piece torn between `hard_cheese` at 0.45 and `emmental_cheese` at 0.40 would show a
**0.45 confidence** under argmax — apparently unsure. Summed, `bin_hard` comes out at
**0.85**: the model is certain about the bin and merely hesitant about the name.

That is not a detail. It means **the bin decision is systematically more reliable than
the type**, which is exactly the property the arm needs.

Measured on the test set:

| derivation | bin top-1 | bin macro-F1 |
|---|---|---|
| dedicated 5-class model ([[sim_bin]]) | 0.763 | 0.709 |
| **from 11 types, by sum** | **0.771** | **0.716** |
| from 11 types, by argmax | 0.765 | 0.712 |

The margin over the dedicated model is small enough to be noise. The point is that you
**lose nothing** and gain the type name for free.

## Evidence in the confusions

21% of type errors stay inside the same bin — `cream_cheese` → `fresh_cheese`,
`goat_cheese_soft` → `soft_cheese`, `emmental_cheese` → `hard_cheese`. Those are
invisible to the sorting decision. See [[Confusion analysis]].
