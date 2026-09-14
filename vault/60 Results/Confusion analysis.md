---
tags: [results]
---

# Confusion analysis

![[confusion-sim_type13.png]]

## The dominant confusions

| true → predicted | rate | images | same bin? |
|---|---|---|---|
| `processed_cheese` → `not_cheese` | **1.00** | 5 | ❌ |
| `semi_hard_cheese` → `hard_cheese` | 0.41 | 59 | ❌ |
| `cream_cheese` → `fresh_cheese` | 0.32 | 24 | ✅ harmless |
| `goat_cheese_soft` → `soft_cheese` | 0.29 | 25 | ✅ harmless |
| `cream_cheese` → `soft_cheese` | 0.27 | 20 | ❌ |
| `emmental_cheese` → `hard_cheese` | 0.25 | 10 | ✅ harmless |
| `emmental_cheese` → `semi_hard_cheese` | 0.23 | 9 | ❌ |
| `raclette_cheese` → `hard_cheese` | 0.19 | 19 | ❌ |

> [!success] 21% of type errors never reach the sorting decision
> When the model confuses emmental with another hard cheese, the bin stays correct.
> This is the empirical justification for [[Probability marginalisation]].

## The one that actually hurts

`semi_hard_cheese` → `hard_cheese`, 41% of that class, **different bins**. What separates
them is paste grain, and in a belt render the piece occupies a small part of the frame.

> [!tip] The fix is architectural, not a better model
> Detect and crop the piece before classifying — [[Robot pipeline]].

## The class that does not exist

`processed_cheese`: 160 training images, 5 in test, **all 5 predicted `not_cheese`**.
F1 0.000. Either drop the class or gather more instances.
