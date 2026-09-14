---
tags: [integration, ship-it]
priority: high
---

# Output contract

What the perception block hands the arm controller.

```python
SortResult(
    status          = "ok",              # TEST THIS FIRST
    bin             = "bin_hard",        # None unless status == "ok"
    bin_confidence  = 0.85,
    cheese_type     = "emmental_cheese", # for display
    type_confidence = 0.45,
    topk_types      = [("emmental_cheese", 0.45), ("hard_cheese", 0.40), ...],
    latency_ms      = 14.2,
)
```

## The four statuses

| status | meaning | what the arm should do |
|---|---|---|
| `ok` | a cheese was recognised | place it in `bin` |
| `empty` | nothing on the belt | do nothing |
| `not_cheese` | an object, but not cheese | reject lane / raise |
| `uncertain` | below `min_confidence` | let it pass |

> [!important] `bin` is `None` in every non-`ok` case
> By design, so a distracted caller cannot sort a piece by accident.
> `result.actionable` is the shorthand for `status == "ok"`.

```python
r = sorter.predict(frame)
if r.actionable:
    place_in_bin(r.bin)
else:
    signal(r.status)
```

## Confidence semantics

`bin_confidence` is the **sum** of the probabilities of all types in that bin, so it is
systematically at least as high as `type_confidence` — see
[[Probability marginalisation]]. Trust the bin more than the name.

> [!warning] Confidence is not a safety net
> A `bin_blue` piece has been observed predicted `bin_soft` at **0.917**. Confident
> errors exist and no threshold filters them.
