---
tags: [integration]
script: src/predict.py
---

# Inference API

```python
from predict import CheeseSorter

sorter = CheeseSorter("runs/sim_type13/best.pt", min_confidence=0.55)
sorter.warmup()                          # absorbs the cuDNN cost before the first piece

r = sorter.predict(rgb_frame)            # ndarray HxWx3, uint8 or float 0-1, RGB or RGBA
if r.actionable:
    place_in_bin(r.bin)

# several pieces in one frame
results = sorter.predict_batch(rgb_frame, boxes=[(x0, y0, x1, y1), ...])
```

Output: [[Output contract]]. `r.as_dict()` is JSON-serialisable.

## Input handling

Accepts PIL images or numpy arrays, uint8 or float 0-1, RGB or **RGBA** — Isaac Sim
camera sensors emit several of these. Conversion is automatic.

## Preprocessing (applied internally)

```
RGB -> Resize(438) -> CenterCrop(384) -> /255 -> (x - mean) / std -> NCHW
```

with mean/std from the training checkpoint, not hardcoded.

## Latency — H100, 1280×720 input

| pieces per frame | time | rate |
|---|---|---|
| 1 | 13.6 ms | 73 fps |
| 4 | 30.5 ms | 33 fps |
| 8 | 49.5 ms | 20 fps |

Perception will not be the bottleneck.

> [!tip] The first call costs ~50 ms
> cuDNN autotuning. `warmup()` absorbs it — call it at startup.

## Two classes

| class | use |
|---|---|
| `CheeseSorter` | **the one to use** — type + bin + status |
| `CheeseClassifier` | raw single-head wrapper, used internally and by older heads |
