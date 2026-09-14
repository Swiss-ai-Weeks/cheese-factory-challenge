---
tags: [integration]
script: src/export.py
---

# ONNX export

```bash
.venv/bin/python src/export.py runs/sim_type13/best.pt
```

Produces `best.onnx` (~350 MB) and `best_sidecar.json`. Verified against PyTorch:
max deviation ~2e-06.

## The graph

Takes an **already normalised** NCHW float32 batch, returns logits. Dynamic batch axis,
opset 17.

> [!warning] The graph does not do the preprocessing or the marginalisation
> Both live in Python. If you go the TensorRT route you must reimplement:
> 1. resize 438 → centre-crop 384 → `/255` → `(x - mean) / std`
> 2. softmax, then **sum probabilities per bin** using the mapping in the sidecar
>
> See [[Probability marginalisation]].

## The sidecar

Carries class order, input size, mean/std, and the exact preprocessing string.
**Read the class order from it — never assume alphabetical.**
